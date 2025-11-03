using Microsoft.ML.OnnxRuntime;
using NAudio.Wave;
using Serilog;
using System.Text.Json;
using System.Threading.Channels;
using MinimalDiarization.Core;  // For EcapaTdnnModel

namespace MinimalDiarizationTest;

public static partial class Algos
{
    // Threshold for user voice match (tune: 0.6-0.8; higher = stricter)
    public const float UserVoiceThreshold = 0.7f;
    // File paths (app dir; portable)
    public const string EnrollmentAudioPath = "user_enrollment.wav";
    public const string EnrollmentEmbeddingPath = "user_embedding.json";

    // Validate embedding JSON (simple size check)
    public static bool IsValidEmbeddingFile(string path)
    {
        if (!File.Exists(path)) return false;
        var json = File.ReadAllText(path);
        return json.Length > 100;  // Rough: ~192 floats as JSON >100 chars
    }

    // Average multiple embeddings (for robust enrollment)
    public static float[] AverageEmbeddings(IEnumerable<float[]> embeddings)
    {
        var avg = new float[EcapaTdnnModel.EmbeddingDim];
        int count = 0;
        foreach (var emb in embeddings)
        {
            for (int i = 0; i < avg.Length; i++)
                avg[i] += emb[i];
            count++;
        }
        if (count > 0)
        {
            for (int i = 0; i < avg.Length; i++)
                avg[i] /= count;
            // Re-normalize L2 after averaging
            var span = avg.AsSpan();
            NormalizeL2(span);
        }
        return avg;
    }

    private static void NormalizeL2(Span<float> vec)
    {
        float norm = 0f;
        for (int i = 0; i < vec.Length; i++)
            norm += vec[i] * vec[i];
        norm = MathF.Sqrt(norm);
        if (norm > 1e-8f)
            for (int i = 0; i < vec.Length; i++)
                vec[i] /= norm;
    }
}

// User embedding: Stores averaged reference for verification
public class UserEmbedding
{
    public float[] Embedding { get; }
    public UserEmbedding(float[] embedding) => Embedding = embedding.ToArray();  // Deep copy
}


public class UserVoiceEnroller
{
    private readonly EcapaTdnnModel _ecapaModel;
    private readonly string _audioPath;
    private readonly string _embeddingPath;
    private const int AudioSampleRate = 16000;
    private const int ChunkDurationMs = 30;
    private const int MinEnrollmentSamples = AudioSampleRate / 5;  // ~0.2s

    public UserVoiceEnroller(EcapaTdnnModel ecapaModel, string audioPath = "user_enrollment.wav", string embeddingPath = "user_embedding.json")
    {
        _ecapaModel = ecapaModel ?? throw new ArgumentNullException(nameof(ecapaModel));
        _audioPath = audioPath;
        _embeddingPath = embeddingPath;
    }

    /// <summary>
    /// Enrolls user voice: Records via mic, computes embedding, saves WAV + JSON (overwrites existing).
    /// Returns UserEmbedding if successful; null on failure/short recording.
    /// </summary>
    public async Task<UserEmbedding?> EnrollAsync(CancellationToken cancellationToken = default)
    {
        Log.Information("=== ENROLLMENT PHASE ===");
        Log.Information("Read this paragraph aloud clearly (recording stops after ~10s or Ctrl+C):");
        Log.Information("\"The quick brown fox jumps over the lazy dog. It was a bright cold day in April, and the clocks were striking thirteen.\"");

        var enrollmentChunks = new List<float[]>();
        var channel = Channel.CreateUnbounded<float[]>();
        using var waveIn = new WaveInEvent { WaveFormat = new WaveFormat(AudioSampleRate, 16, 1), BufferMilliseconds = ChunkDurationMs };
        waveIn.DeviceNumber = 0;
        waveIn.DataAvailable += (s, e) =>
        {
            if (cancellationToken.IsCancellationRequested) return;
            var chunk = new float[e.BytesRecorded / 2];
            for (int i = 0; i < chunk.Length; i++)
                chunk[i] = BitConverter.ToInt16(e.Buffer, i * 2) / 32768f;
            channel.Writer.TryWrite(chunk);
        };

        try
        {
            waveIn.StartRecording();
            Log.Information("Recording... Speak now!");

            var enrollCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            enrollCts.CancelAfter(TimeSpan.FromSeconds(10));

            await foreach (var chunk in channel.Reader.ReadAllAsync(enrollCts.Token))
            {
                enrollmentChunks.Add(chunk);
            }
        }
        catch (OperationCanceledException)
        {
            Log.Information("Enrollment stopped (timeout/Ctrl+C).");
        }
        finally
        {
            waveIn.StopRecording();
            channel.Writer.Complete();
        }

        if (enrollmentChunks.Sum(c => c.Length) < MinEnrollmentSamples)
        {
            Log.Warning("Enrollment too short (<0.2s); no files saved.");
            return null;
        }

        // Concat to full waveform
        var fullWave = enrollmentChunks.SelectMany(c => c).ToArray();
        var pcmBytes = Algos.FloatToPcm16(fullWave.AsSpan());

        // Compute embedding
        var embeddings = new List<float[]> { _ecapaModel.ExtractEmbedding(pcmBytes.AsSpan()) };
        var userEmb = Algos.AverageEmbeddings(embeddings);

        // Save audio as WAV (in-memory first, direct overwrite)
        await using var wavMs = new MemoryStream();
        using var writer = new WaveFileWriter(new IgnoreDisposeStream(wavMs), new WaveFormat(AudioSampleRate, 16, 1));
        writer.Write(pcmBytes, 0, pcmBytes.Length);
        writer.Flush();  // Ensure header/buffer flushed
        wavMs.Position = 0;

        await using var audioFile = new FileStream(_audioPath, FileMode.Create, FileAccess.Write, FileShare.None);  // Overwrite existing
        await wavMs.CopyToAsync(audioFile);
        audioFile.Flush(true);  // Sync to disk
        GC.Collect();  // Force handle release (empirical for locks)
        Log.Information("Saved enrollment audio to {Path} ({Size} bytes).", _audioPath, new FileInfo(_audioPath).Length);

        // Save embedding as JSON (direct overwrite)
        var embJson = JsonSerializer.Serialize(userEmb);
        await File.WriteAllTextAsync(_embeddingPath, embJson);
        GC.Collect();  // Force release
        Log.Information("Saved user embedding to {Path}.", _embeddingPath);

        Log.Information("Enrollment complete! Files overwritten for this run.");
        return new UserEmbedding(userEmb);
    }

    /// <summary>
    /// Loads existing embedding from JSON; validates and returns UserEmbedding or null.
    /// Cleans up invalid files.
    /// </summary>
    public async Task<UserEmbedding?> LoadAsync()
    {
        if (!Algos.IsValidEmbeddingFile(_embeddingPath) || !File.Exists(_audioPath))
        {
            Log.Information("No valid enrollment files found.");
            return null;
        }

        try
        {
            var json = await File.ReadAllTextAsync(_embeddingPath);
            var embArray = JsonSerializer.Deserialize<float[]>(json, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
            if (embArray?.Length == EcapaTdnnModel.EmbeddingDim)
            {
                Log.Information("Embedding loaded successfully ({Dim} dims).", embArray.Length);
                return new UserEmbedding(embArray);
            }
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Failed to load embedding; cleaning up.");
        }

        // Cleanup on failure
        try { File.Delete(_embeddingPath); } catch { }
        try { File.Delete(_audioPath); } catch { }
        return null;
    }

}