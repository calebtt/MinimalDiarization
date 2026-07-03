using Microsoft.ML.OnnxRuntime;
using NAudio.Wave;
using Serilog;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Threading.Channels;
using MinimalSileroVAD.Core;  // For VadSpeechSegmenterSileroV5
using MinimalEcapaDiarization.Core;  // For EcapaTdnnModel

namespace MinimalDiarizationTest;

public static partial class Algos
{
    // Helper: Compute cosine similarity between two embeddings (for speaker change detection)
    public static float CosineSimilarity(ReadOnlySpan<float> emb1, ReadOnlySpan<float> emb2)
    {
        float dot = 0f, norm1 = 0f, norm2 = 0f;
        for (int i = 0; i < emb1.Length; i++)
        {
            dot += emb1[i] * emb2[i];
            norm1 += emb1[i] * emb1[i];
            norm2 += emb2[i] * emb2[i];
        }
        float n1 = MathF.Sqrt(norm1), n2 = MathF.Sqrt(norm2);
        return n1 > 0 && n2 > 0 ? dot / (n1 * n2) : 0f;
    }

    // Helper: Convert float[] waveform to PCM16 bytes (mono, 16kHz)
    public static byte[] FloatToPcm16(ReadOnlySpan<float> floats)
    {
        var bytes = new byte[floats.Length * 2];
        for (int i = 0; i < floats.Length; i++)
        {
            short s = (short)(Math.Clamp(floats[i], -1f, 1f) * 32767f);
            bytes[i * 2] = (byte)(s & 0xFF);
            bytes[i * 2 + 1] = (byte)((s >> 8) & 0xFF);
        }
        return bytes;
    }

    // Tunable within ECAPA baseline ~0.6-0.8; too low merges distinct speakers together.
    public const float SpeakerMatchThreshold = 0.65f;

    // Incrementally folds a new embedding into a running mean and re-normalizes to unit
    // length (embeddings are L2-normalized, and cosine similarity assumes that).
    public static float[] UpdateRunningMean(float[] existingMean, int existingCount, ReadOnlySpan<float> newEmbedding)
    {
        var updated = new float[existingMean.Length];
        for (int i = 0; i < updated.Length; i++)
            updated[i] = (existingMean[i] * existingCount + newEmbedding[i]) / (existingCount + 1);

        float norm = 0f;
        for (int i = 0; i < updated.Length; i++)
            norm += updated[i] * updated[i];
        norm = MathF.Sqrt(norm);
        if (norm > 1e-8f)
            for (int i = 0; i < updated.Length; i++)
                updated[i] /= norm;

        return updated;
    }
}

// Speaker record: holds ID + running-mean embedding + sample count for clustering.
public record Speaker(int Id, float[] Embedding, int Count = 1);

// Minimal diarization test app: Mic -> VAD segments -> ECAPA embeddings -> multi-speaker clustering.
// Requires: silero_vad.onnx in /models/, ecapa_tdnn.onnx in /models/.
internal static class Program
{
    private const int AudioSampleRate = 16000;
    private const int ChunkDurationMs = 30;  // ~480 samples @16kHz (Silero-friendly)
    private const int ChunkSamples = AudioSampleRate * ChunkDurationMs / 1000;
    private static bool EnableEcho = false;  // Disable for clean testing

    private static double audioTimeSec = 0;  // Running time counter (s)

    // For speaker tracking: List of known speakers
    private static readonly List<Speaker> _knownSpeakers = new();

    // Static model instance for event handler access
    private static EcapaTdnnModel? _ecapaModel;

    private static async Task Main(string[] _)
    {
        Log.Logger = new LoggerConfiguration()
            .WriteTo.Console(outputTemplate: "{Timestamp:HH:mm:ss.fff} [{Level:u3}] {Message:lj}{NewLine}")
            .MinimumLevel.Information()
            .CreateLogger();

        EcapaTdnnModel? ecapaModel = null;
        try
        {
            // Load ECAPA model from local file (mirror Silero setup; embed if preferred)
            var modelPath = Path.Combine("models", "ecapa_tdnn.onnx");
            if (!File.Exists(modelPath))
                throw new FileNotFoundException($"ECAPA model not found at {modelPath}; run export_ecapa_onnx.py to generate.");

            await using var modelFile = File.OpenRead(modelPath);
            ecapaModel = new EcapaTdnnModel(modelFile);
            _ecapaModel = ecapaModel;  // Assign static for event access

            Log.Information("Starting MinimalDiarizationTest");
            Log.Information("EnableEcho: {EnableEcho}", EnableEcho);
            Log.Information("SpeakerMatchThreshold: {Thresh} (tune lower for noisy audio)", Algos.SpeakerMatchThreshold);

            using var segmenter = new VadSpeechSegmenterSileroV5(msPerFrame: ChunkDurationMs);
            segmenter.SentenceBegin += OnSentenceBegin;
            segmenter.SentenceCompleted += OnSentenceCompleted;

            using var cts = new CancellationTokenSource();
            Console.CancelKeyPress += (_, e) => { e.Cancel = true; cts.Cancel(); };
            Log.Information("Press Ctrl+C to stop…");

            // NAudio's WaveInEvent P/Invokes winmm.dll and only works on Windows; use a
            // parec (PipeWire/PulseAudio) subprocess for mic capture everywhere else.
            var micDevice = Environment.GetEnvironmentVariable("MIC_DEVICE") ?? "@DEFAULT_SOURCE@";
            var captureStream = OperatingSystem.IsWindows()
                ? CaptureAndEchoMicrophoneChunksAsync(ChunkSamples, EnableEcho, cts.Token)
                : CaptureMicrophoneChunksViaParecAsync(ChunkSamples, micDevice, cts.Token);

            int chunkCounter = 0;
            await foreach (var rawChunk in captureStream)
            {
                if (cts.Token.IsCancellationRequested) break;
                chunkCounter++;

                ProcessChunk(segmenter, rawChunk, cts.Token, chunkCounter);
            }

            // Final summary
            Log.Information("Diarization Summary ({Count} speakers detected):", _knownSpeakers.Count);
            foreach (var speaker in _knownSpeakers)
                Log.Information("  Speaker {Id}: {Segments} segments (first at inferred time)", speaker.Id, 1 /* Placeholder; track per-ID count if needed */);
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Application error: {ex}", ex.Message);
        }
        finally
        {
            ecapaModel?.Dispose();  // Explicit dispose
            _ecapaModel = null;
            _knownSpeakers.Clear();  // Cleanup
            Log.CloseAndFlush();
        }
    }

    private static void OnSentenceBegin(object? sender, object e)
    {
        Log.Information("*** Speech Begin at {Time:F2}s ***", audioTimeSec);
    }

    private static void OnSentenceCompleted(object? sender, MemoryStream sentencePcm)
    {
        var durationSeconds = sentencePcm.Length / 2f / AudioSampleRate;
        Log.Information("*** Speech Completed at {Time:F2}s — Duration {Dur:F2}s ({Bytes} bytes) ***",
            audioTimeSec, durationSeconds, sentencePcm.Length);

        if (_ecapaModel == null)
        {
            Log.Error("ECAPA model not initialized; skipping embedding extraction.");
            return;
        }

        // Extract embedding
        var pcmSpan = sentencePcm.ToArray().AsSpan();
        var embedding = _ecapaModel.ExtractEmbedding(pcmSpan);

        // Find best match among known speakers (online clustering)
        int assignedId = -1;
        float maxSim = -1f;
        if (_knownSpeakers.Count > 0)
        {
            foreach (var known in _knownSpeakers)
            {
                var sim = Algos.CosineSimilarity(embedding, known.Embedding.AsSpan());
                if (sim > maxSim)
                {
                    maxSim = sim;
                    assignedId = known.Id;
                }
            }

            if (maxSim >= Algos.SpeakerMatchThreshold)
            {
                Log.Information("*** Assigned to Speaker {Id} (max sim={Sim:F3}) ***", assignedId, maxSim);
            }
            else
            {
                // New speaker
                assignedId = _knownSpeakers.Count;  // Next ID
                Log.Information("*** New Speaker {Id} (max sim={Sim:F3} < {Thresh:F3}) ***", assignedId, maxSim, Algos.SpeakerMatchThreshold);
            }
        }
        else
        {
            // First speaker
            assignedId = 0;
            Log.Information("*** New Speaker {Id} (first segment) ***", assignedId);
        }

        // Add/update: fold into a running mean rather than overwriting, so a single noisy
        // or short utterance can't drag a speaker's reference embedding off track.
        var existing = _knownSpeakers.FirstOrDefault(s => s.Id == assignedId);
        if (existing != null)
        {
            _knownSpeakers.Remove(existing);
            var updatedEmbedding = Algos.UpdateRunningMean(existing.Embedding, existing.Count, embedding);
            _knownSpeakers.Add(existing with { Embedding = updatedEmbedding, Count = existing.Count + 1 });
        }
        else
        {
            _knownSpeakers.Add(new Speaker(assignedId, embedding.ToArray()));
        }

        // Optional: Log embedding norm for debugging
        var embNorm = embedding.Aggregate(0f, (sum, x) => sum + x * x);
        Log.Debug("Embedding norm: {Norm:F3}", MathF.Sqrt(embNorm));
    }

    private static void ProcessChunk(VadSpeechSegmenterSileroV5 segmenter, float[] chunk, CancellationToken ct, int chunkCounter)
    {
        float avgAmp = chunk.Average(Math.Abs);
        if (chunkCounter % 20 == 0)  // Less frequent logging
            Log.Information("Chunk #{Chunk} AvgAmp {Amp:F3}", chunkCounter, avgAmp);

        var monoPcm = Algos.FloatToPcm16(chunk);
        segmenter.PushFrame(monoPcm, AudioSampleRate, ChunkDurationMs);
        audioTimeSec += (double)ChunkSamples / AudioSampleRate;
    }

    // Mic capture (mirrors VAD test: NAudio, channel for async yield)
    private static async IAsyncEnumerable<float[]> CaptureAndEchoMicrophoneChunksAsync(
        int chunkSamples, bool enableEcho, [EnumeratorCancellation] CancellationToken ct)
    {
        var channel = Channel.CreateBounded<float[]>(10);
        using var waveIn = new WaveInEvent
        {
            WaveFormat = new WaveFormat(AudioSampleRate, 16, 1),
            BufferMilliseconds = ChunkDurationMs
        };

        waveIn.DeviceNumber = 0;  // Default mic
        var bufferedProvider = enableEcho ? new BufferedWaveProvider(waveIn.WaveFormat) : null;
        WaveOutEvent? waveOut = null;
        if (enableEcho && bufferedProvider != null)
        {
            bufferedProvider.BufferDuration = TimeSpan.FromMilliseconds(200);  // Low latency
            waveOut = new WaveOutEvent();
            waveOut.Init(bufferedProvider);
            waveOut.Play();
        }

        waveIn.DataAvailable += (s, e) =>
        {
            if (ct.IsCancellationRequested) return;
            if (e.BytesRecorded < chunkSamples * 2) return;  // Skip partials

            var chunk = new float[e.BytesRecorded / 2];
            for (int i = 0; i < chunk.Length; i++)
                chunk[i] = BitConverter.ToInt16(e.Buffer, i * 2) / 32768f;

            if (!channel.Writer.TryWrite(chunk))  // Bounded; drop if full (rare)
                Log.Warning("Dropped audio chunk (channel full)");

            bufferedProvider?.AddSamples(e.Buffer, 0, e.BytesRecorded);
        };

        Log.Information("Starting microphone recording…");
        waveIn.StartRecording();

        try
        {
            await foreach (var chunk in channel.Reader.ReadAllAsync(ct))
                yield return chunk;
        }
        finally
        {
            waveIn.StopRecording();
            waveOut?.Stop();
            waveOut?.Dispose();
        }
    }

    // Linux/macOS mic capture: shells out to `parec` (PipeWire/PulseAudio) for raw
    // 16kHz/16-bit/mono PCM instead of NAudio's Windows-only device APIs.
    private static async IAsyncEnumerable<float[]> CaptureMicrophoneChunksViaParecAsync(
        int chunkSamples, string device, [EnumeratorCancellation] CancellationToken ct)
    {
        var psi = new ProcessStartInfo
        {
            FileName = "parec",
            RedirectStandardOutput = true,
            UseShellExecute = false,
        };
        psi.ArgumentList.Add("--raw");
        psi.ArgumentList.Add($"--device={device}");
        psi.ArgumentList.Add($"--rate={AudioSampleRate}");
        psi.ArgumentList.Add("--format=s16le");
        psi.ArgumentList.Add("--channels=1");
        // PipeWire/PulseAudio's default client latency target is ~2000ms, which makes
        // parec buffer audio in ~2s bursts instead of streaming it continuously. Request
        // a short latency so chunks arrive close to real time.
        psi.ArgumentList.Add("--latency-msec=50");

        using var process = Process.Start(psi)
            ?? throw new InvalidOperationException("Failed to start parec for microphone capture.");
        Log.Information("Capturing microphone via parec (device={Device})…", device);

        try
        {
            var stdout = process.StandardOutput.BaseStream;
            int chunkBytes = chunkSamples * 2;
            var buffer = new byte[chunkBytes];

            while (!ct.IsCancellationRequested)
            {
                int read = 0;
                while (read < chunkBytes)
                {
                    int n = await stdout.ReadAsync(buffer.AsMemory(read, chunkBytes - read), ct);
                    if (n == 0)
                        yield break;  // parec exited or stream closed
                    read += n;
                }

                var chunk = new float[chunkSamples];
                for (int i = 0; i < chunkSamples; i++)
                    chunk[i] = BitConverter.ToInt16(buffer, i * 2) / 32768f;

                yield return chunk;
            }
        }
        finally
        {
            if (!process.HasExited)
                process.Kill(entireProcessTree: true);
        }
    }
}
