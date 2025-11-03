using Microsoft.ML.OnnxRuntime;
using NAudio.Wave;
using Serilog;
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

    // Simple clustering threshold (e.g., >0.8 same speaker)
    public const float SpeakerChangeThreshold = 0.75f;
}

// Minimal diarization test app: Mic -> VAD segments -> ECAPA embeddings -> similarity logging.
// Requires: silero_vad.onnx in /models/, ecapa_tdnn.onnx in /models/.
internal static class Program
{
    private const int AudioSampleRate = 16000;
    private const int ChunkDurationMs = 30;  // ~480 samples @16kHz (Silero-friendly)
    private const int ChunkSamples = AudioSampleRate * ChunkDurationMs / 1000;
    private static bool EnableEcho = false;  // Disable for clean testing

    private static double audioTimeSec = 0;  // Running time counter (s)

    // For speaker tracking: last embedding and speaker ID
    private static float[]? _lastEmbedding;
    private static int _currentSpeakerId = 0;
    private static readonly List<(double time, float[] embedding, int speakerId)> _speakerHistory = new();

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

            using var segmenter = new VadSpeechSegmenterSileroV5(msPerFrame: ChunkDurationMs);
            segmenter.SentenceBegin += OnSentenceBegin;
            segmenter.SentenceCompleted += OnSentenceCompleted;

            using var cts = new CancellationTokenSource();
            Console.CancelKeyPress += (_, e) => { e.Cancel = true; cts.Cancel(); };
            Log.Information("Press Ctrl+C to stop…");

            int chunkCounter = 0;
            await foreach (var rawChunk in CaptureAndEchoMicrophoneChunksAsync(ChunkSamples, EnableEcho, cts.Token))
            {
                if (cts.Token.IsCancellationRequested) break;
                chunkCounter++;

                ProcessChunk(segmenter, rawChunk, cts.Token, chunkCounter);
            }

            // Final summary
            Log.Information("Diarization Summary:");
            foreach (var (time, _, sid) in _speakerHistory)
                Log.Information("  Speaker {Sid} at {Time:F2}s", sid, time);
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Application error: {ex}", ex.Message);
        }
        finally
        {
            ecapaModel?.Dispose();  // Explicit dispose
            _ecapaModel = null;
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

        // Detect speaker change via cosine sim
        if (_lastEmbedding != null)
        {
            var sim = Algos.CosineSimilarity(embedding, _lastEmbedding);
            var isSameSpeaker = sim > Algos.SpeakerChangeThreshold;
            if (!isSameSpeaker)
            {
                _currentSpeakerId++;
                Log.Information("*** Speaker Change Detected (sim={Sim:F3}) → Speaker {_currentSpeakerId} ***", sim, _currentSpeakerId);
            }
            else
            {
                Log.Information("*** Same Speaker Continued (sim={Sim:F3}) → Speaker {_currentSpeakerId} ***", sim, _currentSpeakerId);
            }
        }
        else
        {
            Log.Information("*** New Speaker {_currentSpeakerId} (first segment) ***", _currentSpeakerId);
        }

        // Track history
        _speakerHistory.Add((audioTimeSec, embedding.ToArray(), _currentSpeakerId));
        _lastEmbedding = embedding.ToArray();  // Retain for next

        // Optional: Log partial embedding stats (e.g., norm check)
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
}