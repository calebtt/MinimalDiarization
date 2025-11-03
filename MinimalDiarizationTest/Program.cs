using Microsoft.ML.OnnxRuntime;
using MinimalDiarization.Core;
using MinimalSileroVAD.Core;  // For VadSpeechSegmenterSileroV5
using MinimalVadTest;
using NAudio.Wave;
using Serilog;
using System.Runtime.CompilerServices;
using System.Threading.Channels;

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

    public static async Task<string> TranscribeAudio(SttProviderStreaming stt, MemoryStream pcmStream)
    {
        await stt.ProcessAudioChunkAsync(pcmStream);
        var transcript = await stt.WaitForCompleteTranscriptionAsync();
        stt.Dispose();  // RAII
        return transcript ?? string.Empty;
    }

    // Tunable: Lower for noisy envs like movies (ECAPA baseline ~0.6-0.8)
    public const float SpeakerMatchThreshold = 0.4f;
}


internal static class Program
{
    private const int AudioSampleRate = 16000;
    private const int ChunkDurationMs = 30;
    private const int ChunkSamples = AudioSampleRate * ChunkDurationMs / 1000;
    private static bool EnableEcho = false;

    private static double audioTimeSec = 0;

    private static readonly List<Speaker> _knownSpeakers = new();
    private static EcapaTdnnModel? _ecapaModel;
    private static UserEmbedding? _userEmbedding;
    private static readonly List<(double time, float[] embedding, string label)> _history = new();
    private static IntentPreprocessor? _intentPreprocessor;  // New
    private static SttProviderStreaming? _stt;  // Reusable STT

    private static async Task Main(string[] _)
    {
        Log.Logger = new LoggerConfiguration()
            .WriteTo.Console(outputTemplate: "{Timestamp:HH:mm:ss.fff} [{Level:u3}] {Message:lj}{NewLine}")
            .MinimumLevel.Information()
            .CreateLogger();

        EcapaTdnnModel? ecapaModel = null;
        try
        {
            var modelPath = Path.Combine("models", "ecapa_tdnn.onnx");
            if (!File.Exists(modelPath))
                throw new FileNotFoundException($"ECAPA model not found at {modelPath}; run export_ecapa_onnx.py to generate.");

            await using var modelFile = File.OpenRead(modelPath);
            ecapaModel = new EcapaTdnnModel(modelFile);
            _ecapaModel = ecapaModel;

            await EnrollOrLoadVoice(_ecapaModel);

            _stt = new SttProviderStreaming();  // Your model path
            _intentPreprocessor = new IntentPreprocessor();  // LLM init

            Log.Information("Starting MinimalDiarizationTest");
            Log.Information("EnableEcho: {EnableEcho}", EnableEcho);
            Log.Information("SpeakerMatchThreshold: {Thresh}", Algos.SpeakerMatchThreshold);

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
            Log.Information("Diarization Summary ({Count} speakers detected):", _knownSpeakers.Count);
            var master = _knownSpeakers.FirstOrDefault(s => s.Type == SpeakerType.Master);
            Log.Information("  Master (You): {Count} segments", master?.SegmentCount ?? 0);
            foreach (var speaker in _knownSpeakers.Where(s => s.Type == SpeakerType.Other))
                Log.Information("  Other Speaker {Id}: {Count} segments", speaker.Id, speaker.SegmentCount);
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Application error: {ex}", ex.Message);
        }
        finally
        {
            ecapaModel?.Dispose();
            _ecapaModel = null;
            _stt?.Dispose();
            _intentPreprocessor?.Dispose();
            _knownSpeakers.Clear();
            _history.Clear();
            Log.CloseAndFlush();
        }
    }

    private static async Task EnrollOrLoadVoice(EcapaTdnnModel ecapaModel)
    {
        var userEnroller = new UserVoiceEnroller(ecapaModel);

        _userEmbedding = await userEnroller.LoadAsync();
        if (_userEmbedding != null)
        {
            Log.Information("Loaded existing enrollment — Skipping re-enrollment.");
        }
        else
        {
            _userEmbedding = await userEnroller.EnrollAsync();
        }
    }

    private static void OnSentenceBegin(object? sender, object e)
    {
        Log.Information("*** Speech Begin at {Time:F2}s ***", audioTimeSec);
    }

    private static async void OnSentenceCompleted(object? sender, MemoryStream sentencePcm)
    {
        var durationSeconds = sentencePcm.Length / 2f / AudioSampleRate;
        Log.Information("*** Speech Completed at {Time:F2}s — Duration {Dur:F2}s ({Bytes} bytes) ***",
            audioTimeSec, durationSeconds, sentencePcm.Length);

        if (_ecapaModel == null || _stt == null)
        {
            Log.Error("ECAPA or STT not initialized; skipping.");
            return;
        }

        // Extract embedding
        var pcmSpan = sentencePcm.ToArray().AsSpan();
        var embedding = _ecapaModel.ExtractEmbedding(pcmSpan);

        // Master Check (user embedding)
        SpeakerType assignedType = SpeakerType.Other;
        int assignedId = -1;
        float maxSim = -1f;
        if (_userEmbedding != null)
        {
            var userSim = Algos.CosineSimilarity(embedding, _userEmbedding.Embedding.AsSpan());
            if (userSim >= Algos.UserVoiceThreshold)
            {
                assignedType = SpeakerType.Master;
                assignedId = 0;
                maxSim = userSim;
                Log.Information("*** Master (You) Speaking (sim={Sim:F3} ≥ {Thresh:F3}) ***", maxSim, Algos.UserVoiceThreshold);
            }
        }

        // Fallback clustering for others
        if (assignedType == SpeakerType.Other)
        {
            if (_knownSpeakers.Count > 0)
            {
                float otherMaxSim = -1f;
                int otherId = -1;
                foreach (var known in _knownSpeakers.Where(s => s.Type == SpeakerType.Other))
                {
                    var sim = Algos.CosineSimilarity(embedding, known.Embedding.AsSpan());
                    if (sim > otherMaxSim)
                    {
                        otherMaxSim = sim;
                        otherId = known.Id;
                    }
                }
                maxSim = otherMaxSim;

                if (otherMaxSim >= Algos.SpeakerMatchThreshold)
                {
                    assignedId = otherId;
                    Log.Information("*** Assigned to Other Speaker {Id} (sim={Sim:F3}) ***", assignedId, maxSim);
                }
                else
                {
                    assignedId = _knownSpeakers.Where(s => s.Type == SpeakerType.Other).DefaultIfEmpty(new Speaker(SpeakerType.Other, 0, new float[0])).Max(s => s.Id) + 1;
                    Log.Information("*** New Other Speaker {Id} (sim={Sim:F3} < {Thresh:F3}) ***", assignedId, maxSim, Algos.SpeakerMatchThreshold);
                }
            }
            else
            {
                assignedId = 1;
                Log.Information("*** New Other Speaker {Id} (first non-master) ***", assignedId);
            }
        }

        // Update/Add speaker
        var embeddingArray = embedding.ToArray();
        var existing = _knownSpeakers.FirstOrDefault(s => s.Type == assignedType && s.Id == assignedId);
        if (existing != null)
        {
            var updatedCount = existing.SegmentCount + 1;
            _knownSpeakers.Remove(existing);
            _knownSpeakers.Add(new Speaker(assignedType, assignedId, embeddingArray, updatedCount));
        }
        else
        {
            _knownSpeakers.Add(new Speaker(assignedType, assignedId, embeddingArray, 1));
        }

        // Transcription + LLM
        var transcript = await Algos.TranscribeAudio(_stt, sentencePcm);
        if (!string.IsNullOrWhiteSpace(transcript) && _intentPreprocessor != null && IsPotentialCommand(transcript))
        {
            var currentSpeaker = new Speaker(assignedType, assignedId, embeddingArray);
            var (isCommand, respondTo) = _intentPreprocessor.Analyze(transcript, currentSpeaker);
            if (isCommand)
            {
                if (respondTo == "master")
                {
                    Log.Information("*** LLM Routed: Valid Master Command '{Text}' — Agent Activated ***", transcript);
                    await ProcessCommandAsync(transcript);
                }
                else if (respondTo.StartsWith("speaker_"))
                {
                    Log.Information("*** LLM Routed: {Respond} Command '{Text}' ***", respondTo, transcript);
                    // Optional: await HandleOtherCommand(assignedId, transcript);
                }
                else
                {
                    Log.Information("*** LLM: Ignore '{Text}' ({Respond}) ***", transcript, respondTo);
                }
            }
        }
        else if (!string.IsNullOrWhiteSpace(transcript))
        {
            Log.Information("*** Skipped LLM: '{Text}' (short/non-command) ***", transcript[..50]);
        }

        // Optional norm log
        var embNorm = embedding.Aggregate(0f, (sum, x) => sum + x * x);
        Log.Debug("Embedding norm: {Norm:F3}", MathF.Sqrt(embNorm));

        audioTimeSec += durationSeconds;  // Update time
    }

    private static bool IsPotentialCommand(string text)
    {
        var words = text.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        return words.Length > 2 && (text.Contains('?') || text.Contains('!') || words.Any(w => w.StartsWith("set|play|tell|what|how", StringComparison.OrdinalIgnoreCase)));
    }

    private static async Task ProcessCommandAsync(string command)
    {
        Log.Information("Agent Processing: {Command}", command);
        // Integrate your LLM/agent here, e.g., await YourAgent.Execute(command);
        await Task.Delay(100);  // Placeholder
    }

    private static void ProcessChunk(VadSpeechSegmenterSileroV5 segmenter, float[] chunk, CancellationToken ct, int chunkCounter)
    {
        float avgAmp = chunk.Average(Math.Abs);
        if (chunkCounter % 20 == 0)
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

public class IgnoreDisposeStream : Stream
{
    private readonly Stream _inner;
    public IgnoreDisposeStream(Stream inner) => _inner = inner;

    protected override void Dispose(bool disposing) => _inner.Flush();

    public override bool CanRead => _inner.CanRead;
    public override bool CanSeek => _inner.CanSeek;
    public override bool CanWrite => _inner.CanWrite;
    public override long Length => _inner.Length;
    public override long Position { get => _inner.Position; set => _inner.Position = value; }

    public override void Flush() => _inner.Flush();
    public override int Read(byte[] b, int o, int c) => _inner.Read(b, o, c);
    public override long Seek(long o, SeekOrigin so) => _inner.Seek(o, so);
    public override void SetLength(long v) => _inner.SetLength(v);
    public override void Write(byte[] b, int o, int c) => _inner.Write(b, o, c);
    public override Task<int> ReadAsync(byte[] b, int o, int c, CancellationToken ct) => _inner.ReadAsync(b, o, c, ct);
    public override Task WriteAsync(byte[] b, int o, int c, CancellationToken ct) => _inner.WriteAsync(b, o, c, ct);
    public override Task FlushAsync(CancellationToken ct) => _inner.FlushAsync(ct);
}