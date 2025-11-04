using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.Onnx;
using Microsoft.Extensions.AI;
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

    // Tunable: Lower for noisy envs like movies (ECAPA baseline ~0.6-0.8)
    public const float SpeakerMatchThreshold = 0.6f;
}


internal static class Program
{
    private const int AudioSampleRate = 16000;
    private const int ChunkDurationMs = 30;
    private const int ChunkSamples = AudioSampleRate * ChunkDurationMs / 1000;
    private static bool EnableEcho = false;

    private static double audioTimeSec = 0;

    private static SpeakerTracker? _speakerTracker;
    private static EcapaTdnnModel? _ecapaModel;
    private static UserEmbedding? _userEmbedding;
    private static CommandIntentClassifier? _intentPreprocessor;
    private static SttProviderStreaming? _sttProvider;
    private static CancellationTokenSource _cts = new();

    private static async Task Main(string[] _)
    {
        Log.Logger = new LoggerConfiguration()
            .WriteTo.Console(outputTemplate: "{Timestamp:HH:mm:ss.fff} [{Level:u3}] {Message:lj}{NewLine}")
            .MinimumLevel.Information()
            .CreateLogger();

        var modelFolder = Path.Combine(AppContext.BaseDirectory, "models", "phi3-mini");
        var builder = Kernel.CreateBuilder();

        // Now works with genai_config.json present
        builder.Services.AddOnnxRuntimeGenAIChatCompletion(
            modelId: "phi3-mini",
            modelPath: modelFolder,  // Folder path—loads genai_config.json automatically
            serviceId: "phi3-onnx"
        );

        var kernel = builder.Build();
        var chatService = kernel.GetRequiredService<IChatCompletionService>();
        _intentPreprocessor = new CommandIntentClassifier(chatService);

        await TestIntentPreprocessor(_intentPreprocessor);

        Console.ReadKey();

        try
        {
            var modelPath = Path.Combine("models", "ecapa_tdnn.onnx");
            if (!File.Exists(modelPath))
                throw new FileNotFoundException($"ECAPA model not found at {modelPath}; run export_ecapa_onnx.py to generate.");

            await using var modelFile = File.OpenRead(modelPath);
            _ecapaModel = new EcapaTdnnModel(modelFile);
            _speakerTracker = new SpeakerTracker(_ecapaModel);
            _sttProvider = new SttProviderStreaming();

            await EnrollOrLoadVoice(_ecapaModel);

            Log.Information("Starting MinimalDiarizationTest");
            Log.Information("EnableEcho: {EnableEcho}", EnableEcho);
            Log.Information("SpeakerMatchThreshold: {Thresh}", Algos.SpeakerMatchThreshold);

            using var segmenter = new VadSpeechSegmenterSileroV5(msPerFrame: ChunkDurationMs);
            segmenter.SentenceBegin += OnSentenceBegin;
            segmenter.SentenceCompleted += OnSentenceCompleted;

            Console.CancelKeyPress += (_, e) => { e.Cancel = true; _cts.Cancel(); };
            Log.Information("Starting microphone capture... (Ctrl+C to stop)");

            int chunkCounter = 0;
            await foreach (var chunk in CaptureAndEchoMicrophoneChunksAsync(ChunkSamples, EnableEcho, _cts.Token))
            {
                ProcessChunk(segmenter, chunk, _cts.Token, chunkCounter++);
            }
        }
        catch (OperationCanceledException) { Log.Information("Capture stopped (Ctrl+C)."); }
        catch (Exception ex) { Log.Error(ex, "Application error"); }
        finally
        {
            _speakerTracker?.Dispose();
            _ecapaModel?.Dispose();
        }
    }

    private static async Task TestIntentPreprocessor(CommandIntentClassifier intent)
    {
        var tests = new[]
        {
        ("Hey computer, turn on the lights.", true),
        ("What's the weather?", true),
        ("Play music.", true),
        ("Who are you?", false),
        ("I'll make sure you feel confident the whole way through.", false),
        ("Please open the pod bay doors.", true),
        ("Can you tell me a joke?", true),
        ("This is a random sentence.", false),
        ("Computer, initiate self-destruct sequence.", true),
        ("I love programming.", false),
        ("There's a snake in my boot.", false)
    };

        foreach (var (cmd, expected) in tests)
        {
            var speaker = cmd.Contains("master") ? new Speaker(SpeakerType.Master, 0, new float[192])
                         : new Speaker(SpeakerType.Other, 1, new float[192]);

            bool isCmd = await intent.IsCommandAsync(cmd, speaker);
            Log.Information("Test: \"{Cmd}\" -> {Result} (expected: {Exp})", cmd, isCmd ? "YES" : "NO", expected ? "YES" : "NO");
        }
    }

    private static async Task EnrollOrLoadVoice(EcapaTdnnModel ecapaModel)
    {
        var enroller = new UserVoiceEnroller(ecapaModel);
        _userEmbedding = await enroller.LoadAsync();
        if (_userEmbedding == null)
        {
            Log.Information("IP: No enrollment found. Starting enrollment...");
            _userEmbedding = await enroller.EnrollAsync(_cts.Token);
            if (_userEmbedding == null)
            {
                Log.Warning("IP: Enrollment failed; skipping user verification.");
                return;
            }
        }
        else
        {
            Log.Information("IP: Loaded existing enrollment - Skipping re-enrollment.");
        }

        // Register the master with the tracker
        _speakerTracker!.RegisterMaster(_userEmbedding.Embedding);
    }

    private static void OnSentenceBegin(object? sender, EventArgs e)
    {
        // startSec is not available in EventArgs; we'll use audioTimeSec as approximation
        Log.Information("IP: *** Speech Begin at {Start}s ***", audioTimeSec);
    }

    private static void OnSentenceCompleted(object? sender, MemoryStream pcmStream)
    {
        // Fire and forget async
        _ = OnSentenceCompletedAsync(pcmStream, _cts.Token);
    }

    private static async Task OnSentenceCompletedAsync(MemoryStream pcmStream, CancellationToken ct)
    {
        var pcmBytes = pcmStream.ToArray();
        var endSec = audioTimeSec;
        var dur = (double)pcmBytes.Length / (AudioSampleRate * 2);
        var startSec = endSec - dur;

        Log.Information("IP: *** Speech Completed at {End}s - Duration {Dur}s ({Bytes} bytes) ***", endSec, dur, pcmBytes.Length);

        if (pcmBytes.Length < AudioSampleRate * 1) // <1 s
        {
            Log.Information("IP: Skipping short segment (<1s).");
            return;
        }

        // ---- Embedding ------------------------------------------------
        float[] embedding = _ecapaModel!.ExtractEmbedding(pcmBytes.AsSpan());

        // ---- Speaker assignment ---------------------------------------
        Speaker assigned = _speakerTracker!.AssignSpeaker(embedding, startSec);

        // ---- STT ------------------------------------------------------
        pcmStream.Position = 0;
        await _sttProvider!.ProcessAudioChunkAsync(pcmStream);
        var transcript = await _sttProvider.WaitForCompleteTranscriptionAsync(ct);

        if (string.IsNullOrWhiteSpace(transcript))
        {
            //Log.Information("Streaming STT: Processing failed (empty transcript).");
            return;
        }

        // ---- Intent (runs on **every** transcription) ----------------
        var isCommand = await _intentPreprocessor!.IsCommandAsync(transcript, assigned);
        Log.Information("IP: Intent Analysis: isCommand={IsCommand}, respondTo={RespondTo}", isCommand);
        if (isCommand)
        {
            await ProcessCommandAsync(transcript);
        }
    }

    private static async Task ProcessCommandAsync(string command)
    {
        Log.Information("IP: Agent Processing: {Command}", command);
        // Integrate your LLM/agent here, e.g., await YourAgent.Execute(command);
        await Task.Delay(100);  // Placeholder
    }

    private static void ProcessChunk(VadSpeechSegmenterSileroV5 segmenter, float[] chunk, CancellationToken ct, int chunkCounter)
    {
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