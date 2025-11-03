using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Numerics;
using Serilog;

namespace MinimalDiarization.Core;

public class EcapaTdnnModel : IDisposable
{
    private readonly InferenceSession _session;
    private const int SampleRate = 16000;
    private const int Nfft = 1024;
    private const int WinLength = 1024;
    private const int HopLength = 256;
    private const int NMels = 80;
    private const float Fmin = 0f;
    private const float Fmax = 8000f;
    private const float ClipVal = 1e-5f;
    private readonly float[] _melBasis;  // Flattened [freqs * mels] row-major
    private readonly float _threshold;   // Placeholder
    private bool _isDisposed;

    public const int EmbeddingDim = 192;

    public EcapaTdnnModel(Stream modelStream, float threshold = 0.5f)
    {
        ArgumentNullException.ThrowIfNull(modelStream, nameof(modelStream));
        if (!modelStream.CanRead)
            throw new ArgumentException("Model stream must be readable.", nameof(modelStream));

        try
        {
            var opts = new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            };
            opts.AppendExecutionProvider_CUDA(); // CUDA if available

            using var memoryStream = new MemoryStream();
            modelStream.CopyTo(memoryStream);
            memoryStream.Position = 0;
            _session = new InferenceSession(memoryStream.ToArray(), opts);
            Log.Information("ECAPA-TDNN model loaded successfully from stream.");
        }
        catch (OnnxRuntimeException ex)
        {
            Log.Error(ex, "Failed to load ONNX model from stream.");
            throw;
        }

        _threshold = threshold;
        _melBasis = ComputeMelBasis();
        Log.Information("ECAPA-TDNN model initialized with Slaney mel filterbank.");
    }

    public float[] ExtractEmbedding(ReadOnlySpan<byte> pcm16)
    {
        if (pcm16.Length % 2 != 0)
            throw new ArgumentException("PCM16 data must have even length.");
        int numSamples = pcm16.Length / 2;
        if (numSamples < SampleRate / 10)  // Min ~0.1s
            throw new ArgumentException("Audio too short for reliable embedding.");

        // Decode to float32 [-1,1]
        Span<float> waveform = stackalloc float[numSamples];
        for (int i = 0; i < numSamples; i++)
            waveform[i] = BitConverter.ToInt16(pcm16[(i * 2)..]) / 32768f;

        // Compute mel-spectrogram (amplitude, log-compressed)
        var melSpec = ComputeMelSpectrogram(waveform);

        // Apply sentence-level mean normalization (no std)
        ApplySentenceNormalization(melSpec);

        // Flatten for ONNX
        int numFrames = melSpec.GetLength(0);
        float[] flatMel = new float[numFrames * NMels];
        for (int t = 0; t < numFrames; t++)
            for (int m = 0; m < NMels; m++)
                flatMel[t * NMels + m] = melSpec[t, m];

        // ONNX inference
        var inputs = new[] { NamedOnnxValue.CreateFromTensor("input", new DenseTensor<float>(flatMel, new[] { 1, numFrames, NMels })) };
        using var result = _session.Run(inputs);
        var outputTensor = result.First(r => r.Name == "embedding").AsTensor<float>();
        float[] embRaw = outputTensor.ToArray();
        Span<float> embedding = stackalloc float[EmbeddingDim];
        for (int i = 0; i < EmbeddingDim; i++)
            embedding[i] = embRaw[i];  // Flatten [1,1,192] -> [192]

        // L2 normalize
        NormalizeL2(embedding);

        return embedding.ToArray();
    }

    private float[,] ComputeMelSpectrogram(ReadOnlySpan<float> waveform)
    {
        int numFreqs = Nfft / 2 + 1;
        int numFrames = (waveform.Length + HopLength - 1) / HopLength;
        var powerSpec = new float[numFrames, numFreqs];  // Amplitude spectrum

        // Hann window
        Span<float> window = stackalloc float[WinLength];
        for (int i = 0; i < WinLength; i++)
            window[i] = 0.5f * (1f - MathF.Cos(2f * MathF.PI * i / (WinLength - 1)));

        // Reusable buffers (moved outside loop to avoid stack overflow)
        Span<float> frameData = stackalloc float[WinLength];
        Span<Complex> fftIn = stackalloc Complex[Nfft];

        // STFT frames
        for (int frame = 0; frame < numFrames; frame++)
        {
            int start = frame * HopLength;
            int end = Math.Min(start + WinLength, waveform.Length);
            int copyLen = end - start;
            waveform.Slice(start, copyLen).CopyTo(frameData[..copyLen]);
            frameData[copyLen..].Fill(0f);  // Zero-pad

            // Window
            for (int i = 0; i < WinLength; i++)
                frameData[i] *= window[i];

            // Real FFT
            for (int i = 0; i < WinLength; i++)
                fftIn[i] = new Complex(frameData[i], 0f);
            for (int i = WinLength; i < Nfft; i++)
                fftIn[i] = Complex.Zero;

            Fft(fftIn);

            // Amplitude spectrum (power=1)
            for (int k = 0; k < numFreqs; k++)
                powerSpec[frame, k] = (float)fftIn[k].Magnitude;
        }

        // Mel filterbank (Slaney-normalized)
        var melSpec = new float[numFrames, NMels];
        for (int frame = 0; frame < numFrames; frame++)
        {
            for (int m = 0; m < NMels; m++)
            {
                float sum = 0f;
                for (int k = 0; k < numFreqs; k++)
                    sum += powerSpec[frame, k] * _melBasis[k * NMels + m];
                melSpec[frame, m] = sum;
            }
        }

        // Log compression
        for (int frame = 0; frame < numFrames; frame++)
        {
            for (int m = 0; m < NMels; m++)
                melSpec[frame, m] = MathF.Log(MathF.Max(melSpec[frame, m], ClipVal));
        }

        return melSpec;
    }

    private void ApplySentenceNormalization(float[,] melSpec)
    {
        int numFrames = melSpec.GetLength(0);
        for (int m = 0; m < NMels; m++)
        {
            float mean = 0f;
            for (int t = 0; t < numFrames; t++)
                mean += melSpec[t, m];
            mean /= numFrames;
            for (int t = 0; t < numFrames; t++)
                melSpec[t, m] -= mean;
        }
    }

    private static void Fft(Span<Complex> data)
    {
        int n = data.Length;
        int logN = (int)MathF.Log2(n);
        if ((1 << logN) != n) throw new ArgumentException("FFT length must be power of 2.");

        // Bit-reversal
        for (int i = 0; i < n; i++)
        {
            int rev = ReverseBits(i, logN);
            if (i < rev) (data[i], data[rev]) = (data[rev], data[i]);
        }

        // Butterflies
        for (int stage = 1; stage <= logN; stage++)
        {
            int blockSize = 1 << stage;
            int half = blockSize / 2;
            float theta = -2f * MathF.PI / blockSize;
            var wk = new Complex(MathF.Cos(theta), MathF.Sin(theta));
            for (int block = 0; block < n; block += blockSize)
            {
                var twiddle = Complex.One;
                for (int k = 0; k < half; k++)
                {
                    int even = block + k;
                    int odd = even + half;
                    var temp = data[odd] * twiddle;
                    data[odd] = data[even] - temp;
                    data[even] += temp;
                    twiddle *= wk;
                }
            }
        }
    }

    private static int ReverseBits(int n, int bits)
    {
        int rev = 0;
        for (int i = 0; i < bits; i++)
        {
            rev = (rev << 1) | (n & 1);
            n >>= 1;
        }
        return rev;
    }

    private float[] ComputeMelBasis()
    {
        float nyquist = SampleRate / 2f;
        int numFreqs = Nfft / 2 + 1;

        // Evenly spaced mel points
        float minMel = HzToMel(Fmin);
        float maxMel = HzToMel(Fmax);
        Span<float> melPoints = stackalloc float[NMels + 2];
        float step = (maxMel - minMel) / (NMels + 1);
        for (int i = 0; i < NMels + 2; i++)
            melPoints[i] = minMel + i * step;

        Span<float> hzPoints = stackalloc float[NMels + 2];
        for (int i = 0; i < NMels + 2; i++)
            hzPoints[i] = MelToHz(melPoints[i]);
        hzPoints[0] = 0f;
        hzPoints[^1] = nyquist;

        // FFT freqs
        Span<float> fftFreqs = stackalloc float[numFreqs];
        for (int i = 0; i < numFreqs; i++)
            fftFreqs[i] = i * nyquist / (numFreqs - 1);

        // Triangular basis [freqs, mels]
        var basis = new float[numFreqs, NMels];
        for (int m = 0; m < NMels; m++)
        {
            float fLow = hzPoints[m];
            float fCen = hzPoints[m + 1];
            float fHigh = hzPoints[m + 2];

            for (int k = 0; k < numFreqs; k++)
            {
                float freq = fftFreqs[k];
                if (freq >= fLow && freq <= fCen)
                    basis[k, m] = (freq - fLow) / (fCen - fLow);
                else if (freq > fCen && freq <= fHigh)
                    basis[k, m] = (fHigh - freq) / (fHigh - fCen);
                else
                    basis[k, m] = 0f;
            }
        }

        // Slaney normalization: divide by mel band width
        for (int m = 0; m < NMels; m++)
        {
            float melWidth = melPoints[m + 1] - melPoints[m];
            for (int k = 0; k < numFreqs; k++)
                basis[k, m] /= melWidth;
        }

        // Flatten row-major [freq * mels]
        var flat = new float[numFreqs * NMels];
        for (int k = 0; k < numFreqs; k++)
            for (int m = 0; m < NMels; m++)
                flat[k * NMels + m] = basis[k, m];
        return flat;
    }

    private static float HzToMel(float hz) => 2595f * MathF.Log10(1f + hz / 700f);
    private static float MelToHz(float mel) => 700f * ((float)Math.Pow(10, mel / 2595f) - 1f);

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

    public void Dispose()
    {
        if (!_isDisposed)
        {
            _session?.Dispose();
            _isDisposed = true;
            Log.Information("EcapaTdnnModel disposed.");
        }
    }
}