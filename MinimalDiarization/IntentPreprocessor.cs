using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.Tokenizers;
using MinimalDiarization.Core;
using Serilog;
using System.IO;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace MinimalDiarization.Core;

public class IntentPreprocessor : IDisposable
{
    private readonly InferenceSession _session;
    private readonly BpeTokenizer _tokenizer;
    private const int MaxLength = 128;
    private const string ModelPath = "models/dialogpt-small-onnx.onnx";
    private const string TokenizerPath = "models/tokenizer.json";
    private const int NumLayers = 12;  // DialoGPT-small has 12 layers
    private const int NumHeads = 12;
    private const int HeadSize = 64;  // hidden_size / num_heads = 768 / 12 = 64

    public IntentPreprocessor()
    {
        var opts = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED };
        opts.AppendExecutionProvider_CUDA();
        _session = new InferenceSession(ModelPath, opts);

        _tokenizer = BuildBpeFromJson(TokenizerPath);
        Log.Information("IntentPreprocessor (DialoGPT-small ONNX) initialized: {VocabSize} tokens.", _tokenizer.Vocabulary.Count);
    }

    private static BpeTokenizer BuildBpeFromJson(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        if (!File.Exists(path))
            throw new FileNotFoundException($"Download tokenizer.json from https://huggingface.co/masimka/DialoGPT-small-ONNX/resolve/main/tokenizer.json to {path}");

        var json = JsonNode.Parse(File.ReadAllText(path)) ?? throw new JsonException("Invalid tokenizer.json");

        // Extract vocab (model.vocab as dict -> Dictionary<string, int>)
        var vocabNode = json["model"]?["vocab"] as JsonObject;
        ArgumentNullException.ThrowIfNull(vocabNode, "Missing 'model.vocab'");
        var vocabDict = new Dictionary<string, int>();
        foreach (var kvp in vocabNode)
        {
            var key = kvp.Key ?? throw new JsonException("Null vocab key");
            vocabDict[key] = kvp.Value!.GetValue<int>();
        }
        vocabDict["<|endoftext|>"] = 50256;  // Add special EOS token

        // Extract merges (model.merges as array of arrays -> IEnumerable<string>)
        var mergesNode = json["model"]?["merges"] as JsonArray;
        ArgumentNullException.ThrowIfNull(mergesNode, "Missing 'model.merges'");
        var mergesList = new List<string>();
        foreach (var n in mergesNode)
        {
            if (n is JsonArray pair && pair.Count == 2)
            {
                var left = pair[0]!.GetValue<string>();
                var right = pair[1]!.GetValue<string>();
                mergesList.Add($"{left} {right}");
            }
            else
            {
                Log.Warning("Skipping invalid merge entry: {Entry}", n?.ToJsonString());
            }
        }

        var options = new BpeOptions(vocabDict)
        {
            Merges = mergesList,
            UnknownToken = "<|endoftext|>",
            FuseUnknownTokens = false
        };

        var tokenizer = BpeTokenizer.Create(options);
        if (tokenizer.Vocabulary.Count < 50000)
            throw new InvalidOperationException("Vocab too small; verify tokenizer.json.");

        Log.Debug("Bpe built: {VocabSize} tokens, {Merges} merges.", tokenizer.Vocabulary.Count, mergesList.Count);
        return tokenizer;
    }

    public (bool isCommand, string respondTo) Analyze(string transcript, Speaker speaker)
    {
        var prompt = BuildPrompt(transcript, speaker);
        var inputTokens = _tokenizer.EncodeToIds(prompt).ToArray();
        var outputTokens = new List<int>(inputTokens); // Start with prompt

        if (!_tokenizer.Vocabulary.TryGetValue("<|endoftext|>", out int eosTokenId))
            throw new InvalidOperationException("EOS token '<|endoftext|>' not in vocabulary.");

        List<NamedOnnxValue>? pastKeyValues = null;
        const int maxNewTokens = 64; // Prevent infinite generation
        int generatedCount = 0;

        while (outputTokens.Count < MaxLength && generatedCount < maxNewTokens)
        {
            var currentLength = outputTokens.Count;

            // Input tensors
            var inputIds = outputTokens.Select(t => (long)t).ToArray();
            var inputTensor = new DenseTensor<long>(inputIds, new[] { 1, currentLength });

            var positionIds = Enumerable.Range(0, currentLength).Select(i => (long)i).ToArray();
            var positionTensor = new DenseTensor<long>(positionIds, new[] { 1, currentLength });

            var attentionMask = Enumerable.Repeat(1L, currentLength).ToArray();
            var attentionTensor = new DenseTensor<long>(attentionMask, new[] { 1, currentLength });

            var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", inputTensor),
            NamedOnnxValue.CreateFromTensor("position_ids", positionTensor),
            NamedOnnxValue.CreateFromTensor("attention_mask", attentionTensor)
        };

            // Add past key values (or empty for first step)
            if (pastKeyValues != null)
            {
                inputs.AddRange(pastKeyValues);
            }
            else
            {
                for (int layer = 0; layer < NumLayers; layer++)
                {
                    var empty = new DenseTensor<float>(new float[0], new[] { 1, NumHeads, 0, HeadSize });
                    inputs.Add(NamedOnnxValue.CreateFromTensor($"past_key_values.{layer}.key", empty));
                    inputs.Add(NamedOnnxValue.CreateFromTensor($"past_key_values.{layer}.value", empty));
                }
            }

            using var result = _session.Run(inputs);
            var logitsTensor = result.First(r => r.Name == "logits").AsTensor<float>();

            // Get logits for the *last* position
            var lastLogits = new float[logitsTensor.Dimensions[2]];
            for (int i = 0; i < lastLogits.Length; i++)
            {
                lastLogits[i] = logitsTensor[0, currentLength - 1, i];
            }

            int nextToken = ArgMaxFloatSpan(lastLogits);

            // Stop if EOS
            if (nextToken == eosTokenId) break;

            outputTokens.Add(nextToken);
            generatedCount++;

            // Update past_key_values for next iteration
            pastKeyValues = new List<NamedOnnxValue>();
            for (int layer = 0; layer < NumLayers; layer++)
            {
                var key = result.First(r => r.Name == $"present_key_values.{layer}.key").AsTensor<float>();
                var value = result.First(r => r.Name == $"present_key_values.{layer}.value").AsTensor<float>();
                pastKeyValues.Add(NamedOnnxValue.CreateFromTensor($"past_key_values.{layer}.key", key));
                pastKeyValues.Add(NamedOnnxValue.CreateFromTensor($"past_key_values.{layer}.value", value));
            }
        }

        // Only decode newly generated tokens
        var generatedTokens = outputTokens.Skip(inputTokens.Length);
        var decoded = _tokenizer.Decode(generatedTokens).Trim();

        Log.Debug("Prompt: {Prompt}", prompt);
        Log.Debug("Generated tokens: {Tokens}", string.Join(", ", generatedTokens));
        Log.Debug("Decoded output: \"{Output}\"", decoded);

        return ParseDecision(decoded, transcript, speaker);
    }

    private string BuildPrompt(string transcript, Speaker speaker)
    {
        var speakerDesc = speaker.Type == SpeakerType.Master ? "Master (enrolled user)" : $"Other Speaker {speaker.Id}";
        return string.Join("\n", new[]
        {
            $"Speaker {speaker.Id} says: {transcript}",
            "Does this likely contain a command for an AI agent (e.g., 'play music', 'set timer')? Yes/No.",
            "If yes, who is it intended for: 'master' for the enrolled user, 'speaker_X' for others, 'none' if unclear.",
            "Output ONLY JSON: {\"is_command\": true/false, \"respond_to\": \"master|speaker_1|speaker_2|none\"}"
        });
    }

    private static int ArgMaxFloatSpan(float[] values)
    {
        int maxIdx = 0;
        float maxVal = values[0];
        for (int i = 1; i < values.Length; i++)
            if (values[i] > maxVal) { maxVal = values[i]; maxIdx = i; }
        return maxIdx;
    }

    private (bool, string) ParseDecision(string decoded, string transcript, Speaker speaker)
    {
        if (string.IsNullOrWhiteSpace(decoded))
        {
            Log.Warning("LLM generated empty output.");
            return (false, "none");
        }

        Log.Information("LLM raw output: \"{Output}\"", decoded);

        try
        {
            using var doc = JsonDocument.Parse(decoded);
            var root = doc.RootElement;
            bool isCmd = root.TryGetProperty("is_command", out var cmdProp) && cmdProp.ValueKind == JsonValueKind.True;
            string respondTo = root.TryGetProperty("respond_to", out var toProp)
                ? toProp.GetString() ?? "none"
                : "none";

            Log.Information("LLM Decision → Command: {Cmd}, RespondTo: {To} | Speaker: {Type}{Id} | \"{Text}\"",
                isCmd, respondTo, speaker.Type, speaker.Id, transcript.Trim());

            return (isCmd, respondTo);
        }
        catch (Exception ex) when (ex is JsonException || ex is InvalidOperationException)
        {
            Log.Warning(ex, "Failed to parse LLM JSON output: \"{Output}\"", decoded);
            return (false, "none");
        }
    }

    public void Dispose() => _session?.Dispose();
}