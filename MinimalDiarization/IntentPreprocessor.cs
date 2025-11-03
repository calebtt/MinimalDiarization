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
        var tokens = _tokenizer.EncodeToIds(prompt).ToArray();
        var outputTokens = new List<int>(tokens);
        int eos = 0;
        _tokenizer.Vocabulary.TryGetValue("<|endoftext|>", out eos);

        while (outputTokens.Count < MaxLength)
        {
            var currentLength = outputTokens.Count;
            var inputIds = outputTokens.Select(t => (long)t).ToArray();
            var inputTensor = new DenseTensor<long>(inputIds, new[] { 1, currentLength }, false);
            var inputs = new[] { NamedOnnxValue.CreateFromTensor("input_ids", inputTensor) };

            using var result = _session.Run(inputs);
            var logits = result.First(r => r.Name == "logits").AsTensor<float>();

            var lastLogits = new float[logits.Dimensions[2]];
            for (int v = 0; v < logits.Dimensions[2]; v++)
            {
                lastLogits[v] = logits[0, logits.Dimensions[1] - 1, v];
            }
            var nextToken = ArgMaxFloatSpan(lastLogits);
            if (nextToken == eos) break;
            outputTokens.Add(nextToken);
        }

        var generatedTokens = outputTokens.Skip(tokens.Length);
        var decoded = _tokenizer.Decode(generatedTokens).Trim();

        return ParseDecision(decoded, transcript, speaker);
    }

    private string BuildPrompt(string transcript, Speaker speaker)
    {
        var speakerDesc = speaker.Type == SpeakerType.Master ? "Master (enrolled user)" : $"Other Speaker {speaker.Id}";
        return string.Join("\n", new[]
        {
            $"Context: Multi-speaker env. Transcript from {speakerDesc}: {transcript}",
            "Is this a valid command (e.g., 'play music', 'set timer')? Yes/No.",
            "If yes, who to respond to: 'master' for user, 'speaker_X' for others, 'none' if unclear.",
            "Output ONLY JSON: {\\\"is_command\\\": true/false, \\\"respond_to\\\": \\\"master|speaker_1|speaker_2|or none\\\"}"
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
        Log.Debug("Decoded output: {Decoded}", decoded);
        try
        {
            using var jsonDoc = JsonDocument.Parse(decoded);
            var root = jsonDoc.RootElement;
            var isCmd = root.GetProperty("is_command").GetBoolean();
            var respond = root.GetProperty("respond_to").GetString() ?? "none";
            Log.Information("LLM: {IsCmd}, {Respond} | {Type}:{Id} | {Text:50}", isCmd, respond, speaker.Type, speaker.Id, transcript);
            return (isCmd, respond);
        }
        catch (JsonException)
        {
            Log.Warning("Invalid JSON: {Decoded}; default none.", decoded);
            return (false, "none");
        }
    }

    public void Dispose() => _session?.Dispose();
}