// CommandIntentClassifier.cs (improved)
using Microsoft.Extensions.AI;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.OpenAI;
using Serilog;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace MinimalDiarization.Core;

public sealed class CommandIntentClassifier
{
    private const string IsDirectedAtAssistantKey = "directedAtAssistant";
    private const string ConfidenceKey = "confidence";

    private readonly IChatCompletionService _chat;

    public CommandIntentClassifier(IChatCompletionService chatService)
    {
        _chat = chatService;
    }

    // Few-shot examples + strict JSON output
    public static string BuildPrompt(string text, Speaker speaker)
    {
        string prompt = string.Format(@"
Task: Determine if the user is speaking to the voice assistant.

Respond with ONLY one JSON object:
{{""directedAtAssistant"": true|false, ""confidence"": 0.00-1.00}}

Examples:
{{""directedAtAssistant"": true, ""confidence"": 0.95}}  // ""Hey computer, play music.""
{{""directedAtAssistant"": false, ""confidence"": 0.90}} // ""That movie was great.""

Utterance (Speaker {0}): ""{1}""
", speaker.Id, text);

        return prompt.Trim();
    }


    public async Task<bool> IsCommandAsync(string transcript, Speaker speaker)
    {
        var prompt = BuildPrompt(transcript, speaker);

        var history = new ChatHistory
        {
            new ChatMessageContent(AuthorRole.User, prompt)
        };

        try
        {
            // If your IChatCompletionService allows setting options, set temperature=0
            // For example: var response = await _chat.GetChatMessageContentAsync(history, new ChatOptions{ Temperature = 0 });
            var settings = new OpenAIPromptExecutionSettings();
            settings.MaxTokens = 19;
            settings.Temperature = 0.1;
            settings.TopP = 0.1;
            var response = await _chat.GetChatMessageContentAsync(history, settings);
            var raw = (response?.Content ?? "").Trim();
            Log.Debug("Raw classifier response: {Raw}", raw);

            // Try to parse JSON first
            try
            {
                using var doc = JsonDocument.Parse(raw);
                var root = doc.RootElement;
                bool isCommand = root.GetProperty(IsDirectedAtAssistantKey).GetBoolean();
                double confidence = root.GetProperty(ConfidenceKey).GetDouble();

                Log.Information("Intent: \"{Text}\" → {Answer} (conf {Conf:F2})", transcript, isCommand ? "YES" : "NO", confidence);
                // Use a confidence threshold; tune as needed
                return isCommand && confidence >= 0.5;
            }
            catch (JsonException)
            {
                // Not JSON — fall back to simple token check + heuristics
                var firstWord = raw.Split(' ', StringSplitOptions.RemoveEmptyEntries).FirstOrDefault() ?? "";
                if (firstWord.Equals("Yes", StringComparison.OrdinalIgnoreCase) ||
                    firstWord.Equals("No", StringComparison.OrdinalIgnoreCase))
                {
                    bool isYes = firstWord.Equals("Yes", StringComparison.OrdinalIgnoreCase);
                    Log.Warning("Non-JSON but tokenized answer: {Answer}. Using token. Raw: {Raw}", isYes ? "YES" : "NO", raw);
                    return isYes;
                }

                // Final fallback: local rule-based classifier
                bool fallback = LocalRuleBasedClassifier(transcript);
                Log.Warning("Fallback rule-based classifier result: {Result} for \"{Text}\". Raw response: {Raw}", fallback ? "YES" : "NO", transcript, raw);
                return fallback;
            }
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Classifier inference failed: {Text}", transcript);
            // On error, fallback locally
            return LocalRuleBasedClassifier(transcript);
        }
    }

    // Simple heuristic fallback - tune keywords / regex for your domain
    private static bool LocalRuleBasedClassifier(string text)
    {
        if (string.IsNullOrWhiteSpace(text)) return false;
        var t = text.Trim().ToLowerInvariant();

        // Typical imperative verbs / voice agent triggers
        string[] commandKeywords = new[]
        {
            "open", "close", "play", "pause", "stop", "set", "call", "dial",
            "send", "turn on", "turn off", "launch", "start", "stop", "mute",
            "unmute", "next", "previous", "skip", "search", "find", "navigate",
            "directions", "volume", "add", "remove", "delete", "schedule", "remind"
        };

        // If it starts with one of the verbs or contains "please" + verb
        if (commandKeywords.Any(k => t.StartsWith(k) || t.Contains($" {k}"))) return true;

        // Short utterances with numeric or unit (e.g., "set timer for 5 minutes")
        if (Regex.IsMatch(t, @"\b(\d+|one|two|three|minute|minutes|hour|hours|sec|second)\b")) return true;

        return false;
    }
}
