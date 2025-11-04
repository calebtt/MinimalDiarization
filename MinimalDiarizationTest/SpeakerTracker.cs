// SpeakerTracker.cs
using MinimalDiarization.Core;
using Serilog;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;

namespace MinimalDiarizationTest;

public sealed class SpeakerTracker : IDisposable
{
    private readonly List<Speaker> _knownSpeakers = new();
    private readonly EcapaTdnnModel _ecapaModel;
    private readonly float _threshold;
    private readonly object _lock = new();

    // optional: keep a short history for debugging / clustering
    private readonly List<(double time, float[] embedding, string label)> _history = new();

    public IReadOnlyList<Speaker> KnownSpeakers => _knownSpeakers.AsReadOnly();
    public IReadOnlyList<(double time, float[] embedding, string label)> History => _history.AsReadOnly();

    public SpeakerTracker(EcapaTdnnModel ecapaModel, float threshold = Algos.SpeakerMatchThreshold)
    {
        _ecapaModel = ecapaModel ?? throw new ArgumentNullException(nameof(ecapaModel));
        _threshold = threshold;
    }

    /// <summary>
    /// Registers the enrolled master speaker (Speaker 0). Call once after enrollment.
    /// </summary>
    public void RegisterMaster(float[] masterEmbedding)
    {
        if (masterEmbedding == null) throw new ArgumentNullException(nameof(masterEmbedding));
        lock (_lock)
        {
            // Remove any previous master
            _knownSpeakers.RemoveAll(s => s.Type == SpeakerType.Master);
            _knownSpeakers.Add(new Speaker(SpeakerType.Master, 0, masterEmbedding));
        }
        Log.Information("Registered Master speaker (embedding length {Len}).", masterEmbedding.Length);
    }

    /// <summary>
    /// Assigns the given embedding to an existing speaker or creates a new one.
    /// </summary>
    public Speaker AssignSpeaker(float[] embedding, double startSec)
    {
        if (embedding == null) throw new ArgumentNullException(nameof(embedding));
        if (embedding.Length != EcapaTdnnModel.EmbeddingDim)
            throw new ArgumentException($"Embedding must be {EcapaTdnnModel.EmbeddingDim} dimensions.", nameof(embedding));

        lock (_lock)
        {
            Speaker? best = null;
            float bestSim = float.MinValue;

            foreach (var spk in _knownSpeakers)
            {
                float sim = Algos.CosineSimilarity(embedding, spk.Embedding);
                if (sim > bestSim)
                {
                    bestSim = sim;
                    best = spk;
                }
            }

            if (best != null && bestSim >= _threshold)
            {
                // Update segment count immutably
                int idx = _knownSpeakers.IndexOf(best);
                var updated = best with { SegmentCount = best.SegmentCount + 1 };
                _knownSpeakers[idx] = updated;

                Log.Information(
                    "*** Assigned to {Label} (sim={Sim:F3}) ***",
                    best.Type == SpeakerType.Master ? "Master" : $"Other Speaker {best.Id}",
                    bestSim);

                _history.Add((startSec, embedding, best.Type == SpeakerType.Master ? "master" : $"speaker_{best.Id}"));
                return updated;
            }

            // ---- New speaker -------------------------------------------------
            int newId = _knownSpeakers.Count(s => s.Type == SpeakerType.Other) + 1;
            var newSpeaker = new Speaker(SpeakerType.Other, newId, embedding, 1);
            _knownSpeakers.Add(newSpeaker);

            Log.Information(
                "*** New Speaker Detected: Other Speaker {Id} (best sim={Sim:F3}) ***",
                newId, bestSim);

            _history.Add((startSec, embedding, $"speaker_{newId}"));
            return newSpeaker;
        }
    }

    public void Dispose()
    {
        lock (_lock)
        {
            _knownSpeakers.Clear();
            _history.Clear();
        }
    }
}