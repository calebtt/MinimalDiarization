namespace MinimalDiarization.Core;

public enum SpeakerType
{
    Master = 0,  // Enrolled user (commands)
    Other        // Diarized non-master
}

public record Speaker(SpeakerType Type, int Id, float[] Embedding, int SegmentCount = 0);