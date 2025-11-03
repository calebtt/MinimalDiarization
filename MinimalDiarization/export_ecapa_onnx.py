import torch
from speechbrain.inference.encoders import MelSpectrogramEncoder
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Load pretrained MelSpectrogramEncoder
    logger.info("Loading SpeechBrain ECAPA-TDNN model...")
    encoder = MelSpectrogramEncoder.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb-mel-spec",
        run_opts={"device": "cpu"}  # Use 'cuda' if GPU available
    )
    
    # Extract the core ECAPA-TDNN module (post-normalizer)
    model = encoder.hparams.embedding_model
    model.eval()  # Inference mode
    
    # Dummy input: [B=1, time=100, feats=80] (normalized mel-spec)
    dummy_input = torch.randn(1, 100, 80)
    
    # Export to ONNX (opset 11 for compatibility)
    torch.onnx.export(
        model,
        dummy_input,
        "ecapa_tdnn.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],  # [1, time, 80]
        output_names=["embedding"],  # [1, 1, 192]
        dynamic_axes={
            "input": {1: "time"},
            "embedding": {1: "time"}  # Pooling squeezes to 1
        },
        verbose=False
    )
    
    logger.info("Exported ecapa_tdnn.onnx successfully!")
    # Verify shape
    with torch.no_grad():
        emb = model(dummy_input)
        logger.info(f"Model output shape: {emb.shape}")  # [1, 1, 192]

if __name__ == "__main__":
    main()