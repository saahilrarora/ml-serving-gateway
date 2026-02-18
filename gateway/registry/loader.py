import logging
import os
import tempfile

import torch

logger = logging.getLogger(__name__)


def load_model_from_bytes(model_bytes: bytes) -> tuple:
    """Load a PyTorch model from raw bytes (as received from file upload).

    Returns:
        tuple of (model, input_size)
    """
    # write bytes to a temp file — torch.load needs a file path
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
    try:
        tmp.write(model_bytes)
        tmp.close()  # flush and close before torch reads it

        # deserialize the model from the .pt file
        model = torch.load(tmp.name, weights_only=False, map_location="cpu")
        model.eval()  # set to inference mode (no dropout, no batchnorm updates)

        # detect input size from the first linear layer's weight shape
        input_size = _detect_input_size(model)
        logger.info(f"Model loaded successfully, input_size={input_size}")

        return model, input_size
    finally:
        os.unlink(tmp.name)  # clean up the temp file regardless of success/failure


def _detect_input_size(model) -> int:
    """Walk the model's layers to find the input dimension of the first Linear layer."""
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            return module.in_features  # e.g. 10 for a Linear(10, 32)
    return -1  # unknown — model has no Linear layers
