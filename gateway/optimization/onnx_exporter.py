"""ONNX export and runtime wrapper for PyTorch models."""
import logging

import torch
import onnxruntime as ort

logger = logging.getLogger(__name__)


def export_to_onnx(model, input_size: int, output_path: str) -> str:
    """Export a PyTorch model to ONNX format.

    Uses dynamic axes so the exported model accepts any batch size.
    Returns the output path on success.
    """
    dummy_input = torch.randn(1, input_size)

    # dynamo=False uses the TorchScript exporter, which produces ONNX graphs
    # that onnxruntime's quantization tools can reliably process
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        opset_version=17,
        dynamo=False,
    )

    logger.info("ONNX export complete → %s", output_path)
    return output_path


class OnnxModelWrapper:
    """Wraps an ONNX Runtime session to be callable like a PyTorch model.

    The batcher calls model_info.model(stacked_tensor) and expects something
    iterable where each element has .tolist(). Numpy arrays satisfy both,
    so this is a drop-in replacement for nn.Module — zero batcher changes.
    """

    def __init__(self, onnx_path: str):
        self.session = ort.InferenceSession(onnx_path)
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, tensor: torch.Tensor):
        original_shape = tensor.shape
        # ONNX expects 2D (batch, features). The batcher may stack 2D request
        # tensors into 3D+, so flatten to 2D before calling and restore after.
        if tensor.dim() > 2:
            tensor = tensor.reshape(-1, tensor.shape[-1])

        numpy_input = tensor.numpy()
        outputs = self.session.run(None, {self.input_name: numpy_input})
        result = outputs[0]

        # restore original batch structure so the batcher fans out correctly
        if len(original_shape) > 2:
            result = result.reshape(*original_shape[:-1], -1)

        return result
