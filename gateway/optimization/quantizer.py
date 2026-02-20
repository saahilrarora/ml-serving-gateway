"""INT8 dynamic quantization for ONNX models."""
import logging

from onnxruntime.quantization import quantize_dynamic, QuantType

logger = logging.getLogger(__name__)


def quantize_onnx_model(input_path: str, output_path: str) -> str:
    """Apply INT8 dynamic quantization to an ONNX model.

    Dynamic quantization converts float32 weights to int8 at load time,
    reducing model size and often improving CPU inference speed.
    Returns the output path on success.
    """
    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,
    )

    logger.info("ONNX INT8 quantization complete → %s", output_path)
    return output_path
