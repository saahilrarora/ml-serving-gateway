"""Tests for the inference optimization pipeline.

Verifies ONNX export, INT8 quantization, benchmarking, and the full
pipeline orchestrator using the real TextClassifier model.
"""
import asyncio
import os
import tempfile

import pytest
import pytest_asyncio
import torch

from demo.models.text_classifier import TextClassifier
from gateway.optimization.benchmarker import benchmark
from gateway.optimization.onnx_exporter import OnnxModelWrapper, export_to_onnx
from gateway.optimization.quantizer import quantize_onnx_model
from gateway.optimization.pipeline import OptimizationPipeline
from gateway.registry.model_store import ModelStore


# -- Fixtures ---------------------------------------------------------------

@pytest.fixture
def model():
    """A fresh TextClassifier in eval mode."""
    m = TextClassifier(input_size=10, num_classes=3)
    m.eval()
    return m


@pytest.fixture
def sample_input():
    """Single-sample input tensor: shape (1, 10)."""
    return torch.randn(1, 10)


@pytest.fixture
def artifact_dir():
    """Temp directory for ONNX files, cleaned up after test."""
    d = tempfile.mkdtemp(prefix="mlgw_test_")
    yield d
    import shutil
    shutil.rmtree(d, ignore_errors=True)


@pytest_asyncio.fixture
async def store(model):
    """ModelStore with one TextClassifier loaded."""
    s = ModelStore()
    await s.add("clf", model, input_size=10)
    return s


# -- ONNX Export Tests -------------------------------------------------------

def test_onnx_export_creates_file(model, artifact_dir):
    """export_to_onnx should produce a valid .onnx file."""
    onnx_path = os.path.join(artifact_dir, "model.onnx")
    result = export_to_onnx(model, input_size=10, output_path=onnx_path)

    assert result == onnx_path
    assert os.path.exists(onnx_path)
    assert os.path.getsize(onnx_path) > 0


def test_onnx_output_matches_pytorch(model, artifact_dir, sample_input):
    """ONNX model should produce outputs close to the original PyTorch model."""
    onnx_path = os.path.join(artifact_dir, "model.onnx")
    export_to_onnx(model, input_size=10, output_path=onnx_path)

    wrapper = OnnxModelWrapper(onnx_path)

    with torch.no_grad():
        pytorch_out = model(sample_input).numpy()

    onnx_out = wrapper(sample_input)

    # outputs should be very close (float32 rounding differences are OK)
    for py_val, onnx_val in zip(pytorch_out.flat, onnx_out.flat):
        assert abs(py_val - onnx_val) < 1e-5, f"Mismatch: {py_val} vs {onnx_val}"


def test_onnx_wrapper_handles_batches(model, artifact_dir):
    """OnnxModelWrapper should handle variable batch sizes (dynamic axes)."""
    onnx_path = os.path.join(artifact_dir, "model.onnx")
    export_to_onnx(model, input_size=10, output_path=onnx_path)
    wrapper = OnnxModelWrapper(onnx_path)

    # batch of 5
    batch_input = torch.randn(5, 10)
    result = wrapper(batch_input)
    assert result.shape == (5, 3)  # 5 samples, 3 classes

    # single sample
    single_input = torch.randn(1, 10)
    result = wrapper(single_input)
    assert result.shape == (1, 3)


# -- Quantization Tests ------------------------------------------------------

def test_quantization_creates_file(model, artifact_dir):
    """quantize_onnx_model should produce a quantized .onnx file."""
    onnx_path = os.path.join(artifact_dir, "model.onnx")
    quant_path = os.path.join(artifact_dir, "model_quantized.onnx")

    export_to_onnx(model, input_size=10, output_path=onnx_path)
    result = quantize_onnx_model(onnx_path, quant_path)

    assert result == quant_path
    assert os.path.exists(quant_path)
    assert os.path.getsize(quant_path) > 0


def test_quantized_output_matches_original(model, artifact_dir, sample_input):
    """Quantized model output should be close to the original (within tolerance)."""
    onnx_path = os.path.join(artifact_dir, "model.onnx")
    quant_path = os.path.join(artifact_dir, "model_quantized.onnx")

    export_to_onnx(model, input_size=10, output_path=onnx_path)
    quantize_onnx_model(onnx_path, quant_path)

    with torch.no_grad():
        pytorch_out = model(sample_input).numpy()

    quant_wrapper = OnnxModelWrapper(quant_path)
    quant_out = quant_wrapper(sample_input)

    # INT8 quantization introduces more error than float ONNX, so wider tolerance
    for py_val, q_val in zip(pytorch_out.flat, quant_out.flat):
        assert abs(py_val - q_val) < 0.1, f"Quantized mismatch too large: {py_val} vs {q_val}"


# -- Benchmarker Tests --------------------------------------------------------

def test_benchmark_returns_stats(model, sample_input):
    """benchmark() should return a dict with p50, p95, p99 keys."""
    stats = benchmark(model, sample_input, warmup=2, iterations=10)

    assert "p50_ms" in stats
    assert "p95_ms" in stats
    assert "p99_ms" in stats
    # p50 <= p95 <= p99 (sorted percentiles)
    assert stats["p50_ms"] <= stats["p95_ms"] <= stats["p99_ms"]
    # all should be positive numbers
    assert all(v > 0 for v in stats.values())


def test_benchmark_works_with_onnx(model, artifact_dir, sample_input):
    """benchmark() should work with OnnxModelWrapper too."""
    onnx_path = os.path.join(artifact_dir, "model.onnx")
    export_to_onnx(model, input_size=10, output_path=onnx_path)
    wrapper = OnnxModelWrapper(onnx_path)

    stats = benchmark(wrapper, sample_input, warmup=2, iterations=10)

    assert "p50_ms" in stats
    assert all(v > 0 for v in stats.values())


# -- Full Pipeline Tests ------------------------------------------------------

@pytest.mark.asyncio
async def test_pipeline_completes(store):
    """Full pipeline should run and set status to 'complete'."""
    pipeline = OptimizationPipeline(store)
    await pipeline.optimize("clf")

    info = await store.get("clf")
    assert info.optimization_status["status"] == "complete"
    assert "backend" in info.optimization_status
    assert "speedup" in info.optimization_status
    assert "benchmarks" in info.optimization_status
    # winner should be one of the valid backends
    assert info.optimization_status["backend"] in ("pytorch", "onnx", "onnx_quantized")


@pytest.mark.asyncio
async def test_pipeline_updates_backend(store):
    """After pipeline, the model's backend field should match the winner."""
    pipeline = OptimizationPipeline(store)
    await pipeline.optimize("clf")

    info = await store.get("clf")
    assert info.backend == info.optimization_status["backend"]


@pytest.mark.asyncio
async def test_pipeline_model_still_works(store):
    """After optimization, the model should still produce valid outputs."""
    pipeline = OptimizationPipeline(store)
    await pipeline.optimize("clf")

    info = await store.get("clf")
    test_input = torch.randn(1, 10)

    # the model callable (whatever backend won) should still work
    with torch.no_grad():
        output = info.model(test_input)

    # should produce 3 class probabilities
    if hasattr(output, 'tolist'):
        result = output.tolist() if not isinstance(output, list) else output
    else:
        result = output

    assert len(result[0]) == 3  # 3 classes


@pytest.mark.asyncio
async def test_pipeline_nonexistent_model(store):
    """Pipeline should handle a missing model gracefully."""
    pipeline = OptimizationPipeline(store)
    # should not raise — just logs a warning
    await pipeline.optimize("nonexistent")


@pytest.mark.asyncio
async def test_pipeline_creates_artifacts(store):
    """Pipeline should create ONNX files in the artifact directory."""
    pipeline = OptimizationPipeline(store)
    await pipeline.optimize("clf")

    info = await store.get("clf")
    assert info.artifact_dir is not None
    assert os.path.exists(info.artifact_dir)
    # at least the base ONNX file should exist
    assert os.path.exists(os.path.join(info.artifact_dir, "model.onnx"))


@pytest.mark.asyncio
async def test_cleanup_on_unload(store):
    """Unloading a model should clean up its optimization artifacts."""
    pipeline = OptimizationPipeline(store)
    await pipeline.optimize("clf")

    info = await store.get("clf")
    artifact_dir = info.artifact_dir

    await store.remove("clf")
    assert not os.path.exists(artifact_dir)
