"""Optimization pipeline: ONNX export → INT8 quantization → benchmark → pick winner."""
import asyncio
import logging
import os
import tempfile

import torch

from gateway.optimization.benchmarker import benchmark
from gateway.optimization.onnx_exporter import OnnxModelWrapper, export_to_onnx
from gateway.optimization.quantizer import quantize_onnx_model
from gateway.registry.model_store import ModelStore

logger = logging.getLogger(__name__)


class OptimizationPipeline:
    """Runs background optimization on uploaded models.

    Triggered by POST /models/{id}/load?optimize=true. Creates ONNX and
    quantized variants, benchmarks all backends, and swaps in the fastest.
    The model stays usable on its original PyTorch backend the entire time.
    """

    def __init__(self, model_store: ModelStore):
        self._store = model_store

    async def optimize(self, model_id: str) -> None:
        """Run the full optimization pipeline for a model. Meant to be fire-and-forget via asyncio.create_task."""
        try:
            model_info = await self._store.get(model_id)
            if not model_info:
                logger.warning("Optimization skipped: model '%s' not found", model_id)
                return

            pytorch_model = model_info.model
            input_size = model_info.input_size

            # create a temp directory for ONNX artifacts
            artifact_dir = tempfile.mkdtemp(prefix=f"mlgw_{model_id}_")
            await self._store.set_artifact_dir(model_id, artifact_dir)

            onnx_path = os.path.join(artifact_dir, "model.onnx")
            quant_path = os.path.join(artifact_dir, "model_quantized.onnx")
            sample_input = torch.randn(1, input_size)

            # track which backends we successfully create
            candidates = {"pytorch": pytorch_model}

            # --- Step 1: ONNX export ---
            await self._store.set_optimization_status(
                model_id, {"status": "running", "stage": "exporting_onnx"})

            try:
                await asyncio.to_thread(export_to_onnx, pytorch_model, input_size, onnx_path)
                candidates["onnx"] = OnnxModelWrapper(onnx_path)
            except Exception as e:
                logger.warning("ONNX export failed for '%s': %s", model_id, e)

            # --- Step 2: INT8 quantization (only if ONNX export succeeded) ---
            if "onnx" in candidates:
                await self._store.set_optimization_status(
                    model_id, {"status": "running", "stage": "quantizing"})
                try:
                    await asyncio.to_thread(quantize_onnx_model, onnx_path, quant_path)
                    candidates["onnx_quantized"] = OnnxModelWrapper(quant_path)
                except Exception as e:
                    logger.warning("Quantization failed for '%s': %s", model_id, e)

            # --- Step 3: Benchmark all available backends ---
            await self._store.set_optimization_status(
                model_id, {"status": "running", "stage": "benchmarking"})

            benchmarks = {}
            for name, model in candidates.items():
                try:
                    stats = await asyncio.to_thread(benchmark, model, sample_input)
                    benchmarks[name] = stats
                except Exception as e:
                    logger.warning("Benchmark failed for '%s' backend '%s': %s",
                                   model_id, name, e)

            if not benchmarks:
                await self._store.set_optimization_status(
                    model_id, {"status": "failed", "error": "all benchmarks failed"})
                return

            # --- Step 4: Pick the winner (lowest p50 latency) ---
            winner = min(benchmarks, key=lambda k: benchmarks[k]["p50_ms"])
            winning_model = candidates[winner]

            # only swap if the winner isn't already the current backend
            if winner != "pytorch":
                await self._store.update_backend(model_id, winning_model, winner)

            pytorch_p50 = benchmarks.get("pytorch", {}).get("p50_ms", 0)
            winner_p50 = benchmarks[winner]["p50_ms"]
            speedup = round(pytorch_p50 / winner_p50, 2) if winner_p50 > 0 else 1.0

            await self._store.set_optimization_status(model_id, {
                "status": "complete",
                "backend": winner,
                "original_p50_ms": pytorch_p50,
                "optimized_p50_ms": winner_p50,
                "speedup": speedup,
                "benchmarks": benchmarks,
            })

            logger.info("Optimization complete for '%s': winner=%s, speedup=%.2fx",
                        model_id, winner, speedup)

        except Exception as e:
            logger.exception("Optimization pipeline failed for '%s'", model_id)
            await self._store.set_optimization_status(
                model_id, {"status": "failed", "error": str(e)})
