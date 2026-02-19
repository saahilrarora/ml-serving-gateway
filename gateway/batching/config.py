from dataclasses import dataclass


@dataclass
class BatchConfig:
    max_batch_size: int = 32      # flush when this many requests are queued
    max_wait_ms: float = 50.0     # flush after this many ms even if batch isn't full
