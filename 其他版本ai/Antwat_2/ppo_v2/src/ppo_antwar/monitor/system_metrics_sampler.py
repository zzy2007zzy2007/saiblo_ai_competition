import json
import os
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional

import psutil

from loguru import logger

from .constants import SAMPLE_INTERVAL, SAMPLES_PER_WRITE


class SystemMetricsSampler:
    """系统指标采样器 - 在后台线程中周期性采样系统资源使用情况

    采样指标：
    - CPU 使用率百分比
    - 内存使用量（GB）
    - GPU 使用率百分比（如果可用）
    - GPU 内存使用量（GB，如果可用）
    """

    def __init__(self, output_dir: str, sample_interval: int = SAMPLE_INTERVAL):
        self._output_dir = output_dir
        self._sample_interval = sample_interval
        self._samples_per_write = SAMPLES_PER_WRITE
        self._samples: deque = deque(maxlen=1000)
        self._phase: str = "training"
        self._nvml_handle = self._init_nvml()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    @staticmethod
    def _init_nvml():
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(gpu_name, bytes):
                gpu_name = gpu_name.decode()
            logger.info(f"NVML initialized, GPU: {gpu_name}")
            return handle
        except Exception:
            return None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("System metrics sampler started")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._flush()
        if self._nvml_handle is not None:
            try:
                import pynvml

                pynvml.nvmlShutdown()
                self._nvml_handle = None
            except Exception:
                pass

    def set_phase(self, phase: str) -> None:
        self._phase = phase

    def _run(self) -> None:
        while not self._stop_event.wait(self._sample_interval):
            try:
                sample = self._sample_once()
                self._samples.append(sample)
                self._maybe_flush()
            except Exception:
                logger.debug("Failed to sample system metrics")

    def _sample_once(self) -> Dict[str, Any]:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        ram_used_gb = memory.used / (1024**3)
        ram_total_gb = memory.total / (1024**3)

        gpu_util = -1.0
        gpu_mem_used_mb = -1
        gpu_mem_total_mb = -1
        gpu_temp = -1
        if self._nvml_handle is None:
            self._nvml_handle = self._init_nvml()
        if self._nvml_handle is not None:
            try:
                import pynvml

                gpu_info = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                gpu_util = float(gpu_info.gpu)
                gpu_mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                gpu_mem_used_mb = int(gpu_mem_info.used / (1024**2))
                gpu_mem_total_mb = int(gpu_mem_info.total / (1024**2))
                gpu_temp = int(pynvml.nvmlDeviceGetTemperature(self._nvml_handle, 0))
            except Exception:
                logger.debug("Failed to sample GPU metrics")

        return {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "phase": self._phase,
            "cpu_percent": cpu_percent,
            "ram_percent": memory.percent,
            "ram_used_gb": round(ram_used_gb, 2),
            "ram_total_gb": round(ram_total_gb, 2),
            "gpu_util": gpu_util,
            "gpu_mem_used_mb": gpu_mem_used_mb,
            "gpu_mem_total_mb": gpu_mem_total_mb,
            "gpu_temp": gpu_temp,
        }

    def _maybe_flush(self) -> None:
        if (
            len(self._samples) >= self._samples_per_write
            and len(self._samples) % self._samples_per_write == 0
        ):
            self._flush()

    def _flush(self) -> None:
        os.makedirs(self._output_dir, exist_ok=True)

        metrics_file = os.path.join(self._output_dir, "system_metrics.json")
        log_file = os.path.join(self._output_dir, "system_metrics.json.log")

        samples_list = list(self._samples)
        latest = samples_list[-1] if samples_list else {}
        latest_samples = samples_list[-10:] if samples_list else []

        stats = self._compute_stats(samples_list)

        data = {
            "latest": latest,
            "latest_samples": latest_samples,
            "stats": stats,
            "sample_count": len(samples_list),
        }

        with open(metrics_file, "w") as f:
            json.dump(data, f, indent=2)

        with open(log_file, "a") as f:
            for s in samples_list[-self._samples_per_write :]:
                f.write(json.dumps(s) + "\n")

    def _compute_stats(self, samples: List[Dict]) -> Dict[str, float]:
        if not samples:
            return {}
        cpu_vals = [s["cpu_percent"] for s in samples]
        ram_vals = [s["ram_percent"] for s in samples]
        gpu_vals = [s["gpu_util"] for s in samples if s["gpu_util"] >= 0]
        training_samples = [s for s in samples if s.get("phase") == "training"]
        battle_samples = [s for s in samples if s.get("phase") == "battle"]

        stats = {
            "cpu_avg": round(sum(cpu_vals) / len(cpu_vals), 1),
            "cpu_max": round(max(cpu_vals), 1),
            "ram_avg": round(sum(ram_vals) / len(ram_vals), 1),
            "ram_max": round(max(ram_vals), 1),
        }
        if gpu_vals:
            stats["gpu_avg"] = round(sum(gpu_vals) / len(gpu_vals), 1)
            stats["gpu_max"] = round(max(gpu_vals), 1)
        if training_samples:
            train_cpu = [s["cpu_percent"] for s in training_samples]
            stats["training_cpu_avg"] = round(sum(train_cpu) / len(train_cpu), 1)
        if battle_samples:
            battle_cpu = [s["cpu_percent"] for s in battle_samples]
            stats["battle_cpu_avg"] = round(sum(battle_cpu) / len(battle_cpu), 1)
        return stats
