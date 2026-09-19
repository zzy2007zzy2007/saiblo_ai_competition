"""
系统资源采样模块
"""

import os
import json
import time
import threading
from typing import Optional, Dict, List
from collections import deque


class SystemMetricsSampler:
    """系统资源采样器，与训练并行执行"""

    def __init__(self, log_dir: Optional[str], sample_interval: int = 10):
        self.log_dir = log_dir
        self.sample_interval = sample_interval
        self._sampling = False
        self._sample_thread: Optional[threading.Thread] = None
        self._latest_metrics: Optional[Dict] = None
        self._samples: deque = deque(maxlen=1000)
        self._phase = "training"
        self._sample_count = 0
        self._samples_per_write = 6
        self._pending_writes = 0

    def _get_system_metrics(self) -> Dict:
        """获取当前系统指标"""
        import torch

        metrics = {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "phase": self._phase,
            "cpu_percent": self._get_cpu_percent(),
            "ram_percent": self._get_ram_percent(),
            "ram_used_gb": self._get_ram_used_gb(),
            "ram_total_gb": self._get_ram_total_gb(),
        }

        if torch.cuda.is_available():
            try:
                import subprocess
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
                     '--format=csv,noheader,nounits'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    parts = result.stdout.strip().split(',')
                    metrics["gpu_util"] = float(parts[0].strip())
                    metrics["gpu_mem_used_mb"] = int(parts[1].strip())
                    metrics["gpu_mem_total_mb"] = int(parts[2].strip())
                    metrics["gpu_temp"] = int(parts[3].strip())
            except Exception:
                metrics["gpu_util"] = 0
                metrics["gpu_mem_used_mb"] = 0
                metrics["gpu_mem_total_mb"] = 0
                metrics["gpu_temp"] = 0
        else:
            metrics["gpu_util"] = 0
            metrics["gpu_mem_used_mb"] = 0
            metrics["gpu_mem_total_mb"] = 0
            metrics["gpu_temp"] = 0

        return metrics

    def _get_cpu_percent(self) -> float:
        """获取CPU使用率"""
        try:
            import psutil
            return psutil.cpu_percent(interval=0.1)
        except Exception:
            return 0.0

    def _get_ram_percent(self) -> float:
        """获取内存使用率"""
        try:
            import psutil
            return psutil.virtual_memory().percent
        except Exception:
            return 0.0

    def _get_ram_used_gb(self) -> float:
        """获取已使用内存（GB）"""
        try:
            import psutil
            return psutil.virtual_memory().used / (1024**3)
        except Exception:
            return 0.0

    def _get_ram_total_gb(self) -> float:
        """获取总内存（GB）"""
        try:
            import psutil
            return psutil.virtual_memory().total / (1024**3)
        except Exception:
            return 0.0

    def _sampling_loop(self) -> None:
        """采样循环，在独立线程中运行"""
        while self._sampling:
            metrics = self._get_system_metrics()
            self._latest_metrics = metrics
            self._samples.append(metrics)
            self._sample_count += 1
            self._pending_writes += 1

            if self._pending_writes >= self._samples_per_write:
                self._flush_to_file()
                self._pending_writes = 0

            time.sleep(self.sample_interval)

    def _flush_to_file(self) -> None:
        """将采样数据写入文件"""
        if not self.log_dir or not self._samples:
            return

        os.makedirs(self.log_dir, exist_ok=True)

        latest_samples = list(self._samples)

        stats = self._compute_stats(latest_samples)

        data = {
            "latest": latest_samples[-1] if latest_samples else {},
            "latest_samples": latest_samples[-10:] if len(latest_samples) > 10 else latest_samples,
            "stats": stats,
            "sample_count": self._sample_count
        }

        metrics_file = os.path.join(self.log_dir, "system_metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(data, f, indent=2)

        log_file = os.path.join(self.log_dir, "system_metrics.json.log")
        with open(log_file, "a") as f:
            for sample in latest_samples[-self._samples_per_write:]:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    def _compute_stats(self, samples: List[Dict]) -> Dict:
        """计算采样统计"""
        if not samples:
            return {}

        cpu_values = [s["cpu_percent"] for s in samples]
        ram_values = [s["ram_percent"] for s in samples]
        gpu_values = [s.get("gpu_util", 0) for s in samples]

        training_samples = [s for s in samples if s.get("phase") == "training"]
        battle_samples = [s for s in samples if s.get("phase") == "battle"]

        stats = {
            "cpu_avg": sum(cpu_values) / len(cpu_values) if cpu_values else 0,
            "cpu_max": max(cpu_values) if cpu_values else 0,
            "ram_avg": sum(ram_values) / len(ram_values) if ram_values else 0,
            "ram_max": max(ram_values) if ram_values else 0,
            "gpu_avg": sum(gpu_values) / len(gpu_values) if gpu_values else 0,
            "gpu_max": max(gpu_values) if gpu_values else 0,
        }

        if training_samples:
            training_cpu = [s["cpu_percent"] for s in training_samples]
            training_gpu = [s.get("gpu_util", 0) for s in training_samples]
            stats["training_cpu_avg"] = sum(training_cpu) / len(training_cpu)
            stats["training_gpu_avg"] = sum(training_gpu) / len(training_gpu)

        if battle_samples:
            battle_cpu = [s["cpu_percent"] for s in battle_samples]
            battle_gpu = [s.get("gpu_util", 0) for s in battle_samples]
            stats["battle_cpu_avg"] = sum(battle_cpu) / len(battle_cpu)
            stats["battle_gpu_avg"] = sum(battle_gpu) / len(battle_gpu)

        return stats

    def start(self, phase: str = "training") -> None:
        """开始采样"""
        if self._sampling:
            return

        self._phase = phase
        self._sampling = True
        self._sample_thread = threading.Thread(target=self._sampling_loop, daemon=True)
        self._sample_thread.start()

    def stop(self) -> None:
        """停止采样"""
        self._sampling = False
        if self._sample_thread is not None:
            self._sample_thread.join(timeout=5)
            self._sample_thread = None
        self._flush_to_file()

    def set_phase(self, phase: str) -> None:
        """设置当前阶段"""
        self._phase = phase

    def get_current_metrics(self) -> Optional[Dict]:
        """获取当前系统指标"""
        return self._latest_metrics

    def get_stats(self) -> Dict:
        """获取采样统计"""
        if not self._samples:
            return {}
        return self._compute_stats(list(self._samples))

    def get_all_samples(self) -> List[Dict]:
        """获取所有采样数据"""
        return list(self._samples)

    @property
    def sample_count(self) -> int:
        """获取采样数量"""
        return self._sample_count