"""GPU scheduling utilities for batch experiment runners.

Provides a GPUScheduler that assigns jobs to the least-loaded GPU
and waits for any GPU to free up when all are at capacity.
Supports multi-GPU jobs (e.g. tensor parallelism).
"""

import os
import subprocess
import time

import torch


def detect_mode():
    """Detect whether to run locally or via Slurm."""
    gpu_count = torch.cuda.device_count()
    print(f"GPUs available: {gpu_count}")
    return "local" if gpu_count > 0 else "slurm"


class GPUScheduler:
    """Schedules subprocess jobs across GPUs, picking the least-loaded GPU."""

    def __init__(self, gpus, processes_per_gpu):
        self.gpus = gpus
        self.processes_per_gpu = processes_per_gpu
        self._gpu_processes = {gid: [] for gid in gpus}
        self.total_jobs = 0

    def _refresh(self, gpu_id):
        """Remove finished processes from a GPU's process list."""
        self._gpu_processes[gpu_id] = [
            p for p in self._gpu_processes[gpu_id] if p.poll() is None
        ]

    def _refresh_all(self):
        for gid in self.gpus:
            self._refresh(gid)

    def pick_gpus(self, n=1):
        """Pick n GPUs with free slots, preferring least-loaded. Blocks until available."""
        while True:
            self._refresh_all()

            free = [
                gid for gid in self.gpus
                if len(self._gpu_processes[gid]) < self.processes_per_gpu
            ]
            if len(free) >= n:
                free.sort(key=lambda gid: len(self._gpu_processes[gid]))
                return free[:n]

            print(f"[Scheduler] Need {n} free GPU(s), {len(free)} available, waiting...")
            while True:
                for gid in self.gpus:
                    for proc in self._gpu_processes[gid]:
                        if proc.poll() is not None:
                            self._refresh_all()
                            free = [
                                g for g in self.gpus
                                if len(self._gpu_processes[g]) < self.processes_per_gpu
                            ]
                            if len(free) >= n:
                                free.sort(key=lambda g: len(self._gpu_processes[g]))
                                return free[:n]
                time.sleep(1)

    def launch(self, cmd, log_file, gpu_ids):
        """Launch a command on the given GPU(s) and track the process.

        Args:
            cmd: The command string to run (without CUDA_VISIBLE_DEVICES prefix).
            log_file: Path to write stdout/stderr.
            gpu_ids: List of GPU IDs to expose via CUDA_VISIBLE_DEVICES.
        """
        gpu_ids_str = ",".join(str(g) for g in gpu_ids)
        full_cmd = f"export CUDA_VISIBLE_DEVICES={gpu_ids_str} && {cmd}"

        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, 'w') as log_f:
            proc = subprocess.Popen(full_cmd, shell=True, stdout=log_f, stderr=subprocess.STDOUT)

        for gid in gpu_ids:
            self._gpu_processes[gid].append(proc)
        self.total_jobs += 1

        running = len(self._gpu_processes[gpu_ids[0]])
        print(f"[GPU {gpu_ids_str}] Starting ({running}/{self.processes_per_gpu}): {cmd.split('&&')[-1].strip()[:120]}")

    def wait_all(self):
        """Wait for all tracked processes to finish."""
        print("Waiting for all remaining jobs to finish...")
        for gid in self.gpus:
            for proc in self._gpu_processes[gid]:
                proc.wait()
        print("All jobs completed.")
