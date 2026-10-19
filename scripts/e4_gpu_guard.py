"""Lightweight NVML sampling; never stops or changes another user's process."""
import ctypes, os, re, threading, time
from pathlib import Path
import pynvml
from e4_full_common import ROOT

class GPUGuard:
    def __init__(self):
        version = re.search(r'Kernel Module\s+([0-9.]+)', Path('/proc/driver/nvidia/version').read_text()).group(1)
        library = ROOT/'.cache/nvidia-monitor'/version/'lib/libnvidia-ml.so.1'
        if library.exists(): pynvml.nvmlLib = ctypes.CDLL(str(library))
        pynvml.nvmlInit(); self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.owner = os.getpid(); self.lock = threading.Lock(); self.active = False
        self.seen = set(); self.error = None; self.closed = threading.Event()
        self.thread = threading.Thread(target=self._monitor, daemon=True); self.thread.start()
    def other_pids(self):
        compute = pynvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
        graphics = pynvml.nvmlDeviceGetGraphicsRunningProcesses(self.handle)
        return sorted({p.pid for p in compute + graphics if p.pid != self.owner})
    def _monitor(self):
        while not self.closed.wait(.25):
            try:
                pids = self.other_pids()
                with self.lock:
                    if self.active: self.seen.update(pids)
            except Exception as exc: self.error = repr(exc)
    def wait_idle(self, callback=None):
        while True:
            if self.error: raise RuntimeError('NVML monitor failed: ' + self.error)
            pids = self.other_pids()
            if not pids: return
            if callback: callback(pids)
            time.sleep(5)
    def begin(self):
        with self.lock: self.seen = set(self.other_pids()); self.active = True
    def end(self):
        with self.lock:
            self.seen.update(self.other_pids()); self.active = False; pids = sorted(self.seen)
        if self.error: raise RuntimeError('NVML monitor failed: ' + self.error)
        return pids
    def telemetry(self):
        mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        return dict(used_mib=mem.used/1024**2,
                    sm_clock_mhz=pynvml.nvmlDeviceGetClockInfo(self.handle,pynvml.NVML_CLOCK_SM),
                    temperature_c=pynvml.nvmlDeviceGetTemperature(self.handle,pynvml.NVML_TEMPERATURE_GPU))
    def close(self):
        self.closed.set(); self.thread.join(timeout=2); pynvml.nvmlShutdown()
