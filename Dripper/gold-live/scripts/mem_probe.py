"""Where does the memory actually go? Measure, do not attribute.

    python -m scripts.mem_probe

Written because the one-hour soak showed RSS climbing 587 -> 833 MB and it
would have been easy, and wrong, to credit that to the unbounded collections
fixed at the same time. It was not them: they cost about 10 MB per 20,000
utterances, which at the observed rate is 10 MB a fortnight. Nearly all of the
growth is loading the Piper voice and onnxruntime warming its arena allocator
over the first few synthesis calls -- which converges, and matches the plateau
the soak actually showed.

Re-run this after any change to the audio or model path.
"""
import ctypes
import gc
import tempfile
from ctypes import wintypes
from pathlib import Path

class PMC(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

_k32 = ctypes.windll.kernel32
_k32.GetCurrentProcess.restype = ctypes.c_void_p
_gpmi = getattr(_k32, "K32GetProcessMemoryInfo", None) or ctypes.windll.psapi.GetProcessMemoryInfo
_gpmi.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), wintypes.DWORD]
_gpmi.restype = wintypes.BOOL


def rss_mb():
    c = PMC(); c.cb = ctypes.sizeof(PMC)
    if not _gpmi(_k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
        raise OSError(f"GetProcessMemoryInfo failed: {ctypes.GetLastError()}")
    return c.WorkingSetSize / 1e6


def mark(label):
    gc.collect()
    print(f"  {label:<46} {rss_mb():>8.1f} MB", flush=True)

mark("interpreter start")

from runtime.session import RETAINED_DROPS, RETAINED_UTTERANCES
mark("goldlive imports")

# 1. The collections I bounded -- how much do they actually hold?
from collections import deque
tr = deque(maxlen=RETAINED_UTTERANCES); dr = deque(maxlen=RETAINED_DROPS)
for i in range(20_000):
    text = f"utterance {i} " + "x" * 400
    (dr.append((text, 0.5)) if i % 4 else tr.append(text))
mark("20k utterances through bounded deques")

# 2. What an UNBOUNDED version would have cost over the same workload
unbounded = []
for i in range(20_000):
    unbounded.append(f"utterance {i} " + "x" * 400)
mark("the same 20k retained unbounded (the old behaviour)")
del unbounded; gc.collect()
mark("after releasing the unbounded copy")

# 3. Piper + onnxruntime: loaded once, then repeated synthesis
try:
    from platform_.tts.piper import PiperTTS
    from shared.paths import data_root
    import asyncio
    tts = PiperTTS(voices_dir=data_root() / "voices")
    voice = "en_US-john-medium"
    tmp = Path(tempfile.mkdtemp())
    asyncio.run(tts.synthesize("Warming the voice up.", voice, tmp / "w.wav"))
    mark("piper voice loaded (one-off cost)")
    for n in (1, 10, 25):
        for i in range(n):
            asyncio.run(tts.synthesize(
                f"Gold is holding just under the level we marked earlier, number {i}.",
                voice, tmp / f"s{i}.wav"))
        mark(f"after {n} more synthesis calls")
except Exception as e:
    print("  piper probe unavailable:", e)
