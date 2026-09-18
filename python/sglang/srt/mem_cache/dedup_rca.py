"""Rank-divergence forensics for the HiCache host-dedup load-back path.

Debug-only, gated by SGLANG_DEDUP_RCA=1. Never enabled in production.

What it does
------------
1. ``log(tag, **fields)``: appends one line per rank-local decision to
   ``$SGLANG_DEDUP_RCA_DIR/rank<r>.log`` (default /tmp/dedup_rca) with a per-rank
   sequence number, so the per-rank decision streams can be diffed offline.
2. ``fingerprint(tag, fields)``: all-gathers a small tuple of ints over the
   TP CPU (gloo) group at a program point every rank reaches, compares, and
   on the FIRST mismatch logs every rank's tuple plus the local ring buffer
   and raises (fail-stop) -- BEFORE any NCCL collective that would otherwise
   hang forever with mismatched participants.
3. ``heartbeat()``: re-arms a faulthandler all-thread stack dump so a stalled
   scheduler loop prints where every thread sits instead of idling silently.
"""

from __future__ import annotations

import faulthandler
import os
import sys
import threading
import time
import zlib
from collections import deque
from typing import Any, Iterable, Optional

import torch

_ENABLED: Optional[bool] = None
_LOG_FILE = None
_SEQ = 0
_RING: deque = deque(maxlen=int(os.environ.get("SGLANG_DEDUP_RCA_RING", "400")))
_RANK = -1
_GROUP = None
_LOCK = threading.Lock()
_STALL_SECONDS = float(os.environ.get("SGLANG_DEDUP_RCA_STALL_SECONDS", "90"))
_MISMATCHES = 0
_FAIL_STOP = os.environ.get("SGLANG_DEDUP_RCA_FAIL_STOP", "1") == "1"
# SGLANG_DEDUP_RCA_FP=0 keeps the decision logs + stall dump but skips the
# gloo fingerprints, so the production hang signature can reproduce as-is.
_FP_ENABLED = os.environ.get("SGLANG_DEDUP_RCA_FP", "1") == "1"
# Fault injection: on rank INJECT_RANK, the INJECT_AFTER-th load-back
# (1-based) is forced to fail as if the device allocator returned None.
_INJECT_RANK = int(os.environ.get("SGLANG_DEDUP_RCA_INJECT_LOAD_FAIL_RANK", "-1"))
_INJECT_AFTER = int(os.environ.get("SGLANG_DEDUP_RCA_INJECT_LOAD_FAIL_AFTER", "0"))
_INJECT_COUNT = 0


def enabled() -> bool:
    global _ENABLED
    if _ENABLED is None:
        _ENABLED = os.environ.get("SGLANG_DEDUP_RCA", "0") == "1"
    return _ENABLED


def _rank() -> int:
    global _RANK
    if _RANK < 0:
        try:
            _RANK = torch.distributed.get_rank()
        except Exception:
            _RANK = int(os.environ.get("RANK", "0"))
    return _RANK


def _log_file():
    global _LOG_FILE
    if _LOG_FILE is None:
        d = os.environ.get("SGLANG_DEDUP_RCA_DIR", "/tmp/dedup_rca")
        os.makedirs(d, exist_ok=True)
        _LOG_FILE = open(os.path.join(d, f"rank{_rank()}.log"), "a", buffering=1)
    return _LOG_FILE


def crc(indices) -> int:
    """CRC32 over an int index tensor / list (order-sensitive)."""
    if indices is None:
        return 0
    if isinstance(indices, torch.Tensor):
        if indices.numel() == 0:
            return 0
        return zlib.crc32(indices.detach().cpu().to(torch.int64).numpy().tobytes())
    return zlib.crc32(",".join(str(int(i)) for i in indices).encode())


def log(tag: str, **fields: Any) -> None:
    """Append one decision line to the per-rank log and the ring buffer."""
    if not enabled():
        return
    global _SEQ
    with _LOCK:
        _SEQ += 1
        seq = _SEQ
        parts = " ".join(f"{k}={_fmt(v)}" for k, v in fields.items())
        line = f"{time.time():.6f} seq={seq} {tag} {parts}"
        _RING.append(line)
        try:
            _log_file().write(line + "\n")
        except Exception:
            pass


def _fmt(v: Any) -> str:
    if isinstance(v, torch.Tensor):
        if v.numel() <= 4:
            return str(v.tolist())
        return f"T[n={v.numel()} crc={crc(v):08x} first={int(v.flatten()[0])} last={int(v.flatten()[-1])}]"
    if isinstance(v, float):
        return f"{v:.4f}"
    if isinstance(v, (list, tuple)) and len(v) > 12:
        return f"{type(v).__name__}[n={len(v)} crc={crc(list(map(int, v))) if all(isinstance(x, int) for x in v) else '?'}]"
    return str(v)


def set_group(group) -> None:
    """Register the CPU (gloo) process group used for fingerprints."""
    global _GROUP
    _GROUP = group


def _cpu_group():
    if _GROUP is not None:
        return _GROUP
    try:
        from sglang.srt.distributed import get_tp_group

        return get_tp_group().cpu_group
    except Exception:
        return None


def fingerprint(tag: str, fields: dict, extra: Optional[dict] = None) -> bool:
    """All-gather ``fields`` (name->int) over the TP CPU group; fail-stop on mismatch.

    Returns True when all ranks agree. ``extra`` is rank-local detail that is
    only printed when a mismatch is detected.
    """
    if not enabled() or not _FP_ENABLED:
        return True
    group = _cpu_group()
    if group is None or not torch.distributed.is_initialized():
        return True
    world = torch.distributed.get_world_size(group=group)
    if world <= 1:
        return True
    keys = list(fields.keys())
    local = torch.tensor([int(fields[k]) for k in keys], dtype=torch.int64)
    gathered = [torch.empty_like(local) for _ in range(world)]
    torch.distributed.all_gather(gathered, local, group=group)
    same = all(torch.equal(g, local) for g in gathered)
    log(f"fp:{tag}", ok=int(same), **fields)
    if same:
        return True
    global _MISMATCHES
    _MISMATCHES += 1
    rank = _rank()
    lines = [
        f"[DEDUP_RCA] rank {rank}: FINGERPRINT MISMATCH at '{tag}' (#{_MISMATCHES})",
        f"[DEDUP_RCA] keys: {keys}",
    ]
    for r, g in enumerate(gathered):
        vals = g.tolist()
        diff = [keys[i] for i in range(len(keys)) if vals[i] != local[i].item()]
        lines.append(
            f"[DEDUP_RCA]   rank {r}: {dict(zip(keys, vals))}"
            + (f"  DIFFERS_FROM_ME_IN={diff}" if r != rank and diff else "")
        )
    if extra:
        lines.append(f"[DEDUP_RCA] rank {rank} local detail: {extra}")
    lines.append(f"[DEDUP_RCA] rank {rank} last {len(_RING)} decisions:")
    lines.extend(f"[DEDUP_RCA]   {l}" for l in _RING)
    text = "\n".join(lines)
    try:
        _log_file().write(text + "\n")
    except Exception:
        pass
    print(text, file=sys.stderr, flush=True)
    if _FAIL_STOP:
        raise RuntimeError(
            f"[DEDUP_RCA] rank {rank}: TP ranks diverged at '{tag}'; see "
            f"{os.environ.get('SGLANG_DEDUP_RCA_DIR', '/tmp/dedup_rca')}/rank*.log"
        )
    return False


def heartbeat() -> None:
    """Re-arm the stall dump: if no heartbeat for STALL_SECONDS, dump all threads."""
    if not enabled() or _STALL_SECONDS <= 0:
        return
    try:
        faulthandler.dump_traceback_later(
            _STALL_SECONDS, repeat=True, file=_log_file(), exit=False
        )
    except Exception:
        pass


def summarize_ops(ops: Iterable) -> dict:
    """Ints describing a list of CacheOperation (pending load/write queue)."""
    n_ops = 0
    kv_tokens = 0
    host_crc = 0
    dev_crc = 0
    n_xfers = 0
    xfer_tokens = 0
    node_ids = []
    for op in ops:
        n_ops += 1
        h = getattr(op, "host_indices", None)
        d = getattr(op, "device_indices", None)
        if h is not None:
            kv_tokens += int(h.numel())
            host_crc = zlib.crc32(crc(h).to_bytes(4, "little"), host_crc)
        if d is not None:
            dev_crc = zlib.crc32(crc(d).to_bytes(4, "little"), dev_crc)
        for t in getattr(op, "pool_transfers", None) or []:
            n_xfers += 1
            th = getattr(t, "host_indices", None)
            if th is not None:
                xfer_tokens += int(th.numel())
        node_ids.extend(getattr(op, "node_ids", []))
    return {
        "n_ops": n_ops,
        "kv_tokens": kv_tokens,
        "host_crc": host_crc,
        "dev_crc": dev_crc,
        "n_xfers": n_xfers,
        "xfer_tokens": xfer_tokens,
        "nodes_crc": crc(node_ids),
    }


def node_counter() -> int:
    """Global radix node id counter (deterministic iff insert order is identical)."""
    try:
        from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeNode

        return int(UnifiedTreeNode.counter)
    except Exception:
        return -1


def inject_load_fail() -> bool:
    """True exactly once, on the configured rank's N-th load-back (fault injection)."""
    global _INJECT_COUNT
    if not enabled() or _INJECT_RANK < 0 or _rank() != _INJECT_RANK:
        return False
    _INJECT_COUNT += 1
    if _INJECT_COUNT == _INJECT_AFTER:
        log("INJECT_LOAD_FAIL", count=_INJECT_COUNT)
        print(f"[DEDUP_RCA] rank {_rank()}: INJECTING load-back failure #{_INJECT_COUNT}", file=sys.stderr, flush=True)
        return True
    return False
