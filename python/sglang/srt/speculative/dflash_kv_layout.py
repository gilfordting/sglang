"""Layout accounting for a TP-sharded draft with a DCP-sharded target."""

import numpy as np


def draft_kv_bytes_per_target_token(
    *,
    num_layers: int,
    total_kv_heads: int,
    head_dim: int,
    tp_size: int,
    dcp_size: int,
    dtype: str,
) -> int:
    """Draft bytes per physical target row, covering DCP logical token slots.

    The draft partitions heads by TP and replicates the target's token domain.
    Each physical target row therefore backs dcp_size draft rows on each rank.
    """
    if min(num_layers, total_kv_heads, head_dim, tp_size, dcp_size) <= 0:
        raise ValueError("Draft KV dimensions and parallel sizes must be positive")
    if total_kv_heads >= tp_size and total_kv_heads % tp_size:
        raise ValueError("Draft KV heads must divide evenly across TP ranks")
    if total_kv_heads < tp_size and tp_size % total_kv_heads:
        raise ValueError("TP ranks must divide evenly across replicated KV heads")
    dtype = (dtype or "").lower()
    if dtype in ("bf16", "bfloat16", "fp16", "float16"):
        element_bytes = 2
    elif dtype == "float32":
        element_bytes = 4
    elif dtype in ("fp8_e4m3", "fp8_e5m2", "fp8", "mxfp8"):
        element_bytes = 1
    else:
        raise ValueError(f"Unresolved draft KV dtype: {dtype}")
    return (
        num_layers
        * 2
        * max(1, total_kv_heads // tp_size)
        * head_dim
        * element_bytes
        * dcp_size
    )


def draft_kv_transfer_start(
    seq_len: int, window_size: int | None, page_size: int
) -> int:
    """First logical token position needed by the draft, rounded to a page."""
    if seq_len < 0 or page_size <= 0 or (window_size is not None and window_size <= 0):
        raise ValueError("Invalid draft transfer extent")
    start = max(0, seq_len - window_size) if window_size is not None else 0
    return start // page_size * page_size


def draft_ring_tail_slots(
    *,
    req_pool_idx: int,
    seq_len: int,
    window_size: int,
    page_size: int,
    ring_tokens: int,
    ring_page_offset: int,
):
    """Draft-ring slots (one per token) for the transferred draft-KV tail
    [draft_kv_transfer_start(seq_len), seq_len) of request row
    ``req_pool_idx``, in ascending position order.

    Mirrors the static map installed by DFlashWorkerV2._install_draft_ring_map:
    row r, position p -> ring_page_offset + r * ring_tokens + p % ring_tokens.
    The ring is the DECODE side's private draft pool under
    --speculative-dflash-draft-ring; the PREFILL side keeps the full draft KV
    and sends the same positions out of its shared target index space, so the
    two sides' page lists line up positionally.

    Page structure: tail_start, ring_tokens and ring_page_offset are all
    page-aligned, so each logical page of the tail maps onto exactly one ring
    page (the wrap at p % ring_tokens can only happen on a page boundary).
    """
    page = int(page_size)
    ring = int(ring_tokens)
    offset = int(ring_page_offset)
    if page <= 0 or ring <= 0 or req_pool_idx < 0:
        raise ValueError("Invalid draft ring geometry")
    if ring % page or offset % page:
        raise ValueError(
            f"Draft ring geometry must be page-aligned: ring_tokens={ring} "
            f"ring_page_offset={offset} page_size={page}"
        )
    tail_start = draft_kv_transfer_start(seq_len, window_size, page)
    if seq_len - tail_start > ring:
        # Would wrap onto itself; compute_dflash_draft_ring_geometry sizes
        # the ring for window + page + block, so this cannot happen unless
        # the two sides disagree on the window.
        raise ValueError(
            f"Draft tail of {seq_len - tail_start} tokens exceeds the ring "
            f"({ring} tokens); prefill/decode draft windows disagree?"
        )
    positions = np.arange(tail_start, seq_len, dtype=np.int64)
    return offset + int(req_pool_idx) * ring + positions % ring


def draft_ring_transfer_page_indices(
    *,
    req_pool_idx: int,
    seq_len: int,
    window_size: int,
    page_size: int,
    ring_tokens: int,
    ring_page_offset: int,
):
    """Destination page ids (int32, in the decode draft ring pool) for the
    DFLASH_KV state component, one per transferred draft page, matching the
    prefill side's ``kv_to_page_indices(req_to_token[tail_start:seq_len])``
    in count and order."""
    slots = draft_ring_tail_slots(
        req_pool_idx=req_pool_idx,
        seq_len=seq_len,
        window_size=window_size,
        page_size=page_size,
        ring_tokens=ring_tokens,
        ring_page_offset=ring_page_offset,
    )
    page = int(page_size)
    # Same reduction as kv_to_page_indices: first slot of each page // page.
    return (slots[::page] // page).astype("int32")
