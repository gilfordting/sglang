"""CPU coverage for the decode-side DFlash draft ring fed by prefill's draft-KV
tail (--speculative-dflash-draft-ring on a PD decode worker).

Prefill keeps the full draft KV and sends positions [tail_start, seq_len) as
the DFLASH_KV state component; decode must hand NIXL the same number of
destination pages, each one a ring page of the request's row. No GPU.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from sglang.srt.mem_cache import memory_pool
from sglang.srt.mem_cache.common import kv_to_page_indices
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.speculative.dflash_kv_layout import (
    draft_kv_transfer_start,
    draft_ring_tail_slots,
    draft_ring_transfer_page_indices,
)
from sglang.srt.speculative.dflash_utils import compute_dflash_draft_ring_geometry
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

WINDOW = 4096
BLOCK = 8
PAGE = 64
RING, KEEP = compute_dflash_draft_ring_geometry(
    draft_window_size=WINDOW, block_size=BLOCK, page_size=PAGE
)
OFFSET = PAGE  # the pool's first page is the reserved padding slot

SEQ_LENS = [
    1,
    63,
    64,
    65,
    WINDOW - 1,
    WINDOW,
    WINDOW + 1,
    WINDOW + PAGE - 1,
    WINDOW + PAGE,
    WINDOW + PAGE + 1,
    RING - 1,
    RING,
    RING + 1,
    RING + WINDOW + 3,
    2 * RING + PAGE,
    10 * RING + 17,
    123_457,
]


def _prefill_pages(seq_len: int, page: int = PAGE, window: int = WINDOW):
    """What disaggregation/prefill.py computes for DFLASH_KV: page ids of the
    target req_to_token row over [tail_start, seq_len). Logical slots are
    arbitrary here (page-aligned allocation, scrambled page order)."""
    tail_start = draft_kv_transfer_start(seq_len, window, page)
    n_pages = -(-seq_len // page)
    perm = torch.randperm(n_pages, generator=torch.Generator().manual_seed(seq_len))
    row = perm.repeat_interleave(page) * page + torch.arange(n_pages * page) % page
    row = row + page  # skip the padding page like the allocator does
    return kv_to_page_indices(row[tail_start:seq_len], page)


def _install_real_ring_map(rows: int, page: int = PAGE):
    """Run DFlashWorkerV2._install_draft_ring_map on a CPU table so the pure
    index math is checked against the map the draft actually uses."""
    from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

    width = max(SEQ_LENS) + PAGE
    table = torch.zeros((rows + 1, width), dtype=torch.int32)
    fake = SimpleNamespace(
        draft_model_runner=SimpleNamespace(
            req_to_token_pool=SimpleNamespace(req_to_token=table),
            token_to_kv_pool=SimpleNamespace(size=(rows + 3) * RING),
        ),
        draft_ring_tokens=RING,
        page_size=page,
        ps=SimpleNamespace(tp_rank=1),
        draft_window_size=WINDOW,
        block_size=BLOCK,
        _ring_page_offset=0,
    )
    DFlashWorkerV2._install_draft_ring_map(fake, target_rows=rows)
    assert fake._ring_page_offset == OFFSET
    return table


@pytest.mark.parametrize("seq_len", SEQ_LENS)
@pytest.mark.parametrize("row", [0, 1, 7])
def test_decode_ring_pages_match_prefill_page_count_and_real_map(seq_len, row):
    table = _install_real_ring_map(rows=8)
    tail_start = draft_kv_transfer_start(seq_len, WINDOW, PAGE)
    from_table = kv_to_page_indices(table[row, tail_start:seq_len], PAGE)
    pure = draft_ring_transfer_page_indices(
        req_pool_idx=row,
        seq_len=seq_len,
        window_size=WINDOW,
        page_size=PAGE,
        ring_tokens=RING,
        ring_page_offset=OFFSET,
    )
    # Same count as the sender (conn.py raises on a length mismatch) ...
    assert len(pure) == len(_prefill_pages(seq_len))
    # ... and exactly the pages the draft's own req_to_token addresses.
    np.testing.assert_array_equal(pure, from_table)
    assert pure.dtype == np.int32


@pytest.mark.parametrize("seq_len", SEQ_LENS)
def test_tail_pages_are_whole_distinct_pages_of_the_row(seq_len):
    row = 5
    slots = draft_ring_tail_slots(
        req_pool_idx=row,
        seq_len=seq_len,
        window_size=WINDOW,
        page_size=PAGE,
        ring_tokens=RING,
        ring_page_offset=OFFSET,
    )
    tail_start = draft_kv_transfer_start(seq_len, WINDOW, PAGE)
    assert len(slots) == seq_len - tail_start
    # Every slot belongs to this row's ring region, never the padding page.
    lo, hi = OFFSET + row * RING, OFFSET + (row + 1) * RING
    assert slots.min() >= lo and slots.max() < hi
    # No two tail positions share a slot (the tail is shorter than the ring).
    assert len(set(slots.tolist())) == len(slots)
    # Each logical page of the tail lands in exactly one ring page, with the
    # slot offset inside that page equal to the position offset -- the wrap
    # (p % ring) can only fall on a page boundary.
    positions = np.arange(tail_start, seq_len)
    for p0 in range(tail_start, seq_len, PAGE):
        sel = (positions >= p0) & (positions < p0 + PAGE)
        page_slots = slots[sel]
        assert page_slots[0] % PAGE == 0
        np.testing.assert_array_equal(page_slots - page_slots[0], np.arange(sel.sum()))
    pages = draft_ring_transfer_page_indices(
        req_pool_idx=row,
        seq_len=seq_len,
        window_size=WINDOW,
        page_size=PAGE,
        ring_tokens=RING,
        ring_page_offset=OFFSET,
    )
    assert len(set(pages.tolist())) == len(pages)


def test_wrap_around_splits_contiguity_only_at_a_page_boundary():
    # Tail straddles the ring end: positions RING-128 .. RING+64.
    seq_len = RING + 64
    tail_start = draft_kv_transfer_start(seq_len, WINDOW, PAGE)
    assert tail_start < RING < seq_len
    slots = draft_ring_tail_slots(
        req_pool_idx=0,
        seq_len=seq_len,
        window_size=WINDOW,
        page_size=PAGE,
        ring_tokens=RING,
        ring_page_offset=OFFSET,
    )
    positions = np.arange(tail_start, seq_len)
    wrapped = positions >= RING
    np.testing.assert_array_equal(slots[~wrapped], OFFSET + positions[~wrapped])
    np.testing.assert_array_equal(slots[wrapped], OFFSET + positions[wrapped] - RING)
    # The single discontinuity sits exactly at position RING (page-aligned).
    jumps = np.flatnonzero(np.diff(slots) != 1)
    assert jumps.tolist() == [RING - tail_start - 1]


@pytest.mark.parametrize("seq_len", SEQ_LENS)
def test_transferred_tail_covers_every_future_read_window(seq_len):
    """After transfer decode writes only positions >= seq_len. Every later
    draft read window (window back from the prefix, page-aligned on the
    left, as paged backends do) must start inside the transferred tail, and
    the tail plus one draft block must fit the ring without aliasing."""
    tail_start = draft_kv_transfer_start(seq_len, WINDOW, PAGE)
    for future in (seq_len, seq_len + 1, seq_len + BLOCK, seq_len + 3 * RING):
        read_start = draft_kv_transfer_start(future, WINDOW, PAGE)
        assert read_start >= tail_start
    assert (seq_len - tail_start) <= KEEP
    assert KEEP + BLOCK <= RING


def test_mismatched_window_cannot_wrap_onto_itself():
    with pytest.raises(ValueError, match="exceeds the ring"):
        draft_ring_tail_slots(
            req_pool_idx=0,
            seq_len=3 * RING,
            window_size=2 * RING,
            page_size=PAGE,
            ring_tokens=RING,
            ring_page_offset=OFFSET,
        )
    with pytest.raises(ValueError, match="page-aligned"):
        draft_ring_tail_slots(
            req_pool_idx=0,
            seq_len=100,
            window_size=WINDOW,
            page_size=PAGE,
            ring_tokens=RING + 1,
            ring_page_offset=OFFSET,
        )


# --- decode.py wiring -------------------------------------------------------


def _ring_pd(draft_pool):
    from sglang.srt.disaggregation.decode import _DFlashDraftRingPD

    worker = SimpleNamespace(draft_ring_tokens=RING, _ring_page_offset=OFFSET)
    args = SimpleNamespace(speculative_draft_window_size=WINDOW)
    return _DFlashDraftRingPD.from_draft_worker(worker, draft_pool, args)


def test_ring_pd_refuses_uninstalled_ring():
    from sglang.srt.disaggregation.decode import _DFlashDraftRingPD

    pool = SimpleNamespace(page_size=PAGE)
    with pytest.raises(RuntimeError, match="not installed"):
        _DFlashDraftRingPD.from_draft_worker(
            SimpleNamespace(draft_ring_tokens=RING, _ring_page_offset=0),
            pool,
            SimpleNamespace(speculative_draft_window_size=WINDOW),
        )
    with pytest.raises(RuntimeError, match="draft-window-size"):
        _DFlashDraftRingPD.from_draft_worker(
            SimpleNamespace(draft_ring_tokens=RING, _ring_page_offset=OFFSET),
            pool,
            SimpleNamespace(speculative_draft_window_size=None),
        )


@pytest.mark.parametrize("seq_len", [1, WINDOW + 5, RING + 64, 3 * RING + 1])
def test_retraction_backup_moves_ring_tail_between_rows(monkeypatch, seq_len):
    monkeypatch.setattr(memory_pool.current_platform, "synchronize", lambda: None)
    rows = 4
    draft = MHATokenToKVPool(
        size=(rows + 2) * RING,
        page_size=PAGE,
        dtype=torch.float32,
        head_num=1,
        head_dim=1,
        layer_num=2,
        device="cpu",
        enable_memory_saver=False,
        enable_alt_stream=False,
    )
    ring = _ring_pd(draft)
    old_row, new_row = 1, 3
    # Fill row `old_row` as the ring would hold it: slot value = position.
    tail_start = draft_kv_transfer_start(seq_len, WINDOW, PAGE)
    positions = torch.arange(tail_start, seq_len, dtype=torch.float32)
    old_slots = torch.from_numpy(ring.tail_slots(old_row, seq_len))
    for layer in range(2):
        draft.k_buffer[layer].fill_(-1)
        draft.v_buffer[layer].fill_(-1)
        draft.k_buffer[layer][old_slots, 0, 0] = positions + layer
        draft.v_buffer[layer][old_slots, 0, 0] = -positions - layer
    cpu = ring.offload(old_row, seq_len)
    for layer in range(2):
        draft.k_buffer[layer].fill_(-7)
        draft.v_buffer[layer].fill_(-7)
    ring.load(cpu, new_row, seq_len)
    new_slots = torch.from_numpy(ring.tail_slots(new_row, seq_len))
    for layer in range(2):
        torch.testing.assert_close(
            draft.k_buffer[layer][new_slots, 0, 0], positions + layer
        )
        torch.testing.assert_close(
            draft.v_buffer[layer][new_slots, 0, 0], -positions - layer
        )
        # Nothing outside the new row's tail was touched.
        mask = torch.ones(draft.k_buffer[layer].shape[0], dtype=torch.bool)
        mask[new_slots] = False
        assert (draft.k_buffer[layer][mask] == -7).all()


def test_req_offload_and_load_route_through_the_ring_hook():
    from array import array

    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    req = Req(
        rid="ring-retract",
        origin_input_text="",
        origin_input_ids=array("q", range(10)),
        sampling_params=SamplingParams(),
    )
    req.req_pool_idx = 2
    req.mamba_pool_idx = None
    ring = MagicMock()
    ring.offload.return_value = "cpu-tail"
    kvcache = SimpleNamespace(_pd_dflash_draft_ring=ring)
    allocator = MagicMock()
    allocator.get_kvcache.return_value = kvcache
    allocator.get_cpu_copy.return_value = "cpu-kv"
    table = SimpleNamespace(req_to_token=torch.arange(64).reshape(4, 16))

    req.offload_kv_cache(table, allocator)
    ring.offload.assert_called_once_with(2, req.seqlen - 1)
    assert req.dflash_ring_cpu == "cpu-tail"

    req.req_pool_idx = 0  # resumed on another row
    req.load_kv_cache(table, allocator)
    ring.load.assert_called_once_with("cpu-tail", 0, req.seqlen - 1)
    assert req.dflash_ring_cpu is None

    # Without the hook the generic path is untouched.
    plain = MagicMock()
    plain.get_kvcache.return_value = SimpleNamespace()
    req.offload_kv_cache(table, plain)
    plain.get_cpu_copy.assert_called_once()
    req.load_kv_cache(table, plain)
    plain.load_cpu_copy.assert_called_once()


# --- server_args validation -------------------------------------------------


def _ring_args(**overrides):
    base = dict(
        speculative_dflash_draft_ring=True,
        speculative_algorithm="DFLASH",
        speculative_draft_window_size=WINDOW,
        speculative_num_draft_tokens=BLOCK,
        disaggregation_mode="null",
        disaggregation_transfer_backend="mooncake",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _validate(args):
    from sglang.srt.server_args import ServerArgs

    ServerArgs._validate_dflash_draft_ring(args)


def test_ring_validation_accepts_decode_with_nixl_and_non_pd():
    _validate(_ring_args())
    _validate(
        _ring_args(disaggregation_mode="decode", disaggregation_transfer_backend="nixl")
    )
    _validate(
        _ring_args(speculative_dflash_draft_ring=False, disaggregation_mode="prefill")
    )


def test_ring_validation_rejects_prefill_and_non_nixl_decode():
    with pytest.raises(ValueError, match="decode-only"):
        _validate(
            _ring_args(
                disaggregation_mode="prefill", disaggregation_transfer_backend="nixl"
            )
        )
    with pytest.raises(ValueError, match="nixl"):
        _validate(_ring_args(disaggregation_mode="decode"))


def test_ring_validation_rejects_prefill_with_hicache_dcp():
    from sglang.srt.server_args import ServerArgs

    args = SimpleNamespace(
        dcp_size=8,
        enable_hierarchical_cache=True,
        hicache_storage_backend=None,
        speculative_algorithm="DFLASH",
        speculative_dflash_draft_ring=True,
    )
    with pytest.raises(NotImplementedError, match="speculative"):
        ServerArgs._resolve_hicache_dcp_compatibility(args)
    # Full draft KV (the prefill configuration) passes this check.
    args.speculative_dflash_draft_ring = False
    args.enable_lmcache = False
    args.enable_hisparse = False
    args.get_model_config = lambda: SimpleNamespace(
        hf_config=SimpleNamespace(architectures=["KimiK3ForConditionalGeneration"])
    )
    ServerArgs._resolve_hicache_dcp_compatibility(args)
