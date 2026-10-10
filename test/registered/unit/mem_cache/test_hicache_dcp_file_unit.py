"""
Unit tests for HiCache L3 (file backend) under DCP: every DCP rank backs up its
own interleaved MLA shard of a widened logical page under a rank-qualified key,
and a prefetch into different host slots restores each token to the physical
row the L2 transfer path (``dcp_kernel_indices``) expects.

These are pure CPU tests; they do not launch a server or need CUDA.
Run with:
    python3 -m pytest test/registered/unit/mem_cache/test_hicache_dcp_file_unit.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import shutil
import tempfile
import unittest
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.hicache_storage import HiCacheFile, HiCacheStorageConfig
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.test.test_utils import CustomTestCase

DCP = 8
PHYS_PAGE = 4
LOGICAL_PAGE = PHYS_PAGE * DCP
LAYERS = 3
DIM = 2
PHYS_PAGES = 6


def _host_pool(layout: str, dcp_rank: int) -> MLATokenToKVPoolHost:
    """A DCP-sharded MLA host pool with only the fields the L3 page path uses."""
    pool = object.__new__(MLATokenToKVPoolHost)
    pool.layout = layout
    pool.dcp_size = DCP
    pool.dcp_rank = dcp_rank
    pool.page_size = PHYS_PAGE
    pool.logical_page_size = LOGICAL_PAGE
    pool.layer_num = LAYERS
    pool.kv_cache_dim = DIM
    pool.dtype = torch.float32
    pool.device = "cpu"
    pool.pin_memory = False
    rows = PHYS_PAGE * PHYS_PAGES
    if layout == "layer_first":
        pool.kv_buffer = torch.zeros(LAYERS, rows, 1, DIM)
    else:
        pool.kv_buffer = torch.zeros(rows, LAYERS, 1, DIM)
    return pool


def _row(pool, physical_row: int) -> torch.Tensor:
    if pool.layout == "layer_first":
        return pool.kv_buffer[:, physical_row]
    return pool.kv_buffer[physical_row]


def _fill_logical_page(pool, logical_start: int) -> None:
    """Write each owned logical token's id into its physical row (all layers)."""
    logical = torch.arange(logical_start, logical_start + LOGICAL_PAGE)
    owned = logical[logical % DCP == pool.dcp_rank]
    for token, row in zip(owned.tolist(), pool.dcp_kernel_indices(logical).tolist()):
        _row(pool, row).fill_(float(token))


def _backend(root: str, tp_rank: int, is_rank_replicated: bool) -> HiCacheFile:
    config = HiCacheStorageConfig(
        tp_rank=tp_rank,
        tp_size=DCP,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=is_rank_replicated,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name="kimi-k3",
    )
    with mock.patch.object(
        envs.SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR, "get", return_value=root
    ):
        return HiCacheFile(config)


class TestHiCacheDcpFileRoundTrip(CustomTestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="hicache-dcp-")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _round_trip(self, layout: str, is_rank_replicated: bool) -> list:
        src_pages = [1 * LOGICAL_PAGE, 4 * LOGICAL_PAGE]
        dst_pages = [5 * LOGICAL_PAGE, 2 * LOGICAL_PAGE]
        keys = ["hash-a", "hash-b"]

        for rank in range(DCP):
            pool = _host_pool(layout, rank)
            for start in src_pages:
                _fill_logical_page(pool, start)
            backend = _backend(self.root, rank, is_rank_replicated)
            data = [pool.get_data_page(start) for start in src_pages]
            self.assertTrue(backend.batch_set(keys, data))

        mismatches = []
        for rank in range(DCP):
            pool = _host_pool(layout, rank)
            backend = _backend(self.root, rank, is_rank_replicated)
            pages = backend.batch_get(
                keys, [pool.get_dummy_flat_data_page() for _ in keys]
            )
            for src, dst, page in zip(src_pages, dst_pages, pages):
                self.assertIsNotNone(page)
                pool.set_from_flat_data_page(dst, page)
                dst_logical = torch.arange(dst, dst + LOGICAL_PAGE)
                rows = pool.dcp_kernel_indices(dst_logical).tolist()
                owned = dst_logical[dst_logical % DCP == rank].tolist()
                for dst_token, row in zip(owned, rows):
                    expected = float(dst_token - dst + src)
                    if not torch.all(_row(pool, row) == expected):
                        mismatches.append((rank, dst_token))
        return mismatches

    def test_round_trip_page_first(self):
        self.assertEqual(self._round_trip("page_first", False), [])

    def test_round_trip_layer_first(self):
        self.assertEqual(self._round_trip("layer_first", False), [])

    def test_rank_replicated_keys_collide_under_dcp(self):
        # The pre-DCP MLA keying shares one file per page across ranks, so all
        # but the last writer's shard is lost.
        self.assertNotEqual(self._round_trip("page_first", True), [])

    def test_unaligned_page_index_rejected(self):
        pool = _host_pool("page_first", 0)
        with self.assertRaises(AssertionError):
            pool.get_data_page(PHYS_PAGE)


if __name__ == "__main__":
    unittest.main()
