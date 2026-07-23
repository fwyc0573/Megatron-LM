"""Unit tests for communication time estimator module."""

from __future__ import annotations

import pathlib
import sys
import unittest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.comm_sim.nccl_comm import get_comm_op_exc_time


class TestCommOpExcTime(unittest.TestCase):
    def setUp(self) -> None:
        # 8 unique ranks to match real collective group semantics.
        self.comm_group = list(range(8))
        self.data_sizes_to_test = [
            512,
            1024,
            32768,
            65536,
            262144,
            33554433,
            100000000,
            134217728,
            200000000,
            268435456,
            500000000,
        ]

    def test_allreduce_estimation(self) -> None:
        for data_size in self.data_sizes_to_test:
            result = get_comm_op_exc_time(self.comm_group, data_size, "allreduce")
            self.assertGreater(result, 0)

    def test_send_recv_estimation(self) -> None:
        for data_size in self.data_sizes_to_test:
            result = get_comm_op_exc_time(self.comm_group[:2], data_size, "send_recv")
            self.assertGreater(result, 0)


if __name__ == "__main__":
    unittest.main()
