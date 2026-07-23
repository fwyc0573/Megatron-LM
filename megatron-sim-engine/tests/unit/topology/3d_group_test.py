import unittest
from src.core.static_graphs.parallel_group_manager import ParallelGroupManager

class TestParallelGroupManager(unittest.TestCase):
    def test_groups(self):
        local_size = 4
        world_size = 4
        pp_size = 2
        tp_size = 2

        manager = ParallelGroupManager(local_size=4, world_size=4, pp_size=2, tp_size=2)
        print(f"manager.get_dp_groups() -> : {manager.get_dp_groups()}")
        print(f"manager.get_pp_groups() -> : {manager.get_pp_groups()}")
        print(f"manager.get_tp_groups() -> : {manager.get_tp_groups()}")
        print(f"manager.get_mp_groups() -> : {manager.get_mp_groups()}")


if __name__ == '__main__':
    unittest.main()