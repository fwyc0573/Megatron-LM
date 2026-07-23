from src.core.static_graphs.parallel_group_manager import ParallelGroupManager
from src.core.static_graphs.model_layer_allocator import ModelLayerAllocator
import json
from typing import Dict, List


class StaticGraphsModule:
    def __init__(self, args, json_save=False):
        self.args = args
        self.json_save = json_save
        self._validate_args()
        self._standard_abstract_model = self._get_standard_abstract_model()

        self.parallel_group_manager = ParallelGroupManager(
            local_size=self.args.nproc_per_node,
            world_size=self.args.nproc_per_node * self.args.nnodes,
            pp_size=self.args.pipeline_model_parallel_size,
            tp_size=self.args.tensor_model_parallel_size
        )
        
        self.model_layer_allocator = ModelLayerAllocator(
            groups=self.parallel_group_creator.get_dp_groups(),
            total_layers=self.args.num_layers
        )

        self.all_groups: Dict[str, List[int]] = self.generate_parallel_group()
        
    def _validate_args(self):
        required_base_args = ['nproc_per_node', 'nnodes', 'model-type', 'tensor-model-parallel-size', 'pipeline-model-parallel-size', 'micro_batch_size', 'global_batch_size']
        model_specific_args = {
            'gpt': ['num_layers', 'hidden_size', 'num_attention_heads', 'seq_length', 'max_position_embeddings'],
            'llama': ['num_layers', 'hidden_size', 'num_attention_heads', 'seq_length', 'max_position_embeddings'],
            'bert': ['num_layers', 'hidden_size', 'num_attention_heads', 'seq_length', 'max_position_embeddings'],
        }
        for arg in required_base_args:
            if getattr(self.args, arg, None) is None:
                raise ValueError(f"{arg} is required")

        if self.args.model_type in model_specific_args:
            for arg in model_specific_args[self.args.model_type]:
                if getattr(self.args, arg, None) is None:
                    raise ValueError(f"{arg} is required for model type {self.args.model_type}")

    def _save_args_to_json(self):
        args_dict = vars(self.args)
        with open('config.json', 'w') as json_file:
            json.dump(args_dict, json_file, indent=4)

    def generate_parallel_group(self) -> Dict[str, List[int]]:
        """
        local_size = 8, world_size = 16, pp_size = 2, tp_size = 4
        manager.get_dp_groups() -> : [[0, 4], [1, 5], [2, 6], [3, 7], [8, 12], [9, 13], [10, 14], [11, 15]]
        manager.get_pp_groups() -> : [[0, 8], [1, 9], [2, 10], [3, 11], [4, 12], [5, 13], [6, 14], [7, 15]]
        manager.get_tp_groups() -> : [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
        manager.get_mp_groups() -> : [[0, 1, 2, 3, 8, 9, 10, 11], [4, 5, 6, 7, 12, 13, 14, 15]]
        """
        return self.parallel_group_manager.get_all_groups()

    def allocate_model_layers(self):
        return self.model_layer_allocator.get_layer_assignments(self.all_groups)
    
    # TODO: 写几个基本类型的model（标明一般结构和层数？）
    def _get_standard_abstract_model(self):
        pass

    # TODO: 生成每个rank的独立model（表征结构）
    def _generate_rank_individual_abstract_model(self):
        pass

    # TODO: 生成每个rank的独立model（真实结构）
    def _generate_rank_individual_realistic_model(self):
        pass

    # TODO: 基于realistic_model生成每个rank的独立graphs
    def _generate_rank_individual_op_graph(self):
        pass

    # TODO: 在rank_individual_op_graph的基础上进行op修改，添加同步/异步 op和 comm. op
    def _generate_rank_interconnected_op_graph(self):
        pass

    def _get_rank_individual_op_graph(self):
        pass

    def _get_rank_interconnected_op_graph(self):
        pass