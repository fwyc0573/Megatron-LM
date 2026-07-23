from src.core.static_graphs.static_graphs_module import StaticGraphsModule
import argparse



def _parse_args():
    parser = argparse.ArgumentParser(description='Static Graphs Parser')
    # parser.add_argument('--model-type', type=str, required=True,
    #                     choices=['gpt', 'llama', 'bert', 'CV', 'others'],
    #                     help='Type of the model to partition.')
    # parser.add_argument('--tensor-model-parallel-size', type=int, default=1,
    #                     help='Size of the tensor model parallel.')
    # parser.add_argument('--pipeline-model-parallel-size', type=int, default=1,
    #                     help='Size of the pipeline model parallel.')
    # parser.add_argument('--micro-batch-size', type=int, required=True,
    #                     help='Size of the micro batch.')
    # parser.add_argument('--global-batch-size', type=int, required=True,
    #                     help='Size of the global batch.')

    # Arguments required for certain model types
    # parser.add_argument('--num-layers', type=int, default=None,
    #                     help='Number of layers in the model.')
    # parser.add_argument('--hidden-size', type=int, default=None,
    #                     help='Size of the hidden layers.')
    # parser.add_argument('--num-attention-heads', type=int, default=None,
    #                     help='Number of attention heads.')
    # parser.add_argument('--seq-length', type=int, default=None,
    #                     help='Sequence length.')
    # parser.add_argument('--max-position-embeddings', type=int, default=None,
    #                     help='Max position embeddings.')

    return parser.parse_args()



if __name__ == "__main__":
    args = _parse_args()
    static_graphs_module = StaticGraphsModule(args, json_save=True)
    layer_assignments = static_graphs_module.allocate_model_layers()
    print(layer_assignments)
