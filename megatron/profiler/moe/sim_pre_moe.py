from megatron.profiler.moe.sim_routing import sim_routing
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.profiler.moe.sim_dispatching import sim_dispatching


def set_pre_distribution_moe(config: TransformerConfig):

    config.pre_fixed_routing_results = sim_routing(config=config, hidden_states_shape=config.routing_hidden_states_shape)
    print(f"[DEBUG] pre_fixed_routing_results: {config.pre_fixed_routing_results}")

    # in scaling mode, we need to further simulate the communication process (merge a series of values)
    # TODO-YC: to check below func, we can run it in the real runing mode, and compare the results with the pre results.
    if config.is_scaling_mode:
        dispatching_results = sim_dispatching(config)
        print(f"[DEBUG] dispatching_results: {dispatching_results}")

        config.per_rank_dispatching_results = dispatching_results['per_rank_results']
        config.num_global_tokens_per_expert = dispatching_results['num_global_tokens_per_expert']