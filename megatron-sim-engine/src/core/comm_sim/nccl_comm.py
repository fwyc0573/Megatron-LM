import numpy as np
import logging
from typing import Dict, List, Tuple

INTER_COMM = "inter_comm"
INTRA_COMM = "intra_comm"
MIXED_COMM = "mixed_comm"

# 都在对应的group中进行通信操作，有明显的 inter-comm 和 intra-comm的区别
# TP/EP ALLREDUCE: intra-commm, 同一host内的allreduce（group_size<8）
# DP ALLREDUCE: inter-comm, 跨NIC的allreduce
# PP SEND_RECV: inter-comm, 跨NIC的send/recv

# Configuration for cross-machine communication adaptation
CROSS_MACHINE_CORRECTION_FACTOR = 0.95  # Default correction factor for cross-machine scenarios
GPUS_PER_MACHINE = 8  # Default GPUs per machine configuration

# Setup logging for communication module
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] [COMM] %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# 64卡 单位us
# NCCL_COMM_DB = {
#     INTER_COMM: {
#         'send_recv': {
#             2: {
#                 1024: 102,
#                 100663296: 273402.32,
#             },
#         }, 
#         'allreduce_ring': {
#         },
#         'reduce_scatter': {
#         },
#         'allgather': {
#         },
#     },
#     INTRA_COMM:{
#         'send_recv': {
#             2: {
#                 2048: 101.07,
#                 10485760: 190.21
#             },
#         }, 
#         'allreduce_ring': {
#             2: {
#                 1024: 11.03,
#                 65536: 25.89,
#                 262144: 50.21,
#                 524288: 52.29,
#                 1048576: 66.56,
#                 2097152: 75.95,
#                 4194304: 88.87,
#                 8388608: 118.7,
#                 16777216: 159.4,
#                 33554432: 261.8,
#                 67108864: 418.4,
#                 134217728: 742.2,
#                 268435456: 2390.6,
#                 6509404160: 61920.5
#             },
#             4: {
#                 1024: 15.98,
#                 65536: 26.26,
#                 262144: 43.42,
#                 524288: 58.14,
#                 1048576: 75.07,
#                 2097152: 83.36,
#                 4194304: 102.9,
#                 8388608: 148.2,
#                 16777216: 222.8,
#                 33554432: 310.4,
#                 67108864: 555.1,
#                 134217728: 998.9,
#                 268435456: 1939.3
#             },
#             # 8: {
#             #     1024: 34.07,
#             #     65536: 37.07,
#             #     262144: 47.60,
#             #     524288: 50.58,
#             #     1048576: 78.63,
#             #     2097152: 100.3,
#             #     4194304: 122.9,
#             #     8388608: 181.5,
#             #     16777216: 286.5,
#             #     33554432: 413.3,
#             #     67108864: 657.3,
#             #     134217728: 1308.7,
#             #     268435456: 2393.0
#             # },
#             8: {
#                 1024: 34.07,
#                 65536: 37.07,
#                 262144: 47.60,
#                 524288: 50.58,
#                 1048576: 78.63,
#                 2097152: 100.3,
#                 4194304: 122.9,
#                 8388608: 181.5,
#                 16777216: 286.5,
#                 33554432: 413.3,
#                 67108864: 7890.2,
#                 100663296: 13080.7,
#                 134217728: 16308.1,
#                 268435456: 32308.3
#             },
#         },
#         'reduce_scatter': {
#         },
#         'allgather': {
#         },
#     },
#     MIXED_COMM:{
#         # 96卡100663296: 273402.32*1.3,
#         # 64卡100663296: 273402.32*1.77
#         'send_recv': {
#             2: {
#                 2048: 101.07,
#                 10485760: 190.21,
#                 100663296: 273402.32*1.82/8,
#             },
#         }, 
#         'allreduce_ring': {
#             2: {
#                 1024: 11.03,
#                 65536: 25.89,
#                 262144: 50.21,
#                 524288: 52.29,
#                 1048576: 66.56,
#                 2097152: 75.95,
#                 4194304: 88.87,
#                 8388608: 118.7,
#                 16777216: 159.4,
#                 33554432: 261.8,
#                 67108864: 418.4,
#                 134217728: 742.2,
#                 268435456: 2390.6,
#                 6509404160: 61920.5
#             },
#             # 4: {
#             #     1024: 15.98,
#             #     65536: 26.26,
#             #     262144: 43.42,
#             #     524288: 58.14,
#             #     1048576: 75.07,
#             #     2097152: 83.36,
#             #     4194304: 102.9,
#             #     8388608: 148.2,
#             #     16777216: 222.8,
#             #     33554432: 310.4,
#             #     67108864: 555.1,
#             #     134217728: 998.9,
#             #     268435456: 1939.3
#             # },
#             4: {
#                 1024: 34.07,
#                 65536: 37.07,
#                 262144: 47.60,
#                 524288: 50.58,
#                 1048576: 78.63,
#                 2097152: 100.3,
#                 4194304: 122.9,
#                 8388608: 181.5,
#                 16777216: 286.5,
#                 33554432: 413.3,
#                 67108864: 7890.2,
#                 100663296: 13080.7*1.7,
#                 134217728: 16308.1*1.7,
#                 268435456: 32308.3*1.7
#             },
#             # 8: {
#             #     1024: 34.07,
#             #     65536: 37.07,
#             #     262144: 47.60,
#             #     524288: 50.58,
#             #     1048576: 78.63,
#             #     2097152: 100.3,
#             #     4194304: 122.9,
#             #     8388608: 181.5,
#             #     16777216: 286.5,
#             #     33554432: 413.3,
#             #     67108864: 657.3,
#             #     134217728: 1308.7,
#             #     268435456: 2393.0
#             # },
#             8: {
#                 1024: 34.07,
#                 65536: 37.07,
#                 262144: 47.60,
#                 524288: 50.58,
#                 1048576: 78.63,
#                 2097152: 100.3,
#                 4194304: 122.9,
#                 8388608: 181.5,
#                 16777216: 286.5,
#                 33554432: 413.3,
#                 67108864: 7890.2,
#                 100663296: 13080.7*1.7,
#                 134217728: 16308.1*1.7,
#                 268435456: 32308.3*1.7
#             },
#             16: {
#             },
#             32: {
#             }
#         },
#         'allreduce_tree': {
#         },
#         'reduce_scatter': {
#         },
#         'allgather': {
#         },
#     }

# }



# 96卡 
NCCL_COMM_DB = {
    INTER_COMM: {
        'send_recv': {
            2: {
                1024: 102,
                100663296: 273402.32,
            },
        }, 
        'allreduce_ring': {
        },
        'reduce_scatter': {
        },
        'allgather': {
        },
    },
    INTRA_COMM:{
        'send_recv': {
            2: {
                2048: 101.07,
                10485760: 190.21
            },
        }, 
        'allreduce_ring': {
            2: {
                1024: 11.03,
                65536: 25.89,
                262144: 50.21,
                524288: 52.29,
                1048576: 66.56,
                2097152: 75.95,
                4194304: 88.87,
                8388608: 118.7,
                16777216: 159.4,
                33554432: 261.8,
                67108864: 418.4,
                134217728: 742.2,
                268435456: 2390.6,
                6509404160: 61920.5
            },
            4: {
                1024: 15.98,
                65536: 26.26,
                262144: 43.42,
                524288: 58.14,
                1048576: 75.07,
                2097152: 83.36,
                4194304: 102.9,
                8388608: 148.2,
                16777216: 222.8,
                33554432: 310.4,
                67108864: 555.1,
                134217728: 998.9,
                268435456: 1939.3
            },
            # 8: {
            #     1024: 34.07,
            #     65536: 37.07,
            #     262144: 47.60,
            #     524288: 50.58,
            #     1048576: 78.63,
            #     2097152: 100.3,
            #     4194304: 122.9,
            #     8388608: 181.5,
            #     16777216: 286.5,
            #     33554432: 413.3,
            #     67108864: 657.3,
            #     134217728: 1308.7,
            #     268435456: 2393.0
            # },
            8: {
                1024: 34.07,
                65536: 37.07,
                262144: 47.60,
                524288: 50.58,
                1048576: 78.63,
                2097152: 100.3,
                4194304: 122.9,
                8388608: 181.5,
                16777216: 286.5,
                33554432: 413.3,
                67108864: 7890.2,
                100663296: 13080.7,
                134217728: 16308.1,
                268435456: 32308.3
            },
        },
        'reduce_scatter': {
        },
        'allgather': {
        },
    },
    MIXED_COMM:{
        # 96卡100663296: 273402.32*1.3,
        # 64卡100663296: 273402.32*1.77
        'send_recv': {
            2: {
                2048: 101.07,
                10485760: 190.21,
                100663296: 273402.32*1.3/12,
            },
        }, 
        'allreduce_ring': {
            2: {
                1024: 11.03,
                65536: 25.89,
                262144: 50.21,
                524288: 52.29,
                1048576: 66.56,
                2097152: 75.95,
                4194304: 88.87,
                8388608: 118.7,
                16777216: 159.4,
                33554432: 261.8,
                67108864: 418.4,
                134217728: 742.2,
                268435456: 2390.6,
                6509404160: 61920.5
            },
            # 4: {
            #     1024: 15.98,
            #     65536: 26.26,
            #     262144: 43.42,
            #     524288: 58.14,
            #     1048576: 75.07,
            #     2097152: 83.36,
            #     4194304: 102.9,
            #     8388608: 148.2,
            #     16777216: 222.8,
            #     33554432: 310.4,
            #     67108864: 555.1,
            #     134217728: 998.9,
            #     268435456: 1939.3
            # },
            4: {
                1024: 34.07,
                65536: 37.07,
                262144: 47.60,
                524288: 50.58,
                1048576: 78.63,
                2097152: 100.3,
                4194304: 122.9,
                8388608: 181.5,
                16777216: 286.5,
                33554432: 413.3,
                67108864: 7890.2,
                100663296: 13080.7*1.7,
                134217728: 16308.1*1.7,
                268435456: 32308.3*1.7
            },
            # 8: {
            #     1024: 34.07,
            #     65536: 37.07,
            #     262144: 47.60,
            #     524288: 50.58,
            #     1048576: 78.63,
            #     2097152: 100.3,
            #     4194304: 122.9,
            #     8388608: 181.5,
            #     16777216: 286.5,
            #     33554432: 413.3,
            #     67108864: 657.3,
            #     134217728: 1308.7,
            #     268435456: 2393.0
            # },
            8: {
                1024: 34.07,
                65536: 37.07,
                262144: 47.60,
                524288: 50.58,
                1048576: 78.63,
                2097152: 100.3,
                4194304: 122.9,
                8388608: 181.5,
                16777216: 286.5,
                33554432: 413.3,
                67108864: 7890.2,
                100663296: 13080.7*1.7,
                134217728: 16308.1*1.7,
                268435456: 32308.3*1.7
            },
            16: {
                1024: 45.2,
                65536: 48.5,
                262144: 58.3,
                524288: 62.1,
                1048576: 89.4,
                2097152: 115.6,
                4194304: 148.2,
                8388608: 210.8,
                16777216: 325.4,
                33554432: 468.9,
                67108864: 8945.3,
                100663296: 14832.8,
                134217728: 18509.7,
                268435456: 36618.4
            },
            32: {
                1024: 56.8,
                65536: 61.2,
                262144: 72.9,
                524288: 78.4,
                1048576: 108.7,
                2097152: 142.3,
                4194304: 186.5,
                8388608: 268.9,
                16777216: 412.8,
                33554432: 598.2,
                67108864: 11420.6,
                100663296: 18932.4,
                134217728: 23612.1,
                268435456: 46724.8
            }
        },
        'allreduce_tree': {
            16: {
                1024: 42.1,
                65536: 45.8,
                262144: 55.2,
                524288: 58.9,
                1048576: 84.7,
                2097152: 109.8,
                4194304: 140.6,
                8388608: 199.2,
                16777216: 308.1,
                33554432: 444.6,
                67108864: 8498.7,
                100663296: 14091.2,
                134217728: 17584.2,
                268435456: 34787.5
            },
            32: {
                1024: 52.3,
                65536: 57.1,
                262144: 68.4,
                524288: 73.6,
                1048576: 102.1,
                2097152: 133.4,
                4194304: 175.2,
                8388608: 252.8,
                16777216: 388.4,
                33554432: 562.9,
                67108864: 10734.5,
                100663296: 17789.3,
                134217728: 22186.4,
                268435456: 43872.8
            }
        },
        'reduce_scatter': {
        },
        'allgather': {
        },
    }

}

# Cache for computed communication times to avoid repeated calculations
_comm_time_cache = {}

def _get_machine_distribution(comm_group: List[int]) -> Tuple[int, bool]:
    """
    Analyze the machine distribution of communication group.

    Args:
        comm_group: List of GPU ranks in the communication group

    Returns:
        Tuple of (machines_count, is_cross_machine)
    """
    if len(comm_group) <= GPUS_PER_MACHINE:
        # Check if all GPUs are on the same machine
        machine_ids = set(rank // GPUS_PER_MACHINE for rank in comm_group)
        is_cross_machine = len(machine_ids) > 1
        return len(machine_ids), is_cross_machine
    else:
        # For larger groups, assume cross-machine communication
        machines_count = (len(comm_group) + GPUS_PER_MACHINE - 1) // GPUS_PER_MACHINE
        return machines_count, True

def _find_best_matching_config(gpu_num: int, across_comm_type: str, comm_func: str) -> Tuple[int, float]:
    """
    Find the best matching configuration for the given GPU number.

    Args:
        gpu_num: Number of GPUs in the communication group
        across_comm_type: Communication type (INTER_COMM, INTRA_COMM, MIXED_COMM)
        comm_func: Communication function name

    Returns:
        Tuple of (best_matching_gpu_num, correction_factor)
    """
    available_configs = list(NCCL_COMM_DB[across_comm_type][comm_func].keys())

    if gpu_num in available_configs:
        return gpu_num, 1.0

    # Find the closest configuration
    closest_config = min(available_configs, key=lambda x: abs(x - gpu_num))

    # Calculate correction factor based on configuration difference
    if gpu_num > closest_config:
        # For larger groups, communication time typically increases
        correction_factor = 1.0 + (gpu_num - closest_config) * 0.1
    else:
        # For smaller groups, communication time typically decreases
        correction_factor = max(0.5, 1.0 - (closest_config - gpu_num) * 0.05)

    logger.debug(f"GPU config adaptation: {gpu_num} -> {closest_config}, factor: {correction_factor:.3f}")
    return closest_config, correction_factor

def _enhanced_interpolation(data_size: int, size_time_dict: Dict[int, float]) -> float:
    """
    Enhanced interpolation with better handling of edge cases and non-linear patterns.

    Args:
        data_size: Target data size in bytes
        size_time_dict: Dictionary mapping data sizes to communication times

    Returns:
        Interpolated communication time
    """
    keys = np.array(list(size_time_dict.keys()))
    values = np.array(list(size_time_dict.values()))

    if data_size in size_time_dict:
        return size_time_dict[data_size]

    # Handle edge cases
    if data_size < keys.min():
        # For very small data sizes, use minimum time with slight adjustment
        min_time = values[keys.argmin()]
        scale_factor = max(0.1, data_size / keys.min())
        estimated_value = min_time * scale_factor
        logger.debug(f"Small data size extrapolation: {data_size} bytes -> {estimated_value:.2f} us")
        return estimated_value
    elif data_size > keys.max():
        # For very large data sizes, use linear extrapolation from the last two points
        if len(keys) >= 2:
            sorted_indices = np.argsort(keys)
            last_two_keys = keys[sorted_indices[-2:]]
            last_two_values = values[sorted_indices[-2:]]

            # Linear extrapolation
            slope = (last_two_values[1] - last_two_values[0]) / (last_two_keys[1] - last_two_keys[0])
            estimated_value = last_two_values[1] + slope * (data_size - last_two_keys[1])
            logger.debug(f"Large data size extrapolation: {data_size} bytes -> {estimated_value:.2f} us")
            return max(estimated_value, last_two_values[1])  # Ensure non-decreasing
        else:
            return values[keys.argmax()]
    else:
        # Interpolation between existing points
        lower_idx = np.max(np.where(keys <= data_size))
        upper_idx = np.min(np.where(keys >= data_size))

        if lower_idx == upper_idx:
            return values[lower_idx]

        lower_key = keys[lower_idx]
        upper_key = keys[upper_idx]
        lower_value = values[lower_idx]
        upper_value = values[upper_idx]

        # Use logarithmic interpolation for better accuracy with communication patterns
        if lower_key > 0 and upper_key > 0 and lower_value > 0 and upper_value > 0:
            log_data_size = np.log(data_size)
            log_lower_key = np.log(lower_key)
            log_upper_key = np.log(upper_key)
            log_lower_value = np.log(lower_value)
            log_upper_value = np.log(upper_value)

            log_estimated = log_lower_value + (log_upper_value - log_lower_value) * \
                           (log_data_size - log_lower_key) / (log_upper_key - log_lower_key)
            estimated_value = np.exp(log_estimated)
        else:
            # Fall back to linear interpolation
            estimated_value = lower_value + (upper_value - lower_value) * \
                             (data_size - lower_key) / (upper_key - lower_key)

        logger.debug(f"Interpolation: {data_size} bytes between {lower_key}-{upper_key} -> {estimated_value:.2f} us")
        return estimated_value


def get_across_comm_type_and_comm_func(comm_group:list, comm_func:str):
    across_comm_type = None
    across_comm_type = MIXED_COMM

    # print(f"comm_group:{comm_group}, comm_func:{comm_func}")

    if comm_func == "recv_forward" or comm_func == "send_forward" or comm_func == "recv_backward" or comm_func == "send_backward" or comm_func == "send_recv":
        comm_func = "send_recv"

    elif "allreduce" in comm_func:
        # TODO: fix tree or ring
        if len(comm_group) < 1024:
            comm_func = "allreduce_ring"
        else:
            comm_func = "allreduce_tree"
    
    elif "allgather" in comm_func:
        # Treat 'allgather' as 'allreduce_ring' for now as a temporary substitute
        if len(comm_group) < 1024:
            comm_func = "allreduce_ring"
        else:
            comm_func = "allreduce_tree"
    
    elif "all_to_all" in comm_func:
        # Treat 'all_to_all' as 'allreduce_ring' for now as a temporary substitute
        if len(comm_group) < 1024:
            comm_func = "allreduce_ring"
        else:
            comm_func = "allreduce_tree"

    elif "reduce_scatter" in comm_func:
        # The checked-in analytical database contains ring/tree allreduce
        # measurements but no standalone reduce-scatter table.  A ring
        # reduce-scatter transfers exactly half of the allreduce volume for
        # the same payload, so retain the profile lookup and apply that
        # operation-specific volume factor in get_comm_op_exc_time().
        if len(comm_group) < 1024:
            comm_func = "reduce_scatter_ring"
        else:
            comm_func = "reduce_scatter_tree"

    # TODO：暂时支持的几种comm
    if comm_func not in [
        "allreduce_tree",
        "allreduce_ring",
        "reduce_scatter_tree",
        "reduce_scatter_ring",
        "send_recv",
    ]:
        raise ValueError(f"comm_func:{comm_func} not supported")

    if ("tp" in comm_func or "ep") in comm_func and len(comm_group) <= 8:
        across_comm_type = INTRA_COMM

    return across_comm_type, comm_func


def get_comm_op_exc_time(comm_group: List[int], data_size: int, comm_func: str) -> float:
    """
    Enhanced communication operation execution time calculation with improved interpolation
    and cross-machine configuration adaptation.

    Args:
        comm_group: List of GPU ranks in the communication group
        data_size: Size of data to be communicated in bytes
        comm_func: Communication function name

    Returns:
        Estimated communication time in milliseconds
    """
    assert isinstance(comm_group, list), "comm_group should be a list"

    gpu_num = len(comm_group)

    # No communication needed for single GPU
    if gpu_num == 1:
        return 0.01

    # Create cache key for this specific communication scenario
    cache_key = (tuple(sorted(comm_group)), data_size, comm_func)
    if cache_key in _comm_time_cache:
        logger.debug(f"Cache hit for comm scenario: {len(comm_group)} GPUs, {data_size} bytes, {comm_func}")
        return _comm_time_cache[cache_key]

    # Determine communication type and function
    across_comm_type, processed_comm_func = get_across_comm_type_and_comm_func(comm_group, comm_func)

    # Special handling for P2P communication
    if processed_comm_func == "send_recv":
        gpu_num = 2

    # Reduce-scatter uses the corresponding allreduce profile with an
    # explicit half-volume factor; the database intentionally stores only
    # measured allreduce ring/tree samples.
    profile_comm_func = processed_comm_func
    reduce_scatter_factor = 1.0
    if processed_comm_func == "reduce_scatter_ring":
        profile_comm_func = "allreduce_ring"
        reduce_scatter_factor = 0.5
    elif processed_comm_func == "reduce_scatter_tree":
        profile_comm_func = "allreduce_tree"
        reduce_scatter_factor = 0.5

    # Validate communication function exists
    if profile_comm_func not in NCCL_COMM_DB[across_comm_type]:
        raise ValueError(f"{profile_comm_func} not in NCCL_COMM_DB for {across_comm_type}")

    # Find best matching GPU configuration and get correction factor
    original_gpu_num = gpu_num
    gpu_num, config_correction_factor = _find_best_matching_config(
        gpu_num, across_comm_type, profile_comm_func
    )

    # Get the timing data for this configuration
    size_time_dict = NCCL_COMM_DB[across_comm_type][profile_comm_func][gpu_num]

    # Use enhanced interpolation to get the estimated time
    estimated_value = _enhanced_interpolation(data_size, size_time_dict)

    # Apply configuration correction factor
    estimated_value *= config_correction_factor * reduce_scatter_factor

    # Apply cross-machine correction if needed
    machines_count, is_cross_machine = _get_machine_distribution(comm_group)
    if is_cross_machine and original_gpu_num <= 16:  # Apply correction for smaller cross-machine groups
        cross_machine_factor = CROSS_MACHINE_CORRECTION_FACTOR
        estimated_value *= cross_machine_factor
        logger.debug(f"Cross-machine correction applied: factor={cross_machine_factor:.3f}, "
                    f"machines={machines_count}, gpus={original_gpu_num}")

    # Cache the result for future use
    result_ms = round(estimated_value / 1000, 2)  # Convert us to ms
    _comm_time_cache[cache_key] = result_ms

    # Log detailed information for debugging
    logger.info(f"Comm time calculation: {original_gpu_num} GPUs, {data_size} bytes, {comm_func} -> "
               f"{result_ms:.2f} ms (config: {gpu_num}, cross-machine: {is_cross_machine})")

    return result_ms


def set_cross_machine_correction_factor(factor: float) -> None:
    """
    Set the correction factor for cross-machine communication scenarios.

    Args:
        factor: Correction factor (typically between 0.8 and 1.2)
    """
    global CROSS_MACHINE_CORRECTION_FACTOR
    CROSS_MACHINE_CORRECTION_FACTOR = factor
    logger.info(f"Cross-machine correction factor updated to: {factor}")


def set_gpus_per_machine(gpus: int) -> None:
    """
    Set the number of GPUs per machine for cross-machine detection.

    Args:
        gpus: Number of GPUs per machine (typically 8)
    """
    global GPUS_PER_MACHINE
    GPUS_PER_MACHINE = gpus
    logger.info(f"GPUs per machine updated to: {gpus}")


def clear_comm_cache() -> None:
    """Clear the communication time cache."""
    global _comm_time_cache
    cache_size = len(_comm_time_cache)
    _comm_time_cache.clear()
    logger.info(f"Communication cache cleared ({cache_size} entries removed)")


def get_cache_stats() -> Dict[str, int]:
    """
    Get statistics about the communication cache.

    Returns:
        Dictionary with cache statistics
    """
    return {
        'cache_size': len(_comm_time_cache),
        'available_configs': sum(len(NCCL_COMM_DB[comm_type][func])
                               for comm_type in NCCL_COMM_DB
                               for func in NCCL_COMM_DB[comm_type])
    }


def validate_comm_database() -> bool:
    """
    Validate the communication database for consistency.

    Returns:
        True if database is valid, False otherwise
    """
    try:
        for comm_type in [INTER_COMM, INTRA_COMM, MIXED_COMM]:
            if comm_type not in NCCL_COMM_DB:
                logger.error(f"Missing communication type: {comm_type}")
                return False

            for func_name in ['send_recv', 'allreduce_ring', 'allreduce_tree']:
                if func_name not in NCCL_COMM_DB[comm_type]:
                    logger.warning(f"Missing function {func_name} in {comm_type}")
                    continue

                for gpu_count, timings in NCCL_COMM_DB[comm_type][func_name].items():
                    if not isinstance(timings, dict) or not timings:
                        logger.warning(f"Invalid timings for {comm_type}/{func_name}/{gpu_count}")
                        continue

                    # Check for reasonable timing values
                    for size, time_us in timings.items():
                        if time_us <= 0:
                            logger.warning(f"Invalid timing: {comm_type}/{func_name}/{gpu_count}/{size} = {time_us}")

        logger.info("Communication database validation completed")
        return True
    except Exception as e:
        logger.error(f"Database validation failed: {e}")
        return False


