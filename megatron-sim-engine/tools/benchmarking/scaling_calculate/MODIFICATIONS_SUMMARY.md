# Modifications Summary for Performance Testing Scripts

## Overview
The `run_performance_tests.sh` script has been successfully modified to work with the actual simulation file `simu_main.py` instead of the generic `main.py`. The script now dynamically configures simulation parameters based on setting directory names and captures both load and execution timing information.

## Key Modifications Made

### 1. Target File Change
- **Before**: Executed `main.py` in each setting directory
- **After**: Executes `/research/d1/gds/ytyang/yichengfeng/megatron-sim-engine/simu_main.py` with dynamic parameter modification

### 2. Dynamic Parameter Configuration
The script now automatically modifies the following parameters in `simu_main.py` for each setting:

#### Path Parameters
- `stages_scheduling_filepath`: Set to `"megatron_operation_log/8192_sim/{setting_name}/schedule"`
- `torchgraph_filepath`: Set to `"megatron_operation_log/8192_sim/{setting_name}/database_profile"`

#### Parallelization Parameters
- `curr_world_size`: Always set to 8192
- `curr_pp_size`: Extracted from setting name (e.g., 16, 32, 64)
- `curr_tp_size`: Extracted from setting name (e.g., 2, 4, 8)
- `curr_exp_size`: Always set to 1 (constant for dense models)

#### Visualization Parameter
- `simulator_engine.visualize_timelines(wrank_id_start_end=[0, X])`:
  - X = `min(8 * curr_tp_size, curr_world_size)`
  - Limits display to first 8 PP stages

### 3. Setting Name Parsing
Implemented robust parsing for setting directory names with format:
```
pp{X}_tp{Y}_ep1_expnNone_dp{Z}_nl96_hs20480_sl2048
```

**Parsing Logic**:
- PP size: Extract number after "pp"
- TP size: Extract number after "tp"
- DP size: Extract number after "dp"
- Validation: Ensures PP × TP × DP = 8192

### 4. Enhanced Output Capture
The script now captures and records both timing values from the output line:
```
print(f"world_size: {world_size}, sim load time:{load_time}, sim execution time:{execution_time}")
```

**Output Format**:
```
setting_name: load_time=X.XX execution_time=Y.YY
```

### 5. Backup and Restore Mechanism
- Creates backup of original `simu_main.py` before modifications
- Restores original file after all tests complete
- Ensures no permanent changes to the simulation script

### 6. Updated Metrics Calculation
Modified `calculate_metrics.sh` to handle the new timing format:
- Parses both load time and execution time
- Uses execution time for throughput calculations
- Displays both timing values in output table
- Updated CSV format to include both timing columns

## Validation Results

### Setting Name Parsing Test
All 6 target configurations parsed successfully:
- ✅ pp16_tp8_ep1_expnNone_dp64_nl96_hs20480_sl2048 → PP:16, TP:8, DP:64
- ✅ pp32_tp8_ep1_expnNone_dp32_nl96_hs20480_sl2048 → PP:32, TP:8, DP:32
- ✅ pp32_tp4_ep1_expnNone_dp64_nl96_hs20480_sl2048 → PP:32, TP:4, DP:64
- ✅ pp64_tp8_ep1_expnNone_dp16_nl96_hs20480_sl2048 → PP:64, TP:8, DP:16
- ✅ pp64_tp4_ep1_expnNone_dp32_nl96_hs20480_sl2048 → PP:64, TP:4, DP:32
- ✅ pp64_tp2_ep1_expnNone_dp64_nl96_hs20480_sl2048 → PP:64, TP:2, DP:64

### Metrics Calculation Test
Sample test with synthetic data shows correct processing:
- ✅ Timing information parsing works correctly
- ✅ Throughput calculations are accurate
- ✅ Cost analysis produces reasonable results
- ✅ CSV and detailed report generation functions properly

## Files Modified/Created

### Modified Files
1. `run_performance_tests.sh` - Complete rewrite for simu_main.py integration
2. `calculate_metrics.sh` - Updated to handle new timing format
3. `test_scripts.sh` - Updated test data format
4. `README.md` - Updated documentation

### New Files Created
1. `validate_parsing.sh` - Setting name parsing validation script
2. `MODIFICATIONS_SUMMARY.md` - This summary document

## Usage Instructions

### Running Performance Tests
```bash
cd /research/d1/gds/ytyang/yichengfeng/megatron-sim-engine/scaling_calculate/
./run_performance_tests.sh
```

### Calculating Metrics
```bash
./calculate_metrics.sh
```

### Validation
```bash
./validate_parsing.sh  # Test setting name parsing
./test_scripts.sh      # Test full workflow with sample data
```

## Expected Output Format

### Time Logs
```
pp16_tp8_ep1_expnNone_dp64_nl96_hs20480_sl2048: load_time=12.34 execution_time=45.67
```

### Metrics Table
```
Setting                                    Load (s)   Exec (s)      Throughput         Days      Cost (USD)
---------------------------------------- ---------- ---------- --------------- ------------ ---------------
pp16_tp8_ep1_expnNone_dp64_nl96_hs20480_sl2048  12.34      45.67          275518         21.0        $4047001
```

## Safety Features
- ✅ Backup and restore of original simu_main.py
- ✅ Timeout protection (30 minutes per test)
- ✅ Comprehensive error handling
- ✅ Detailed logging for debugging
- ✅ Validation of setting directory structure
- ✅ Graceful handling of parsing failures

The modifications are now complete and ready for production use with the actual 8192 setting configurations.