#!/usr/bin/env python3
"""
Comprehensive Validation Script for MoE Scheduling Plan Consistency

This script validates that generated scheduling plans match the ground truth trace data
by comparing line counts and analyzing stage-to-rank mappings.

Author: Auto-generated for MoE validation
Date: 2025-08-19
"""

import os
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import logging

# Add project root to path for package imports.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.core.static_graphs.rank_manager import RankManager
except ImportError:
    print("Warning: Could not import RankManager from src.core.static_graphs")
    RankManager = None

# Configure logging
LOG_DIR = PROJECT_ROOT / "log" / "mg_scheduling"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'scheduling_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SchedulingConsistencyValidator:
    """Validator for scheduling plan consistency with ground truth traces"""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        candidates = [
            self.base_dir / "simulation_inputs" / "megatron_operation_log" / "new_moe",
            Path(
                "/research/d1/gds/ytyang/yichengfeng/megatron-sim-engine/"
                "simulation_inputs/megatron_operation_log/new_moe"
            ),
            self.base_dir.parent / "simulation_inputs" / "megatron_operation_log" / "new_moe",
        ]
        self.target_base = candidates[0]
        for candidate in candidates:
            if candidate.exists():
                self.target_base = candidate
                break

        self.validation_results = {}
        
    def parse_config_params(self, config_name: str) -> Dict[str, int]:
        """Parse configuration parameters from config name"""
        pattern = r'pp(\d+)_tp(\d+)_exp(\d+)_expn(\d+)_dp(\d+)_nl(\d+)_hs(\d+)_sl(\d+)'
        match = re.match(pattern, config_name)
        
        if not match:
            return {}
        
        return {
            'pp': int(match.group(1)),
            'tp': int(match.group(2)),
            'exp': int(match.group(3)),
            'expn': int(match.group(4)),
            'dp': int(match.group(5)),
            'nl': int(match.group(6)),
            'hs': int(match.group(7)),
            'sl': int(match.group(8))
        }
    
    def get_rank_to_stage_mapping(self, pp: int, tp: int, dp: int) -> Dict[int, int]:
        """Calculate rank to stage mapping using PP configuration"""
        world_size = pp * tp * dp
        rank_to_stage = {}
        
        # Simple mapping: ranks are distributed across PP stages
        ranks_per_stage = world_size // pp
        
        for rank in range(world_size):
            stage = rank // ranks_per_stage
            if stage >= pp:  # Handle edge case
                stage = pp - 1
            rank_to_stage[rank] = stage
            
        return rank_to_stage
    
    def count_file_lines(self, file_path: Path) -> int:
        """Count lines in a file"""
        try:
            with open(file_path, 'r') as f:
                return sum(1 for _ in f)
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return -1
    
    def analyze_trace_files(self, config_dir: Path, config_name: str) -> Dict:
        """Analyze ground truth trace files"""
        global_ranks_dir = config_dir / "global_ranks_profile"
        
        if not global_ranks_dir.exists():
            return {"error": "global_ranks_profile directory not found"}
        
        trace_files = list(global_ranks_dir.glob("*.txt"))
        
        # Extract rank information from filenames
        rank_data = {}
        for file_path in trace_files:
            filename = file_path.name
            
            # Try to extract rank number from filename
            rank_match = re.search(r'rank(\d+)', filename)
            if rank_match:
                rank = int(rank_match.group(1))
                line_count = self.count_file_lines(file_path)
                rank_data[rank] = {
                    'file': filename,
                    'lines': line_count
                }
        
        return {
            'total_files': len(trace_files),
            'rank_data': rank_data,
            'ranks_found': sorted(rank_data.keys())
        }
    
    def analyze_schedule_files(self, config_dir: Path, config_name: str) -> Dict:
        """Analyze generated scheduling plan files"""
        schedule_dir = config_dir / "schedule"
        
        if not schedule_dir.exists():
            return {"error": "schedule directory not found"}
        
        schedule_files = list(schedule_dir.glob("*.txt"))
        
        # Extract stage information from filenames
        stage_data = {}
        for file_path in schedule_files:
            filename = file_path.name
            
            # Extract stage number from filename
            stage_match = re.search(r'stage(\d+)', filename)
            if stage_match:
                stage = int(stage_match.group(1))
                line_count = self.count_file_lines(file_path)
                stage_data[stage] = {
                    'file': filename,
                    'lines': line_count
                }
        
        return {
            'total_files': len(schedule_files),
            'stage_data': stage_data,
            'stages_found': sorted(stage_data.keys())
        }
    
    def validate_line_count_consistency(self, config_name: str, params: Dict, 
                                      trace_analysis: Dict, schedule_analysis: Dict) -> Dict:
        """Validate line count consistency between traces and schedules"""
        validation = {
            'config': config_name,
            'params': params,
            'consistent': True,
            'mismatches': [],
            'missing_stages': [],
            'missing_ranks': [],
            'line_count_comparison': {}
        }
        
        if 'error' in trace_analysis or 'error' in schedule_analysis:
            validation['consistent'] = False
            validation['error'] = trace_analysis.get('error', '') + schedule_analysis.get('error', '')
            return validation
        
        # Get rank to stage mapping
        pp = params['pp']
        tp = params['tp']
        dp = params['dp']
        
        rank_to_stage = self.get_rank_to_stage_mapping(pp, tp, dp)
        
        # Group ranks by stage
        stage_to_ranks = defaultdict(list)
        for rank, stage in rank_to_stage.items():
            stage_to_ranks[stage].append(rank)
        
        # Compare line counts for each stage
        for stage in range(pp):
            if stage not in schedule_analysis['stage_data']:
                validation['missing_stages'].append(stage)
                validation['consistent'] = False
                continue
            
            schedule_lines = schedule_analysis['stage_data'][stage]['lines']
            
            # Get corresponding ranks for this stage
            stage_ranks = stage_to_ranks[stage]
            
            # Check if we have trace data for these ranks
            available_ranks = []
            trace_lines_list = []
            
            for rank in stage_ranks:
                if rank in trace_analysis['rank_data']:
                    available_ranks.append(rank)
                    trace_lines_list.append(trace_analysis['rank_data'][rank]['lines'])
                else:
                    validation['missing_ranks'].append(rank)
            
            if not available_ranks:
                validation['consistent'] = False
                continue
            
            # For consistency, all ranks in the same stage should have similar line counts
            # We'll use the first available rank as reference
            reference_trace_lines = trace_lines_list[0]
            
            validation['line_count_comparison'][stage] = {
                'schedule_lines': schedule_lines,
                'trace_lines': reference_trace_lines,
                'stage_ranks': stage_ranks,
                'available_ranks': available_ranks,
                'all_trace_lines': trace_lines_list
            }
            
            # Check for significant discrepancies (allow small differences due to warmup/cooldown)
            line_diff = abs(schedule_lines - reference_trace_lines)
            tolerance = max(5, reference_trace_lines * 0.05)  # 5 lines or 5% tolerance
            
            if line_diff > tolerance:
                validation['mismatches'].append({
                    'stage': stage,
                    'schedule_lines': schedule_lines,
                    'trace_lines': reference_trace_lines,
                    'difference': line_diff,
                    'tolerance': tolerance
                })
                validation['consistent'] = False
        
        return validation
    
    def validate_configuration(self, config_name: str) -> Dict:
        """Validate a single configuration"""
        logger.info(f"Validating configuration: {config_name}")
        
        config_dir = self.target_base / config_name
        if not config_dir.exists():
            return {"error": f"Configuration directory not found: {config_dir}"}
        
        # Parse configuration parameters
        params = self.parse_config_params(config_name)
        if not params:
            return {"error": f"Failed to parse configuration parameters from {config_name}"}
        
        # Analyze trace and schedule files
        trace_analysis = self.analyze_trace_files(config_dir, config_name)
        schedule_analysis = self.analyze_schedule_files(config_dir, config_name)
        
        # Validate consistency
        validation = self.validate_line_count_consistency(
            config_name, params, trace_analysis, schedule_analysis
        )
        
        # Add analysis data to validation result
        validation['trace_analysis'] = trace_analysis
        validation['schedule_analysis'] = schedule_analysis
        
        return validation
    
    def run_validation(self) -> Dict:
        """Run validation for all configurations"""
        logger.info("Starting comprehensive scheduling consistency validation")
        
        if not self.target_base.exists():
            logger.error(f"Target directory not found: {self.target_base}")
            return {"error": "Target directory not found"}
        
        # Find all configuration directories
        config_dirs = [d for d in self.target_base.iterdir() 
                      if d.is_dir() and d.name.startswith('pp')]
        
        logger.info(f"Found {len(config_dirs)} configurations to validate")
        
        results = {
            'total_configs': len(config_dirs),
            'consistent_configs': 0,
            'inconsistent_configs': 0,
            'failed_configs': 0,
            'config_results': {},
            'summary': {}
        }
        
        for config_dir in sorted(config_dirs):
            config_name = config_dir.name
            validation = self.validate_configuration(config_name)
            
            results['config_results'][config_name] = validation
            
            if 'error' in validation:
                results['failed_configs'] += 1
                logger.error(f"Validation failed for {config_name}: {validation['error']}")
            elif validation['consistent']:
                results['consistent_configs'] += 1
                logger.info(f"✅ {config_name}: Consistent")
            else:
                results['inconsistent_configs'] += 1
                logger.warning(f"⚠️ {config_name}: Inconsistent - {len(validation['mismatches'])} mismatches")
        
        # Generate summary
        results['summary'] = {
            'success_rate': results['consistent_configs'] / results['total_configs'] * 100,
            'total_mismatches': sum(len(v.get('mismatches', [])) for v in results['config_results'].values()),
            'configs_with_missing_stages': len([v for v in results['config_results'].values() 
                                              if v.get('missing_stages', [])]),
            'configs_with_missing_ranks': len([v for v in results['config_results'].values() 
                                             if v.get('missing_ranks', [])])
        }
        
        logger.info(f"Validation completed: {results['consistent_configs']}/{results['total_configs']} consistent")
        
        return results
    
    def generate_report(self, results: Dict) -> str:
        """Generate a detailed validation report"""
        report_lines = [
            "MoE Scheduling Plan Consistency Validation Report",
            "=" * 60,
            "",
            f"Total Configurations: {results['total_configs']}",
            f"Consistent: {results['consistent_configs']}",
            f"Inconsistent: {results['inconsistent_configs']}",
            f"Failed: {results['failed_configs']}",
            f"Success Rate: {results['summary']['success_rate']:.1f}%",
            "",
            "Summary Statistics:",
            f"  Total Mismatches: {results['summary']['total_mismatches']}",
            f"  Configs with Missing Stages: {results['summary']['configs_with_missing_stages']}",
            f"  Configs with Missing Ranks: {results['summary']['configs_with_missing_ranks']}",
            "",
            "Detailed Results:",
            "-" * 40
        ]
        
        for config_name, validation in results['config_results'].items():
            if 'error' in validation:
                report_lines.append(f"❌ {config_name}: ERROR - {validation['error']}")
                continue
            
            status = "✅" if validation['consistent'] else "⚠️"
            report_lines.append(f"{status} {config_name}:")
            
            if validation['params']:
                params = validation['params']
                report_lines.append(f"    PP={params['pp']}, TP={params['tp']}, DP={params['dp']}")
            
            if validation['mismatches']:
                report_lines.append(f"    Mismatches: {len(validation['mismatches'])}")
                for mismatch in validation['mismatches']:
                    report_lines.append(
                        f"      Stage {mismatch['stage']}: "
                        f"Schedule={mismatch['schedule_lines']}, "
                        f"Trace={mismatch['trace_lines']}, "
                        f"Diff={mismatch['difference']}"
                    )
            
            if validation['missing_stages']:
                report_lines.append(f"    Missing Stages: {validation['missing_stages']}")
            
            if validation['missing_ranks']:
                report_lines.append(f"    Missing Ranks: {validation['missing_ranks']}")
            
            report_lines.append("")
        
        return "\n".join(report_lines)


def main():
    """Main execution"""
    print("MoE Scheduling Plan Consistency Validator")
    print("=" * 50)
    
    validator = SchedulingConsistencyValidator()
    results = validator.run_validation()
    
    # Generate and save report
    report = validator.generate_report(results)
    
    report_file = LOG_DIR / "scheduling_consistency_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(report)
    print(f"\nDetailed report saved to: {report_file}")
    
    # Return appropriate exit code
    if results['inconsistent_configs'] > 0 or results['failed_configs'] > 0:
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
