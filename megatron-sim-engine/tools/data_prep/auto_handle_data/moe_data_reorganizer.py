#!/usr/bin/env python3
"""
MoE Model Distributed Training Data Reorganizer

This script automatically reorganizes MoE model distributed training data files
from the source directories into a structured format for analysis.

Author: Auto-generated for MoE data processing
Date: 2025-08-19
"""

import os
import shutil
import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('moe_data_reorganizer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MoEDataReorganizer:
    """Main class for reorganizing MoE distributed training data"""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.source_dirs = {
            'node1_realistic': self.base_dir / "moe_mg" / "node1" / "realistic_trace",
            'node2_realistic': self.base_dir / "moe_mg" / "node2" / "realistic_trace", 
            'node1_profiler': self.base_dir / "moe_mg" / "node1" / "profiler_log"
        }
        self.target_base = self.base_dir / "simulation_inputs" / "megatron_operation_log" / "new_moe"
        
        # Setting name mapping for consistent naming
        self.setting_counter = 1
        self.setting_mapping = {}
        
    def validate_source_directories(self) -> bool:
        """Validate that all source directories exist"""
        missing_dirs = []
        for name, path in self.source_dirs.items():
            if not path.exists():
                missing_dirs.append(f"{name}: {path}")
        
        if missing_dirs:
            logger.error(f"Missing source directories: {missing_dirs}")
            return False
        
        logger.info("All source directories validated successfully")
        return True
    
    def extract_setting_from_path(self, path_name: str) -> str:
        """Extract setting configuration from directory path name"""
        # Handle both realistic_trace format (exp) and profiler_log format (ep)
        # Convert ep to exp for consistency
        normalized = path_name.replace('_ep', '_exp')
        return normalized
    
    def get_setting_id(self, setting_name: str) -> str:
        """Get or create a setting ID for the given setting name"""
        if setting_name not in self.setting_mapping:
            setting_id = f"setting{self.setting_counter:04d}"
            self.setting_mapping[setting_name] = setting_id
            self.setting_counter += 1
            logger.info(f"Created new setting mapping: {setting_name} -> {setting_id}")
        
        return self.setting_mapping[setting_name]
    
    def scan_realistic_trace_data(self) -> Dict[str, List[Path]]:
        """Scan realistic trace data from both nodes"""
        realistic_data = defaultdict(list)
        
        for node_name, node_path in [('node1', self.source_dirs['node1_realistic']), 
                                     ('node2', self.source_dirs['node2_realistic'])]:
            if not node_path.exists():
                logger.warning(f"Node path does not exist: {node_path}")
                continue
                
            for setting_dir in node_path.iterdir():
                if setting_dir.is_dir():
                    setting_name = self.extract_setting_from_path(setting_dir.name)
                    realistic_data[setting_name].extend(list(setting_dir.glob("*.txt")))
                    logger.debug(f"Found {len(list(setting_dir.glob('*.txt')))} files in {node_name}/{setting_dir.name}")
        
        logger.info(f"Scanned realistic trace data: {len(realistic_data)} unique settings")
        return dict(realistic_data)
    
    def scan_profiler_data(self) -> Dict[str, List[Path]]:
        """Scan profiler data from node1"""
        profiler_data = defaultdict(list)
        
        profiler_path = self.source_dirs['node1_profiler']
        if not profiler_path.exists():
            logger.warning(f"Profiler path does not exist: {profiler_path}")
            return {}
            
        for setting_dir in profiler_path.iterdir():
            if setting_dir.is_dir():
                setting_name = self.extract_setting_from_path(setting_dir.name)
                profiler_data[setting_name].extend(list(setting_dir.glob("*.txt")))
                logger.debug(f"Found {len(list(setting_dir.glob('*.txt')))} files in profiler/{setting_dir.name}")
        
        logger.info(f"Scanned profiler data: {len(profiler_data)} unique settings")
        return dict(profiler_data)
    
    def create_target_structure(self, setting_id: str) -> Dict[str, Path]:
        """Create target directory structure for a setting"""
        setting_path = self.target_base / setting_id
        
        subdirs = {
            'database_profile': setting_path / 'database_profile',
            'global_ranks_profile': setting_path / 'global_ranks_profile', 
            'schedule': setting_path / 'schedule'
        }
        
        for subdir_path in subdirs.values():
            subdir_path.mkdir(parents=True, exist_ok=True)
        
        logger.debug(f"Created directory structure for {setting_id}")
        return subdirs
    
    def copy_files_safely(self, source_files: List[Path], target_dir: Path, 
                         operation_name: str) -> int:
        """Safely copy files to target directory with conflict resolution"""
        copied_count = 0
        
        for source_file in source_files:
            if not source_file.exists():
                logger.warning(f"Source file does not exist: {source_file}")
                continue
                
            target_file = target_dir / source_file.name
            
            # Handle file conflicts by adding timestamp suffix
            if target_file.exists():
                timestamp = source_file.stat().st_mtime
                name_parts = source_file.stem, int(timestamp), source_file.suffix
                target_file = target_dir / f"{name_parts[0]}_{name_parts[1]}{name_parts[2]}"
                logger.info(f"File conflict resolved: {source_file.name} -> {target_file.name}")
            
            try:
                shutil.copy2(source_file, target_file)
                copied_count += 1
                logger.debug(f"Copied: {source_file} -> {target_file}")
            except Exception as e:
                logger.error(f"Failed to copy {source_file}: {e}")
        
        logger.info(f"{operation_name}: Copied {copied_count} files to {target_dir}")
        return copied_count
    
    def process_setting(self, setting_name: str, realistic_files: List[Path], 
                       profiler_files: List[Path]) -> bool:
        """Process a single setting configuration"""
        setting_id = self.get_setting_id(setting_name)
        logger.info(f"Processing setting: {setting_name} -> {setting_id}")
        
        # Create target directory structure
        target_dirs = self.create_target_structure(setting_id)
        
        # Copy realistic trace files (groundtruth data)
        realistic_copied = 0
        if realistic_files:
            realistic_copied = self.copy_files_safely(
                realistic_files, 
                target_dirs['global_ranks_profile'],
                f"Realistic trace for {setting_id}"
            )
        
        # Copy profiler files (simulation data)
        profiler_copied = 0
        if profiler_files:
            profiler_copied = self.copy_files_safely(
                profiler_files,
                target_dirs['database_profile'], 
                f"Profiler data for {setting_id}"
            )
        
        success = (realistic_copied > 0 or profiler_copied > 0)
        if success:
            logger.info(f"Successfully processed {setting_id}: "
                       f"{realistic_copied} realistic + {profiler_copied} profiler files")
        else:
            logger.warning(f"No files processed for {setting_id}")
            
        return success
    
    def generate_summary_report(self, processed_settings: Dict[str, Dict]) -> None:
        """Generate a summary report of the reorganization process"""
        report_path = self.target_base / "reorganization_summary.txt"
        
        with open(report_path, 'w') as f:
            f.write("MoE Data Reorganization Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total settings processed: {len(processed_settings)}\n")
            f.write(f"Target directory: {self.target_base}\n\n")
            
            f.write("Setting Mappings:\n")
            f.write("-" * 20 + "\n")
            for original, setting_id in self.setting_mapping.items():
                f.write(f"{setting_id}: {original}\n")
            
            f.write("\nFile Counts by Setting:\n")
            f.write("-" * 25 + "\n")
            for setting_name, counts in processed_settings.items():
                setting_id = self.setting_mapping[setting_name]
                f.write(f"{setting_id}:\n")
                f.write(f"  - Realistic trace files: {counts['realistic']}\n")
                f.write(f"  - Profiler files: {counts['profiler']}\n")
        
        logger.info(f"Summary report generated: {report_path}")
    
    def run(self) -> bool:
        """Main execution method"""
        logger.info("Starting MoE data reorganization process")
        
        # Validate source directories
        if not self.validate_source_directories():
            return False
        
        # Scan source data
        realistic_data = self.scan_realistic_trace_data()
        profiler_data = self.scan_profiler_data()
        
        # Get all unique settings
        all_settings = set(realistic_data.keys()) | set(profiler_data.keys())
        logger.info(f"Found {len(all_settings)} unique settings to process")
        
        # Process each setting
        processed_settings = {}
        success_count = 0
        
        for setting_name in sorted(all_settings):
            realistic_files = realistic_data.get(setting_name, [])
            profiler_files = profiler_data.get(setting_name, [])
            
            if self.process_setting(setting_name, realistic_files, profiler_files):
                success_count += 1
                processed_settings[setting_name] = {
                    'realistic': len(realistic_files),
                    'profiler': len(profiler_files)
                }
        
        # Generate summary report
        self.generate_summary_report(processed_settings)
        
        logger.info(f"Reorganization completed: {success_count}/{len(all_settings)} settings processed successfully")
        return success_count > 0


def main():
    """Main entry point"""
    print("MoE Model Distributed Training Data Reorganizer")
    print("=" * 50)
    
    # Initialize reorganizer
    reorganizer = MoEDataReorganizer()
    
    # Run reorganization process
    success = reorganizer.run()
    
    if success:
        print("\n✅ Data reorganization completed successfully!")
        print(f"📁 Results saved to: {reorganizer.target_base}")
        print("📊 Check reorganization_summary.txt for detailed report")
    else:
        print("\n❌ Data reorganization failed!")
        print("📋 Check moe_data_reorganizer.log for error details")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
