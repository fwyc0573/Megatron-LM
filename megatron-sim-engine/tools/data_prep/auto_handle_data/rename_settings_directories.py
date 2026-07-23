#!/usr/bin/env python3
"""
MoE Settings Directory Renamer

This script renames the settingXXXX directories to their full configuration names
based on the mapping information in reorganization_summary.txt.

Author: Auto-generated for MoE data processing
Date: 2025-08-19
"""

import os
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rename_settings_directories.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SettingsDirectoryRenamer:
    """Class for renaming settings directories from settingXXXX to full config names"""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.target_base = self.base_dir / "simulation_inputs" / "megatron_operation_log" / "new_moe"
        self.summary_file = self.target_base / "reorganization_summary.txt"
        self.backup_summary_file = self.target_base / "reorganization_summary_backup.txt"
        
    def validate_environment(self) -> bool:
        """Validate that the target directory and summary file exist"""
        if not self.target_base.exists():
            logger.error(f"Target directory does not exist: {self.target_base}")
            return False
            
        if not self.summary_file.exists():
            logger.error(f"Summary file does not exist: {self.summary_file}")
            return False
            
        logger.info("Environment validation passed")
        return True
    
    def parse_setting_mappings(self) -> Dict[str, str]:
        """Parse the setting mappings from reorganization_summary.txt"""
        mappings = {}
        
        try:
            with open(self.summary_file, 'r') as f:
                content = f.read()
            
            # Find the Setting Mappings section
            mapping_section = re.search(r'Setting Mappings:\s*\n-+\s*\n(.*?)\n\n', content, re.DOTALL)
            if not mapping_section:
                logger.error("Could not find Setting Mappings section in summary file")
                return {}
            
            mapping_lines = mapping_section.group(1).strip().split('\n')
            
            for line in mapping_lines:
                line = line.strip()
                if ':' in line:
                    setting_id, config_name = line.split(':', 1)
                    setting_id = setting_id.strip()
                    config_name = config_name.strip()
                    mappings[setting_id] = config_name
                    logger.debug(f"Found mapping: {setting_id} -> {config_name}")
            
            logger.info(f"Parsed {len(mappings)} setting mappings")
            return mappings
            
        except Exception as e:
            logger.error(f"Error parsing setting mappings: {e}")
            return {}
    
    def get_existing_directories(self) -> List[Path]:
        """Get list of existing settingXXXX directories"""
        existing_dirs = []
        
        for item in self.target_base.iterdir():
            if item.is_dir() and item.name.startswith('setting') and item.name[7:].isdigit():
                existing_dirs.append(item)
        
        existing_dirs.sort(key=lambda x: x.name)
        logger.info(f"Found {len(existing_dirs)} existing setting directories")
        return existing_dirs
    
    def validate_directory_structure(self, directory: Path) -> bool:
        """Validate that a directory has the expected structure"""
        expected_subdirs = ['database_profile', 'global_ranks_profile', 'schedule']
        
        for subdir in expected_subdirs:
            subdir_path = directory / subdir
            if not subdir_path.exists():
                logger.warning(f"Missing expected subdirectory: {subdir_path}")
                return False
        
        return True
    
    def rename_directory_safely(self, old_path: Path, new_name: str) -> bool:
        """Safely rename a directory with conflict checking"""
        new_path = old_path.parent / new_name
        
        # Check if target already exists
        if new_path.exists():
            logger.error(f"Target directory already exists: {new_path}")
            return False
        
        # Validate source directory structure
        if not self.validate_directory_structure(old_path):
            logger.error(f"Source directory structure validation failed: {old_path}")
            return False
        
        try:
            # Perform the rename
            old_path.rename(new_path)
            logger.info(f"Successfully renamed: {old_path.name} -> {new_name}")
            
            # Validate renamed directory structure
            if not self.validate_directory_structure(new_path):
                logger.error(f"Renamed directory structure validation failed: {new_path}")
                # Try to rename back
                try:
                    new_path.rename(old_path)
                    logger.info(f"Rolled back rename: {new_name} -> {old_path.name}")
                except Exception as rollback_e:
                    logger.error(f"Failed to rollback rename: {rollback_e}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to rename {old_path.name} to {new_name}: {e}")
            return False
    
    def update_summary_file(self, mappings: Dict[str, str], renamed_dirs: Dict[str, str]) -> bool:
        """Update the reorganization summary file with new directory names"""
        try:
            # Create backup of original summary file
            shutil.copy2(self.summary_file, self.backup_summary_file)
            logger.info(f"Created backup: {self.backup_summary_file}")
            
            # Read original content
            with open(self.summary_file, 'r') as f:
                content = f.read()
            
            # Update Setting Mappings section
            new_mappings_text = "Setting Mappings:\n--------------------\n"
            for setting_id, config_name in mappings.items():
                if setting_id in renamed_dirs:
                    # Use the new directory name (which is the config name)
                    new_mappings_text += f"{config_name}: {config_name}\n"
                else:
                    # Keep original mapping if not renamed
                    new_mappings_text += f"{setting_id}: {config_name}\n"
            
            # Replace the Setting Mappings section
            updated_content = re.sub(
                r'Setting Mappings:\s*\n-+\s*\n.*?\n\n',
                new_mappings_text + '\n',
                content,
                flags=re.DOTALL
            )
            
            # Update File Counts section to use new directory names
            for old_setting_id, new_dir_name in renamed_dirs.items():
                updated_content = updated_content.replace(f"{old_setting_id}:", f"{new_dir_name}:")
            
            # Write updated content
            with open(self.summary_file, 'w') as f:
                f.write(updated_content)
            
            logger.info("Successfully updated reorganization_summary.txt")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update summary file: {e}")
            return False
    
    def run(self) -> bool:
        """Main execution method"""
        logger.info("Starting settings directory renaming process")
        
        # Validate environment
        if not self.validate_environment():
            return False
        
        # Parse setting mappings
        mappings = self.parse_setting_mappings()
        if not mappings:
            logger.error("No setting mappings found")
            return False
        
        # Get existing directories
        existing_dirs = self.get_existing_directories()
        if not existing_dirs:
            logger.error("No existing setting directories found")
            return False
        
        # Perform renaming
        renamed_dirs = {}
        failed_renames = []
        
        for directory in existing_dirs:
            setting_id = directory.name
            
            if setting_id not in mappings:
                logger.warning(f"No mapping found for {setting_id}, skipping")
                continue
            
            config_name = mappings[setting_id]
            
            logger.info(f"Renaming {setting_id} to {config_name}")
            
            if self.rename_directory_safely(directory, config_name):
                renamed_dirs[setting_id] = config_name
            else:
                failed_renames.append(setting_id)
        
        # Update summary file
        if renamed_dirs:
            if not self.update_summary_file(mappings, renamed_dirs):
                logger.warning("Failed to update summary file, but directories were renamed")
        
        # Report results
        logger.info(f"Renaming completed:")
        logger.info(f"  Successfully renamed: {len(renamed_dirs)} directories")
        logger.info(f"  Failed renames: {len(failed_renames)} directories")
        
        if failed_renames:
            logger.error(f"Failed to rename: {failed_renames}")
        
        if renamed_dirs:
            logger.info("Successfully renamed directories:")
            for old_name, new_name in renamed_dirs.items():
                logger.info(f"  {old_name} -> {new_name}")
        
        return len(renamed_dirs) > 0
    
    def list_current_directories(self) -> None:
        """List current directories for verification"""
        logger.info("Current directories in target location:")
        
        if not self.target_base.exists():
            logger.error(f"Target directory does not exist: {self.target_base}")
            return
        
        directories = []
        for item in self.target_base.iterdir():
            if item.is_dir() and item.name != '__pycache__':
                directories.append(item.name)
        
        directories.sort()
        for i, dir_name in enumerate(directories, 1):
            logger.info(f"  {i:2d}. {dir_name}")
        
        logger.info(f"Total: {len(directories)} directories")


def main():
    """Main entry point"""
    print("MoE Settings Directory Renamer")
    print("=" * 40)
    
    # Initialize renamer
    renamer = SettingsDirectoryRenamer()
    
    # Show current state
    print("\n📁 Current directory structure:")
    renamer.list_current_directories()
    
    # Ask for confirmation
    print(f"\n🔄 This will rename settingXXXX directories to full configuration names")
    print(f"📁 Target location: {renamer.target_base}")

    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--auto':
        print("🤖 Auto mode: proceeding with renaming...")
        confirm = 'y'
    else:
        confirm = input("\nProceed with renaming? (y/N): ").strip().lower()
        if confirm not in ['y', 'yes']:
            print("❌ Operation cancelled by user")
            return 1
    
    # Run renaming process
    success = renamer.run()
    
    if success:
        print("\n✅ Directory renaming completed successfully!")
        print("📁 Updated directory structure:")
        renamer.list_current_directories()
        print(f"📊 Check {renamer.summary_file} for updated mappings")
        print(f"💾 Backup created: {renamer.backup_summary_file}")
    else:
        print("\n❌ Directory renaming failed!")
        print("📋 Check rename_settings_directories.log for error details")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
