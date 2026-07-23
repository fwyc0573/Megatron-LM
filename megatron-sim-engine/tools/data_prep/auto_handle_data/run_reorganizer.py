#!/usr/bin/env python3
"""
Simple runner script for MoE Data Reorganizer

This script provides a user-friendly interface to run the data reorganization process
with various options and safety checks.
"""

import sys
import argparse
from pathlib import Path
import logging

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from moe_data_reorganizer import MoEDataReorganizer

def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('moe_reorganizer_run.log'),
            logging.StreamHandler()
        ]
    )

def run_reorganization(base_dir: str) -> bool:
    """Run the main reorganization process"""
    
    print("\n🚀 Starting data reorganization...")
    
    # Initialize and run reorganizer
    reorganizer = MoEDataReorganizer(base_dir)
    success = reorganizer.run()
    
    if success:
        print("\n✅ Data reorganization completed successfully!")
        print(f"📁 Results saved to: {reorganizer.target_base}")
        print("📊 Check reorganization_summary.txt for detailed report")
        print("📋 Check moe_reorganizer_run.log for detailed logs")
    else:
        print("\n❌ Data reorganization failed!")
        print("📋 Check moe_reorganizer_run.log for error details")
    
    return success

def interactive_mode():
    """Run in interactive mode with user prompts"""
    print("🎯 MoE Data Reorganizer - Interactive Mode")
    print("=" * 50)
    
    # Get base directory
    default_base = "."
    base_dir = input(f"Enter base directory (default: {default_base}): ").strip()
    if not base_dir:
        base_dir = default_base
    
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"❌ Base directory does not exist: {base_path}")
        return False
    
    print(f"📁 Using base directory: {base_path.absolute()}")
    
    # Ask for confirmation
    print("\n📋 Configuration Summary:")
    print(f"  Base directory: {base_path.absolute()}")
    print(f"  Source: moe_mg/node1/, moe_mg/node2/")
    print(f"  Target: simulation_inputs/megatron_operation_log/new_moe/")
    
    confirm = input("\nProceed with reorganization? (y/N): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("❌ Operation cancelled by user")
        return False
    
    # Run reorganization
    return run_reorganization(str(base_path))

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="MoE Model Distributed Training Data Reorganizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_reorganizer.py                    # Interactive mode
  python run_reorganizer.py --auto             # Automatic mode with current directory
  python run_reorganizer.py --base-dir /path   # Specify base directory
        """
    )
    
    parser.add_argument(
        '--base-dir', '-d',
        type=str,
        default='.',
        help='Base directory containing moe_mg/ folder (default: current directory)'
    )
    
    parser.add_argument(
        '--auto', '-a',
        action='store_true',
        help='Run in automatic mode without prompts'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Validate base directory
    base_path = Path(args.base_dir)
    if not base_path.exists():
        print(f"❌ Base directory does not exist: {base_path}")
        return 1
    
    try:
        if args.auto:
            # Automatic mode
            print("🤖 Running in automatic mode...")
            success = run_reorganization(str(base_path))
            return 0 if success else 1
        
        else:
            # Interactive mode
            success = interactive_mode()
            return 0 if success else 1
    
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user (Ctrl+C)")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        logging.exception("Unexpected error occurred")
        return 1

if __name__ == "__main__":
    exit(main())
