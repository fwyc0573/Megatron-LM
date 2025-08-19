#!/usr/bin/env python3
"""
Run All CC-estimator Integration Tests

This script runs all tests for the CC-estimator integration.
"""

import os
import sys
import subprocess
import time
from typing import List, Tuple, Dict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestRunner:
    """Test runner for CC-estimator integration tests"""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = None
        self.total_time = 0
    
    def run_test_file(self, test_file: str) -> Tuple[bool, str, float]:
        """
        Run a single test file
        
        Args:
            test_file: Path to test file
            
        Returns:
            Tuple of (success, output, duration)
        """
        print(f"\n{'='*60}")
        print(f"Running: {test_file}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Run the test file
            result = subprocess.run(
                [sys.executable, test_file],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                print(f"✅ PASSED ({duration:.1f}s)")
                return True, result.stdout, duration
            else:
                print(f"❌ FAILED ({duration:.1f}s)")
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                return False, result.stderr, duration
                
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            print(f"⏰ TIMEOUT ({duration:.1f}s)")
            return False, "Test timed out", duration
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"💥 ERROR ({duration:.1f}s): {e}")
            return False, str(e), duration
    
    def run_all_tests(self) -> Dict[str, Tuple[bool, str, float]]:
        """Run all CC-estimator integration tests"""
        self.start_time = time.time()
        
        # List of test files to run
        test_files = [
            'test_basic_integration.py',
            'test_moe_communication_support.py',
            'test_parameter_semantic_alignment.py',
            'test_ml_predictor_accuracy.py',
        ]
        
        # Get the directory containing this script
        test_dir = os.path.dirname(os.path.abspath(__file__))
        
        print("CC-estimator Integration Test Suite")
        print("=" * 60)
        print(f"Test directory: {test_dir}")
        print(f"Python version: {sys.version}")
        print(f"Starting tests at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Run each test file
        for test_file in test_files:
            test_path = os.path.join(test_dir, test_file)
            
            if os.path.exists(test_path):
                success, output, duration = self.run_test_file(test_path)
                self.test_results[test_file] = (success, output, duration)
            else:
                print(f"⚠️  Test file not found: {test_path}")
                self.test_results[test_file] = (False, "File not found", 0.0)
        
        self.total_time = time.time() - self.start_time
        return self.test_results
    
    def print_summary(self):
        """Print test summary"""
        print(f"\n{'='*80}")
        print("TEST SUMMARY")
        print(f"{'='*80}")
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for success, _, _ in self.test_results.values() if success)
        failed_tests = total_tests - passed_tests
        
        print(f"Total tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
        print(f"Total time: {self.total_time:.1f}s")
        
        print(f"\nDetailed Results:")
        print("-" * 80)
        
        for test_file, (success, output, duration) in self.test_results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{test_file:<40} {status} ({duration:.1f}s)")
        
        if failed_tests > 0:
            print(f"\nFailed Test Details:")
            print("-" * 80)
            
            for test_file, (success, output, duration) in self.test_results.items():
                if not success:
                    print(f"\n{test_file}:")
                    print(output[:500] + "..." if len(output) > 500 else output)
        
        print(f"\n{'='*80}")
        
        if failed_tests == 0:
            print("🎉 All tests passed!")
        else:
            print(f"⚠️  {failed_tests} test(s) failed")
        
        return failed_tests == 0


def check_dependencies():
    """Check if required dependencies are available"""
    print("Checking dependencies...")
    
    dependencies = [
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('sklearn', 'scikit-learn'),
        ('joblib', 'joblib'),
    ]
    
    missing_deps = []
    
    for module_name, package_name in dependencies:
        try:
            __import__(module_name)
            print(f"✅ {package_name}")
        except ImportError:
            print(f"❌ {package_name} (missing)")
            missing_deps.append(package_name)
    
    # Check for CC-estimator (optional)
    try:
        from nccl_predictor import create_a100_predictor
        print("✅ CC-estimator (nccl_predictor)")
        cc_estimator_available = True
    except ImportError:
        print("⚠️  CC-estimator (nccl_predictor) - not available, some tests will be skipped")
        cc_estimator_available = False
    
    if missing_deps:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing_deps)}")
        print("Install with: pip install " + " ".join(missing_deps))
        return False, cc_estimator_available
    
    return True, cc_estimator_available


def main():
    """Main test runner"""
    print("CC-estimator Integration Test Runner")
    print("=" * 50)
    
    # Check dependencies
    deps_ok, cc_estimator_available = check_dependencies()
    
    if not deps_ok:
        print("\n❌ Missing required dependencies. Please install them first.")
        return 1
    
    if not cc_estimator_available:
        print("\n⚠️  CC-estimator not available. Some tests will be skipped.")
        print("This is expected if CC-estimator is not installed.")
    
    # Run tests
    runner = TestRunner()
    results = runner.run_all_tests()
    
    # Print summary
    all_passed = runner.print_summary()
    
    # Return appropriate exit code
    return 0 if all_passed else 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
