#!/usr/bin/env python3
"""
Test runner script for fiberwise-common.

This script provides convenient test execution with different configurations
and reporting options.
"""
import subprocess
import sys
import argparse
from pathlib import Path


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"❌ Command not found: {cmd[0]}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run tests for fiberwise-common")
    parser.add_argument(
        "--type", 
        choices=["unit", "integration", "all"],
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--coverage",
        action="store_true", 
        help="Run with coverage reporting"
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate HTML coverage report"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--parallel",
        "-n",
        type=int,
        default=1,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--slow",
        action="store_true",
        help="Include slow tests"
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install test requirements first"
    )
    
    args = parser.parse_args()
    
    # Change to the directory containing this script
    script_dir = Path(__file__).parent
    sys.path.insert(0, str(script_dir))
    
    success = True
    
    # Install requirements if requested
    if args.install:
        print("Installing test requirements...")
        success = run_command([
            sys.executable, "-m", "pip", "install", "-r", "requirements-test.txt"
        ], "Installing test requirements")
        if not success:
            return 1
    
    # Build pytest command
    pytest_cmd = [sys.executable, "-m", "pytest"]
    
    # Add verbosity
    if args.verbose:
        pytest_cmd.append("-v")
    else:
        pytest_cmd.append("-q")
    
    # Add parallel execution
    if args.parallel > 1:
        pytest_cmd.extend(["-n", str(args.parallel)])
    
    # Add coverage if requested
    if args.coverage:
        pytest_cmd.extend([
            "--cov=fiberwise_common",
            "--cov-report=term-missing"
        ])
        
        if args.html:
            pytest_cmd.extend([
                "--cov-report=html:htmlcov"
            ])
    
    # Add test markers based on type
    if args.type == "unit":
        pytest_cmd.extend(["-m", "unit"])
    elif args.type == "integration":
        pytest_cmd.extend(["-m", "integration"])
    elif not args.slow:
        pytest_cmd.extend(["-m", "not slow"])
    
    # Add test directories
    if args.type == "unit":
        pytest_cmd.append("tests/unit")
    elif args.type == "integration":
        pytest_cmd.append("tests/integration") 
    else:
        pytest_cmd.append("tests/")
    
    # Run the tests
    success = run_command(pytest_cmd, f"Running {args.type} tests")
    
    # Run additional checks if all tests passed
    if success and args.type == "all":
        print("\n" + "="*60)
        print("All tests passed! Running additional quality checks...")
        print("="*60)
        
        # Type checking (if mypy is available)
        type_check_cmd = [sys.executable, "-m", "mypy", "fiberwise_common", "--ignore-missing-imports"]
        run_command(type_check_cmd, "Type checking with mypy")
        
        # Code formatting check (if black is available)
        format_check_cmd = [sys.executable, "-m", "black", "--check", "--diff", "fiberwise_common", "tests"]
        run_command(format_check_cmd, "Code formatting check with black")
    
    # Print summary
    print("\n" + "="*60)
    if success:
        print("🎉 All tests completed successfully!")
        if args.coverage and args.html:
            print("📊 Coverage report generated in htmlcov/index.html")
    else:
        print("💥 Some tests failed. Check the output above for details.")
    print("="*60)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())