#!/usr/bin/env python3
"""
Run stress-ng tests with varying CPU utilization targets and collect performance metrics.
"""

import subprocess
import time
import psutil
import csv
import os
import sys
from datetime import datetime
from typing import Dict
import signal
from tqdm import tqdm
import argparse

# Default test configuration
AVAILABLE_TESTS = ["cpu", "int64", "double", "matrixprod", "cache", "stream"]
DEFAULT_TESTS = ["cpu", "int64", "double", "matrixprod"]
DEFAULT_TEST_DURATION = 10  # seconds


class PowerManager:
    """Manage CPU power saving modes."""

    def __init__(self):
        self.original_governor = None
        self.original_turbo = None

    def disable_power_saving(self):
        """Disable CPU power saving features."""
        print("Disabling power saving modes...")

        # Store original CPU governor
        try:
            with open(
                "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor", "r"
            ) as f:
                self.original_governor = f.read().strip()

            # Set performance governor for all CPUs
            for cpu in range(os.cpu_count()):
                governor_path = (
                    f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor"
                )
                if os.path.exists(governor_path):
                    subprocess.run(
                        ["sudo", "sh", "-c", f"echo performance > {governor_path}"],
                        check=True,
                        capture_output=True,
                    )
        except Exception as e:
            print(f"Warning: Could not set CPU governor: {e}")

        # Disable Intel Turbo Boost if available
        turbo_path = "/sys/devices/system/cpu/intel_pstate/no_turbo"
        if os.path.exists(turbo_path):
            try:
                with open(turbo_path, "r") as f:
                    self.original_turbo = f.read().strip()
                subprocess.run(
                    ["sudo", "sh", "-c", f"echo 1 > {turbo_path}"],
                    check=True,
                    capture_output=True,
                )
            except Exception as e:
                print(f"Warning: Could not disable Intel Turbo: {e}")

    def restore_power_saving(self):
        """Restore original power saving settings."""
        print("Restoring power saving modes...")

        # Restore CPU governor
        if self.original_governor:
            for cpu in range(os.cpu_count()):
                governor_path = (
                    f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor"
                )
                if os.path.exists(governor_path):
                    subprocess.run(
                        [
                            "sudo",
                            "sh",
                            "-c",
                            f"echo {self.original_governor} > {governor_path}",
                        ],
                        check=False,
                        capture_output=True,
                    )

        # Restore Intel Turbo Boost
        if self.original_turbo is not None:
            turbo_path = "/sys/devices/system/cpu/intel_pstate/no_turbo"
            if os.path.exists(turbo_path):
                subprocess.run(
                    ["sudo", "sh", "-c", f"echo {self.original_turbo} > {turbo_path}"],
                    check=False,
                    capture_output=True,
                )


def parse_stress_ng_output(output: str) -> Dict[str, float]:
    """Parse stress-ng output to extract metrics."""
    metrics = {}

    # Look for metric lines starting with "stress-ng: metrc:"
    # The format is: stress-ng: metrc: [PID] test_name bogo_ops real_time ... bogo_ops_per_sec ...
    lines = output.split("\n")
    for line in lines:
        if "stress-ng: metrc:" in line and "stressor" not in line:
            # This is a data line, not the header
            parts = line.split()
            if len(parts) >= 8:
                try:
                    # Format: stress-ng: metrc: [PID] test_name bogo_ops real_time usr_time sys_time bogo_ops/s(real) bogo_ops/s(usr+sys) ...
                    # Index 3 is test name, 4 is bogo ops, 8 is bogo ops/s (real time)
                    metrics["total_bogo_ops"] = float(parts[4])
                    metrics["bogo_ops_per_sec"] = float(parts[8])
                except (ValueError, IndexError):
                    pass

    return metrics


def measure_cpu_utilization(duration: int, interval: float = 0.1) -> float:
    """Measure average CPU utilization over a duration."""
    measurements = []
    start_time = time.time()

    while time.time() - start_time < duration:
        cpu_percent = psutil.cpu_percent(interval=interval)
        measurements.append(cpu_percent)

    return sum(measurements) / len(measurements) if measurements else 0.0


def get_cpu_frequencies() -> Dict[str, float]:
    """Get current CPU frequencies for all cores."""
    frequencies = []
    for cpu in range(os.cpu_count()):
        freq_path = f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq"
        if os.path.exists(freq_path):
            try:
                with open(freq_path, "r") as f:
                    freq_khz = float(f.read().strip())
                    frequencies.append(freq_khz / 1000)  # Convert to MHz
            except Exception:
                pass
    
    if frequencies:
        return {
            "avg": sum(frequencies) / len(frequencies),
            "max": max(frequencies),
            "min": min(frequencies)
        }
    return {"avg": 0.0, "max": 0.0, "min": 0.0}


def run_stress_test(
    test_type: str, cpu_target: int, duration: int, workers: int = None
) -> Dict[str, any]:
    """Run a single stress-ng test and collect metrics."""
    if workers is None:
        workers = WORKERS

    result = {
        "timestamp": datetime.now().isoformat(),
        "test_type": test_type,
        "cpu_target": cpu_target,
        "duration": duration,
        "workers": workers,
        "actual_cpu_utilization": 0.0,
        "bogo_ops_per_sec": 0.0,
        "total_bogo_ops": 0.0,
        "avg_clock_speed_mhz": 0.0,
        "max_clock_speed_mhz": 0.0,
        "error": None,
    }

    # Build stress-ng command
    # These are CPU methods, not standalone stressors
    if test_type in ["double", "int64", "float", "matrixprod"]:
        cmd = [
            "stress-ng",
            "--cpu",
            str(workers),
            "--cpu-method",
            test_type,
            "--timeout",
            f"{duration}s",
            "--metrics",
        ]
        # Only add cpu-load if not running at 100%
        if cpu_target < 100:
            cmd.extend(["--cpu-load", str(cpu_target)])
    elif test_type == "cache":
        cmd = [
            "stress-ng",
            "--cache",
            str(workers),
            "--cache-level",
            "3",  # L3 cache
            "--timeout",
            f"{duration}s",
            "--metrics",
        ]
        # Only add cpu-load if not running at 100%
        if cpu_target < 100:
            cmd.extend(["--cpu-load", str(cpu_target)])
    else:
        # Default case for cpu, stream, and other standalone stressors
        cmd = [
            "stress-ng",
            f"--{test_type}",
            str(workers),
            "--timeout",
            f"{duration}s",
            "--metrics",
        ]
        # Only add cpu-load if not running at 100%
        if cpu_target < 100:
            cmd.extend(["--cpu-load", str(cpu_target)])

    try:
        # Start stress-ng process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid,
        )

        # Start CPU measurement in background
        cpu_measurements = []
        clock_speed_measurements = []
        max_clock_speeds = []
        start_time = time.time()

        # Measure CPU and clock speeds while stress-ng runs
        while process.poll() is None and (time.time() - start_time) < duration + 2:
            cpu_percent = psutil.cpu_percent(interval=0.5)
            freq_stats = get_cpu_frequencies()
            
            if time.time() - start_time > 1:  # Skip first second for warmup
                cpu_measurements.append(cpu_percent)
                clock_speed_measurements.append(freq_stats["avg"])
                max_clock_speeds.append(freq_stats["max"])
        
        # Remove the last two measurements since they can include a drop at the end
        if len(cpu_measurements) >= 2:
            cpu_measurements.pop()
            cpu_measurements.pop()
        if len(clock_speed_measurements) >= 2:
            clock_speed_measurements.pop()
            clock_speed_measurements.pop()
        if len(max_clock_speeds) >= 2:
            max_clock_speeds.pop()
            max_clock_speeds.pop()

        # Get process output
        stdout, stderr = process.communicate(timeout=5)

        # Calculate average CPU utilization and clock speeds
        if cpu_measurements:
            result["actual_cpu_utilization"] = sum(cpu_measurements) / len(
                cpu_measurements
            )
        
        if clock_speed_measurements:
            result["avg_clock_speed_mhz"] = sum(clock_speed_measurements) / len(
                clock_speed_measurements
            )
        
        if max_clock_speeds:
            result["max_clock_speed_mhz"] = max(max_clock_speeds)

        # Parse stress-ng metrics
        metrics = parse_stress_ng_output(stdout + stderr)
        result.update(metrics)

        if process.returncode != 0:
            result["error"] = f"stress-ng exited with code {process.returncode}"

    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        result["error"] = "Test timeout"
    except Exception as e:
        result["error"] = str(e)

    return result


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run stress-ng tests with varying CPU utilization targets and collect performance metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all default tests with default increments
  %(prog)s
  
  # Run only cpu and int64 tests
  %(prog)s --tests cpu int64
  
  # Run all available tests
  %(prog)s --tests all
  
  # Run tests with CPU utilization from 10%% to 100%% in steps of 10
  %(prog)s --cpu-start 10 --cpu-end 100 --cpu-step 10
  
  # Run tests with 1, 4, 8, 16 workers
  %(prog)s --workers 1 4 8 16
  
  # Run tests with workers from 1 to 8 in steps of 2
  %(prog)s --worker-start 1 --worker-end 8 --worker-step 2
  
  # Disable variable worker tests (only run CPU utilization tests)
  %(prog)s --no-worker-tests
  
  # Disable CPU utilization tests (only run worker scaling tests)
  %(prog)s --no-cpu-tests
        """
    )
    
    # Test selection
    parser.add_argument(
        "--tests",
        nargs="+",
        default=DEFAULT_TESTS,
        help=f"Tests to run. Available: {', '.join(AVAILABLE_TESTS)}, 'all'. Default: {' '.join(DEFAULT_TESTS)}"
    )
    
    # CPU utilization range arguments
    parser.add_argument(
        "--cpu-start",
        type=int,
        default=1,
        help="Starting CPU utilization percentage (default: 1)"
    )
    parser.add_argument(
        "--cpu-end",
        type=int,
        default=100,
        help="Ending CPU utilization percentage (default: 100)"
    )
    parser.add_argument(
        "--cpu-step",
        type=int,
        default=1,
        help="CPU utilization increment step (default: 1)"
    )
    parser.add_argument(
        "--cpu-targets",
        nargs="+",
        type=int,
        help="Specific CPU utilization targets (overrides --cpu-start/end/step)"
    )
    
    # Worker count arguments
    parser.add_argument(
        "--worker-start",
        type=int,
        default=1,
        help="Starting number of workers (default: 1)"
    )
    parser.add_argument(
        "--worker-end",
        type=int,
        default=os.cpu_count(),
        help=f"Ending number of workers (default: {os.cpu_count()} - number of CPU cores)"
    )
    parser.add_argument(
        "--worker-step",
        type=int,
        default=1,
        help="Worker count increment step (default: 1)"
    )
    parser.add_argument(
        "--workers",
        nargs="+",
        type=int,
        help="Specific worker counts to test (overrides --worker-start/end/step)"
    )
    parser.add_argument(
        "--fixed-workers",
        type=int,
        default=os.cpu_count(),
        help=f"Number of workers for CPU utilization tests (default: {os.cpu_count()})"
    )
    
    # Test duration
    parser.add_argument(
        "--duration",
        type=int,
        default=DEFAULT_TEST_DURATION,
        help=f"Test duration in seconds (default: {DEFAULT_TEST_DURATION})"
    )
    
    # Test type selection
    parser.add_argument(
        "--no-cpu-tests",
        action="store_true",
        help="Skip variable CPU utilization tests"
    )
    parser.add_argument(
        "--no-worker-tests",
        action="store_true",
        help="Skip variable worker count tests"
    )
    
    # Power management
    parser.add_argument(
        "--no-power-management",
        action="store_true",
        help="Don't disable power saving modes"
    )
    
    # Output file
    parser.add_argument(
        "--output",
        type=str,
        help="Output CSV file name (default: stress_test_results_TIMESTAMP.csv)"
    )
    
    args = parser.parse_args()
    
    # Process test selection
    if 'all' in args.tests:
        args.tests = AVAILABLE_TESTS
    else:
        # Validate test names
        invalid_tests = [t for t in args.tests if t not in AVAILABLE_TESTS]
        if invalid_tests:
            parser.error(f"Invalid tests: {', '.join(invalid_tests)}. Available: {', '.join(AVAILABLE_TESTS)}")
    
    # Process CPU targets
    if args.cpu_targets:
        args.cpu_targets = sorted(set(args.cpu_targets))  # Remove duplicates and sort
    else:
        args.cpu_targets = list(range(args.cpu_start, args.cpu_end + 1, args.cpu_step))
    
    # Validate CPU targets
    invalid_cpu = [c for c in args.cpu_targets if c < 1 or c > 100]
    if invalid_cpu:
        parser.error(f"CPU targets must be between 1 and 100: {invalid_cpu}")
    
    # Process worker counts
    if args.workers:
        args.workers = sorted(set(args.workers))  # Remove duplicates and sort
    else:
        args.workers = list(range(args.worker_start, args.worker_end + 1, args.worker_step))
    
    # Validate worker counts
    max_workers = os.cpu_count() * 2  # Allow up to 2x CPU count
    invalid_workers = [w for w in args.workers if w < 1 or w > max_workers]
    if invalid_workers:
        parser.error(f"Worker counts must be between 1 and {max_workers}: {invalid_workers}")
    
    # Validate fixed workers
    if args.fixed_workers < 1 or args.fixed_workers > max_workers:
        parser.error(f"Fixed workers must be between 1 and {max_workers}")
    
    # Check that at least one test type is enabled
    if args.no_cpu_tests and args.no_worker_tests:
        parser.error("Cannot disable both CPU and worker tests")
    
    return args


def main():
    """Main test runner."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Check if stress-ng is installed
    try:
        subprocess.run(["stress-ng", "--version"], check=True, capture_output=True)
    except FileNotFoundError:
        print("Error: stress-ng is not installed. Please install it:")
        print("  Ubuntu/Debian: sudo apt-get install stress-ng")
        print("  Fedora/RHEL: sudo dnf install stress-ng")
        sys.exit(1)

    # Initialize power manager if needed
    power_manager = PowerManager() if not args.no_power_management else None

    # Results file
    if args.output:
        results_file = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"stress_test_results_{timestamp}.csv"

    # CSV headers
    headers = [
        "timestamp",
        "test_type",
        "cpu_target",
        "duration",
        "workers",
        "actual_cpu_utilization",
        "avg_clock_speed_mhz",
        "max_clock_speed_mhz",
        "bogo_ops_per_sec",
        "total_bogo_ops",
        "error",
    ]

    results = []

    try:
        # Disable power saving if enabled
        if power_manager:
            power_manager.disable_power_saving()

        # Wait for system to stabilize
        print("Waiting for system to stabilize...")
        time.sleep(2)

        # Print test configuration
        print(f"\nTest Configuration:")
        print(f"  Tests: {', '.join(args.tests)}")
        if not args.no_cpu_tests:
            print(f"  CPU targets: {len(args.cpu_targets)} values from {min(args.cpu_targets)}% to {max(args.cpu_targets)}%")
            print(f"  Fixed workers for CPU tests: {args.fixed_workers}")
        if not args.no_worker_tests:
            print(f"  Worker counts: {', '.join(map(str, args.workers))}")
        print(f"  Test duration: {args.duration} seconds")
        print()

        # Run tests
        # Calculate total tests
        total_tests = 0
        if not args.no_cpu_tests:
            total_tests += len(args.tests) * len(args.cpu_targets)
        if not args.no_worker_tests:
            total_tests += len(args.tests) * len(args.workers)

        # Create progress bar
        with tqdm(total=total_tests, desc="Running stress tests", unit="test") as pbar:
            # First run variable CPU utilization tests with fixed workers
            if not args.no_cpu_tests:
                for test_type in args.tests:
                    for cpu_target in args.cpu_targets:
                        # Update progress bar description
                        pbar.set_description(
                            f"Running {test_type} @ {cpu_target}% CPU ({args.fixed_workers} workers)"
                        )

                        # Run the test
                        result = run_stress_test(
                            test_type, cpu_target, args.duration, workers=args.fixed_workers
                        )
                        results.append(result)

                    # Save results incrementally
                    with open(results_file, "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=headers)
                        writer.writeheader()
                        writer.writerows(results)

                    # Update postfix with latest results
                    pbar.set_postfix(
                        {
                            "CPU": f"{result['actual_cpu_utilization']:.1f}%",
                            "Clock": f"{result['avg_clock_speed_mhz']:.0f}MHz",
                            "Bogo ops/s": f"{result['bogo_ops_per_sec']:.1f}",
                        }
                    )

                    # Update progress
                    pbar.update(1)

                    # Cool down between tests
                    time.sleep(2)

            # Then run fixed worker tests at 100% CPU
            if not args.no_worker_tests:
                for test_type in args.tests:
                    for worker_count in args.workers:
                        # Update progress bar description
                        pbar.set_description(
                            f"Running {test_type} @ 100% CPU ({worker_count} workers)"
                        )

                        # Run the test with fixed number of workers at 100% CPU
                        result = run_stress_test(
                            test_type, 100, args.duration, workers=worker_count
                        )
                        results.append(result)

                    # Save results incrementally
                    with open(results_file, "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=headers)
                        writer.writeheader()
                        writer.writerows(results)

                    # Update postfix with latest results
                    pbar.set_postfix(
                        {
                            "CPU": f"{result['actual_cpu_utilization']:.1f}%",
                            "Clock": f"{result['avg_clock_speed_mhz']:.0f}MHz",
                            "Bogo ops/s": f"{result['bogo_ops_per_sec']:.1f}",
                        }
                    )

                    # Update progress
                    pbar.update(1)

                    # Cool down between tests
                    time.sleep(2)

    finally:
        # Restore power settings if they were changed
        if power_manager:
            power_manager.restore_power_saving()

    print(f"\nResults saved to: {results_file}")
    print(f"Total tests completed: {len(results)}")

    # Summary statistics
    successful_results = [r for r in results if r["error"] is None]
    if successful_results:
        print("\nSummary:")
        for test_type in args.tests:
            type_results = [
                r for r in successful_results if r["test_type"] == test_type
            ]
            if type_results:
                avg_bogo = sum(r["bogo_ops_per_sec"] for r in type_results) / len(
                    type_results
                )
                print(f"  {test_type}: avg {avg_bogo:.1f} bogo ops/s")


if __name__ == "__main__":
    main()
