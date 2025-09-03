#!/usr/bin/env python3
"""
Run Phoronix Test Suite benchmarks with varying CPU core counts and collect performance metrics.
"""

import subprocess
import time
import psutil
import csv
import os
import sys
from datetime import datetime
from typing import Dict, Optional, List
import signal
from tqdm import tqdm
import argparse
import re
import threading


class PowerManager:
    """Manage CPU power saving modes."""

    def __init__(self):
        self.original_governor = None

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
            "min": min(frequencies),
        }
    return {"avg": 0.0, "max": 0.0, "min": 0.0}


def parse_phoronix_output(output: str) -> Dict[str, any]:
    """Parse Phoronix Test Suite output to extract benchmark results."""
    results = {}

    # Common patterns for different benchmark results
    patterns = [
        # Pattern for "Average: X Requests Per Second" or similar
        (r"Average:\s+([\d.]+)\s+(.+)", "average_score"),
        # Pattern for final score lines like "Score: X"
        (r"Score:\s+([\d.]+)\s*(.+)?", "score"),
        # Pattern for results like "X Requests/Sec"
        (r"([\d.]+)\s+(Requests?/Sec|req/s)", "requests_per_sec"),
        # Pattern for throughput measurements
        (r"([\d.]+)\s+(MB/s|GB/s|ops/s|Ops/Sec)", "throughput"),
        # Pattern for time measurements
        (r"([\d.]+)\s+(seconds?|ms|milliseconds?)", "time"),
        # Generic result pattern
        (r"Result:\s+([\d.]+)\s*(.+)?", "result"),
    ]

    lines = output.split("\n")
    for line in lines:
        for pattern, key in patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    unit = match.group(2) if match.lastindex > 1 else ""
                    results[key] = value
                    results[f"{key}_unit"] = unit.strip() if unit else ""
                    # Use the first match found
                    if "primary_result" not in results:
                        results["primary_result"] = value
                        results["primary_unit"] = unit.strip() if unit else ""
                except ValueError:
                    pass

    return results


def monitor_cpu_and_clock(
    stop_event: threading.Event, duration: int
) -> Dict[str, float]:
    """Monitor CPU utilization and clock speeds in a separate thread."""
    cpu_measurements = []
    clock_speed_measurements = []
    max_clock_speeds = []
    start_time = time.time()

    # Wait 2 seconds for warmup
    time.sleep(2)

    while not stop_event.is_set() and (time.time() - start_time) < duration:
        cpu_percent = psutil.cpu_percent(interval=0.5)
        freq_stats = get_cpu_frequencies()

        cpu_measurements.append(cpu_percent)
        clock_speed_measurements.append(freq_stats["avg"])
        max_clock_speeds.append(freq_stats["max"])

    # Remove the last measurement if it might be incomplete
    if cpu_measurements:
        cpu_measurements.pop()
    if clock_speed_measurements:
        clock_speed_measurements.pop()
    if max_clock_speeds:
        max_clock_speeds.pop()

    result = {
        "actual_cpu_utilization": sum(cpu_measurements) / len(cpu_measurements)
        if cpu_measurements
        else 0.0,
        "avg_clock_speed_mhz": sum(clock_speed_measurements)
        / len(clock_speed_measurements)
        if clock_speed_measurements
        else 0.0,
        "max_clock_speed_mhz": max(max_clock_speeds) if max_clock_speeds else 0.0,
    }

    return result


def run_benchmark(
    benchmark_name: str, cores: int, test_options: str = "1", timeout: int = 600
) -> Dict[str, any]:
    """Run a single Phoronix benchmark with specified core count."""

    result = {
        "timestamp": datetime.now().isoformat(),
        "benchmark": benchmark_name,
        "cores": cores,
        "actual_cpu_utilization": 0.0,
        "avg_clock_speed_mhz": 0.0,
        "max_clock_speed_mhz": 0.0,
        "primary_result": 0.0,
        "primary_unit": "",
        "error": None,
    }

    # Build taskset command
    if cores == 1:
        taskset_args = ["taskset", "-c", "0"]
    else:
        taskset_args = ["taskset", "-c", f"0-{cores - 1}"]

    # Build phoronix command
    phoronix_cmd = taskset_args + [
        "phoronix-test-suite",
        "batch-benchmark",
        benchmark_name,
    ]

    try:
        # Set environment variables for batch mode
        env = os.environ.copy()
        env["FORCE_TIMES_TO_RUN"] = "1"  # Run test only once
        env["TEST_RESULTS_NAME"] = (
            f"test_{cores}cores_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        env["TEST_RESULTS_IDENTIFIER"] = f"{cores}cores"
        env["TEST_RESULTS_DESCRIPTION"] = f"Running with {cores} cores"

        print(f"\nRunning benchmark with {cores} core(s)...")
        print(f"Command: {' '.join(phoronix_cmd)}")

        # Start the benchmark process
        process = subprocess.Popen(
            phoronix_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            preexec_fn=os.setsid,
        )

        # Start CPU monitoring in background thread
        stop_event = threading.Event()
        monitor_thread = threading.Thread(
            target=lambda: result.update(monitor_cpu_and_clock(stop_event, timeout)),
            daemon=True,
        )
        monitor_thread.start()

        # Send test options if needed (for selecting test configuration)
        stdout, stderr = process.communicate(input=test_options + "\n", timeout=timeout)

        # Stop monitoring
        stop_event.set()
        monitor_thread.join(timeout=5)

        # Parse benchmark results
        benchmark_results = parse_phoronix_output(stdout + stderr)
        result.update(benchmark_results)

        if process.returncode != 0:
            result["error"] = f"Benchmark exited with code {process.returncode}"
            # Still try to parse any results that might be available
            if not result.get("primary_result"):
                result["error"] += (
                    f"\nStderr: {stderr[:500]}"  # Include first 500 chars of error
                )

    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        stop_event.set()
        result["error"] = f"Benchmark timeout after {timeout} seconds"
    except Exception as e:
        stop_event.set()
        result["error"] = str(e)

    return result


def check_phoronix_installed() -> bool:
    """Check if Phoronix Test Suite is installed."""
    try:
        subprocess.run(
            ["phoronix-test-suite", "version"],
            check=True,
            capture_output=True,
            text=True,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def install_benchmark(benchmark_name: str) -> bool:
    """Attempt to install a benchmark if not already installed."""
    print(f"Checking if {benchmark_name} is installed...")
    try:
        # Try to install the benchmark
        process = subprocess.run(
            ["phoronix-test-suite", "install", benchmark_name],
            capture_output=True,
            text=True,
            timeout=300,
        )
        return process.returncode == 0
    except Exception as e:
        print(f"Warning: Could not install {benchmark_name}: {e}")
        return False


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Phoronix Test Suite benchmarks with varying CPU core counts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run nginx benchmark with default settings
  %(prog)s nginx
  
  # Run apache benchmark with specific core counts
  %(prog)s apache --cores 1 2 4 8 16
  
  # Run benchmark with cores from 1 to 8 in steps of 2
  %(prog)s redis --core-start 1 --core-end 8 --core-step 2
  
  # Run benchmark with custom test options
  %(prog)s nginx --test-options "2"
  
  # Run benchmark with longer timeout
  %(prog)s compilation --timeout 1200
        """,
    )

    parser.add_argument(
        "benchmark",
        help="Name of the Phoronix benchmark to run (e.g., nginx, apache, redis)",
    )

    # Core count arguments
    parser.add_argument(
        "--core-start",
        type=int,
        default=1,
        help="Starting number of cores (default: 1)",
    )
    parser.add_argument(
        "--core-end",
        type=int,
        default=os.cpu_count(),
        help=f"Ending number of cores (default: {os.cpu_count()})",
    )
    parser.add_argument(
        "--core-step",
        type=int,
        default=1,
        help="Core count increment step (default: 1)",
    )
    parser.add_argument(
        "--cores",
        nargs="+",
        type=int,
        help="Specific core counts to test (overrides --core-start/end/step)",
    )

    # Test configuration
    parser.add_argument(
        "--test-options",
        type=str,
        default="1",
        help="Options to send to the benchmark for test selection (default: '1')",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout for each benchmark run in seconds (default: 600)",
    )

    # Power management
    parser.add_argument(
        "--no-power-management",
        action="store_true",
        help="Don't disable power saving modes",
    )

    # Output file
    parser.add_argument(
        "--output",
        type=str,
        help="Output CSV file name (default: phoronix_results_BENCHMARK_TIMESTAMP.csv)",
    )

    # Installation
    parser.add_argument(
        "--no-install",
        action="store_true",
        help="Don't attempt to install the benchmark if not found",
    )

    args = parser.parse_args()

    # Process core counts
    if args.cores:
        args.cores = sorted(set(args.cores))  # Remove duplicates and sort
    else:
        args.cores = list(range(args.core_start, args.core_end + 1, args.core_step))

    # Validate core counts
    max_cores = os.cpu_count()
    invalid_cores = [c for c in args.cores if c < 1 or c > max_cores]
    if invalid_cores:
        parser.error(f"Core counts must be between 1 and {max_cores}: {invalid_cores}")

    return args


def main():
    """Main benchmark runner."""
    # Parse command-line arguments
    args = parse_arguments()

    # Check if Phoronix Test Suite is installed
    if not check_phoronix_installed():
        print("Error: Phoronix Test Suite is not installed.")
        print("Please install it from: https://www.phoronix-test-suite.com/")
        print("Or on Ubuntu/Debian: sudo apt-get install phoronix-test-suite")
        sys.exit(1)

    # Attempt to install benchmark if needed
    if not args.no_install:
        print(f"Ensuring {args.benchmark} is installed...")
        install_benchmark(args.benchmark)

    # Initialize power manager if needed
    power_manager = PowerManager() if not args.no_power_management else None

    # Results file
    if args.output:
        results_file = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"phoronix_results_{args.benchmark}_{timestamp}.csv"

    # CSV headers
    headers = [
        "timestamp",
        "benchmark",
        "cores",
        "actual_cpu_utilization",
        "avg_clock_speed_mhz",
        "max_clock_speed_mhz",
        "primary_result",
        "primary_unit",
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
        print("\nBenchmark Configuration:")
        print(f"  Benchmark: {args.benchmark}")
        print(f"  Core counts: {', '.join(map(str, args.cores))}")
        print(f"  Test options: {args.test_options}")
        print(f"  Timeout: {args.timeout} seconds")
        print()

        # Run benchmarks
        total_tests = len(args.cores)

        with tqdm(total=total_tests, desc="Running benchmarks", unit="test") as pbar:
            for core_count in args.cores:
                # Update progress bar description
                pbar.set_description(
                    f"Running {args.benchmark} with {core_count} core(s)"
                )

                # Run the benchmark
                result = run_benchmark(
                    args.benchmark, core_count, args.test_options, args.timeout
                )
                results.append(result)

                # Save results incrementally
                with open(results_file, "w", newline="") as f:
                    writer = csv.DictWriter(
                        f, fieldnames=headers, extrasaction="ignore"
                    )
                    writer.writeheader()
                    writer.writerows(results)

                # Update postfix with latest results
                postfix = {
                    "CPU": f"{result['actual_cpu_utilization']:.1f}%",
                    "Clock": f"{result['avg_clock_speed_mhz']:.0f}MHz",
                }
                if result.get("primary_result"):
                    unit = result.get("primary_unit", "")
                    postfix["Result"] = f"{result['primary_result']:.1f} {unit}"
                pbar.set_postfix(postfix)

                # Update progress
                pbar.update(1)

                # Cool down between tests
                if core_count != args.cores[-1]:  # Not the last test
                    time.sleep(5)

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
        for r in successful_results:
            if r.get("primary_result"):
                unit = r.get("primary_unit", "")
                print(
                    f"  {r['cores']} cores: {r['primary_result']:.2f} {unit} @ {r['actual_cpu_utilization']:.1f}% CPU"
                )

    # Print any errors
    failed_results = [r for r in results if r["error"] is not None]
    if failed_results:
        print("\nFailed tests:")
        for r in failed_results:
            print(f"  {r['cores']} cores: {r['error'][:100]}")


if __name__ == "__main__":
    main()
