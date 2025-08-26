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
import re
from datetime import datetime
from typing import Dict, List, Tuple
import signal
import json
from tqdm import tqdm

# Test configuration
STRESS_TESTS = ['cpu', 'int64', 'fp-math', 'cache', 'stream']
CPU_TARGETS = list(range(1, 101, 5)) + [100]  # 1, 6, 11, ..., 96, 100
TEST_DURATION = 10  # seconds
WORKERS = os.cpu_count()  # Number of CPU cores

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
            with open('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor', 'r') as f:
                self.original_governor = f.read().strip()
            
            # Set performance governor for all CPUs
            for cpu in range(os.cpu_count()):
                governor_path = f'/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor'
                if os.path.exists(governor_path):
                    subprocess.run(['sudo', 'sh', '-c', f'echo performance > {governor_path}'], 
                                 check=True, capture_output=True)
        except Exception as e:
            print(f"Warning: Could not set CPU governor: {e}")
        
        # Disable Intel Turbo Boost if available
        turbo_path = '/sys/devices/system/cpu/intel_pstate/no_turbo'
        if os.path.exists(turbo_path):
            try:
                with open(turbo_path, 'r') as f:
                    self.original_turbo = f.read().strip()
                subprocess.run(['sudo', 'sh', '-c', f'echo 1 > {turbo_path}'], 
                             check=True, capture_output=True)
            except Exception as e:
                print(f"Warning: Could not disable Intel Turbo: {e}")
    
    def restore_power_saving(self):
        """Restore original power saving settings."""
        print("Restoring power saving modes...")
        
        # Restore CPU governor
        if self.original_governor:
            for cpu in range(os.cpu_count()):
                governor_path = f'/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor'
                if os.path.exists(governor_path):
                    subprocess.run(['sudo', 'sh', '-c', 
                                  f'echo {self.original_governor} > {governor_path}'], 
                                 check=False, capture_output=True)
        
        # Restore Intel Turbo Boost
        if self.original_turbo is not None:
            turbo_path = '/sys/devices/system/cpu/intel_pstate/no_turbo'
            if os.path.exists(turbo_path):
                subprocess.run(['sudo', 'sh', '-c', 
                              f'echo {self.original_turbo} > {turbo_path}'], 
                             check=False, capture_output=True)


def parse_stress_ng_output(output: str) -> Dict[str, float]:
    """Parse stress-ng output to extract metrics."""
    metrics = {}
    
    # Look for metric lines starting with "stress-ng: metrc:"
    # The format is: stress-ng: metrc: [PID] test_name bogo_ops real_time ... bogo_ops_per_sec ...
    lines = output.split('\n')
    for line in lines:
        if 'stress-ng: metrc:' in line and 'stressor' not in line:
            # This is a data line, not the header
            parts = line.split()
            if len(parts) >= 8:
                try:
                    # Format: stress-ng: metrc: [PID] test_name bogo_ops real_time usr_time sys_time bogo_ops/s(real) bogo_ops/s(usr+sys) ...
                    # Index 3 is test name, 4 is bogo ops, 8 is bogo ops/s (real time)
                    metrics['total_bogo_ops'] = float(parts[4])
                    metrics['bogo_ops_per_sec'] = float(parts[8])
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


def run_stress_test(test_type: str, cpu_target: int, duration: int) -> Dict[str, any]:
    """Run a single stress-ng test and collect metrics."""
    result = {
        'timestamp': datetime.now().isoformat(),
        'test_type': test_type,
        'cpu_target': cpu_target,
        'duration': duration,
        'workers': WORKERS,
        'actual_cpu_utilization': 0.0,
        'bogo_ops_per_sec': 0.0,
        'total_bogo_ops': 0.0,
        'error': None
    }
    
    # Build stress-ng command
    cmd = [
        'stress-ng',
        f'--{test_type}', str(WORKERS),
        '--timeout', f'{duration}s',
        '--metrics',
        '--cpu-load', str(cpu_target)
    ]
    
    # Special handling for cache test
    if test_type == 'cache':
        cmd = [
            'stress-ng',
            '--cache', str(WORKERS),
            '--cache-level', '3',  # L3 cache
            '--timeout', f'{duration}s',
            '--metrics',
            '--cpu-load', str(cpu_target)
        ]
    
    try:
        # Start stress-ng process
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                 text=True, preexec_fn=os.setsid)
        
        # Start CPU measurement in background
        cpu_measurements = []
        start_time = time.time()
        
        # Measure CPU while stress-ng runs
        while process.poll() is None and (time.time() - start_time) < duration + 2:
            cpu_percent = psutil.cpu_percent(interval=0.5)
            if time.time() - start_time > 1:  # Skip first second for warmup
                cpu_measurements.append(cpu_percent)
        
        # Get process output
        stdout, stderr = process.communicate(timeout=5)
        
        # Calculate average CPU utilization
        if cpu_measurements:
            result['actual_cpu_utilization'] = sum(cpu_measurements) / len(cpu_measurements)
        
        # Parse stress-ng metrics
        metrics = parse_stress_ng_output(stdout + stderr)
        result.update(metrics)
        
        if process.returncode != 0:
            result['error'] = f"stress-ng exited with code {process.returncode}"
    
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        result['error'] = "Test timeout"
    except Exception as e:
        result['error'] = str(e)
    
    return result


def main():
    """Main test runner."""
    # Check if stress-ng is installed
    try:
        subprocess.run(['stress-ng', '--version'], check=True, capture_output=True)
    except FileNotFoundError:
        print("Error: stress-ng is not installed. Please install it:")
        print("  Ubuntu/Debian: sudo apt-get install stress-ng")
        print("  Fedora/RHEL: sudo dnf install stress-ng")
        sys.exit(1)
    
    # Initialize power manager
    power_manager = PowerManager()
    
    # Results file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f'stress_test_results_{timestamp}.csv'
    
    # CSV headers
    headers = ['timestamp', 'test_type', 'cpu_target', 'duration', 'workers',
               'actual_cpu_utilization', 'bogo_ops_per_sec', 'total_bogo_ops', 'error']
    
    results = []
    
    try:
        # Disable power saving
        power_manager.disable_power_saving()
        
        # Wait for system to stabilize
        print("Waiting for system to stabilize...")
        time.sleep(2)
        
        # Run tests
        total_tests = len(STRESS_TESTS) * len(CPU_TARGETS)
        
        # Create progress bar
        with tqdm(total=total_tests, desc="Running stress tests", unit="test") as pbar:
            for test_type in STRESS_TESTS:
                for cpu_target in CPU_TARGETS:
                    # Update progress bar description
                    pbar.set_description(f"Running {test_type} @ {cpu_target}% CPU")
                    
                    # Run the test
                    result = run_stress_test(test_type, cpu_target, TEST_DURATION)
                    results.append(result)
                    
                    # Save results incrementally
                    with open(results_file, 'w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=headers)
                        writer.writeheader()
                        writer.writerows(results)
                    
                    # Update postfix with latest results
                    pbar.set_postfix({
                        'Actual CPU': f"{result['actual_cpu_utilization']:.1f}%",
                        'Bogo ops/s': f"{result['bogo_ops_per_sec']:.1f}"
                    })
                    
                    # Update progress
                    pbar.update(1)
                    
                    # Cool down between tests
                    time.sleep(2)
    
    finally:
        # Always restore power settings
        power_manager.restore_power_saving()
    
    print(f"\nResults saved to: {results_file}")
    print(f"Total tests completed: {len(results)}")
    
    # Summary statistics
    successful_results = [r for r in results if r['error'] is None]
    if successful_results:
        print(f"\nSummary:")
        for test_type in STRESS_TESTS:
            type_results = [r for r in successful_results if r['test_type'] == test_type]
            if type_results:
                avg_bogo = sum(r['bogo_ops_per_sec'] for r in type_results) / len(type_results)
                print(f"  {test_type}: avg {avg_bogo:.1f} bogo ops/s")


if __name__ == '__main__':
    main()