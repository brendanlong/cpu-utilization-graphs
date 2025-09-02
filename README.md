# CPU Utilization and Performance Testing with stress-ng

This repository contains Python scripts for running systematic CPU stress tests using stress-ng and analyzing the relationship between reported CPU utilization and actual performance (measured in Bogo operations per second).

## Overview

The scripts run stress-ng tests to explore:
- How reported CPU utilization relates to actual computational work performed
- Performance scaling with different numbers of worker threads
- Clock speed behavior under various load conditions
- Non-linear relationships between CPU utilization metrics

## Requirements

- Python 3.8+
- stress-ng
- Root/sudo access (for CPU power management)

### Installing stress-ng

```bash
# Ubuntu/Debian
apt-get install stress-ng
```

### Python Dependencies

Install using uv:
```bash
uv sync
```

Or with pip:
```bash
pip install psutil tqdm polars matplotlib numpy scipy
```

## Usage

### 1. Run Stress Tests

```bash
python run_stress_tests.py [options]
```

The script will:
- Optionally disable CPU power saving modes for consistent results
- Run two types of test configurations:
  - Variable CPU utilization (1-100%) with fixed worker count
  - Variable worker count at 100% CPU utilization
- Measure actual CPU utilization, clock speeds, and Bogo operations
- Save results to CSV with timestamps
- Display progress with real-time metrics

### 2. Analyze CPU Scaling

```bash
python analyze_cpu_scaling.py stress_test_results_YYYYMMDD_HHMMSS.csv
```

This script performs detailed analysis of the stress test results:

#### Generated Outputs:

For each test type (cpu, int64, double, matrixprod):
- `cpu_utilization_analysis_{test_type}.png` - Dual plot showing:
  - Left: Relationship between reported and adjusted CPU utilization
  - Right: Performance scaling with worker count
- `cpu_utilization_mapping_{test_type}.csv` - Detailed mapping data

Combined visualizations:
- `cpu_utilization_combined_adjusted.png` - All test types on one plot showing adjusted vs reported CPU
- `clock_speed_vs_cpu_all.png` - Clock speed behavior under different loads (if clock speed data available)

#### Analysis Features:
- Calculates "adjusted" CPU utilization based on actual work performed (Bogo operations)
- Fits piecewise linear models to identify performance breakpoints
- Shows non-linear relationships between reported CPU % and actual computational work
- Identifies scaling limits and efficiency changes

## Output Files

### CSV Format

The stress test results CSV contains:
- `timestamp` - ISO format timestamp
- `test_type` - Type of stress test (cpu, int64, double, matrixprod, cache, stream)
- `cpu_target` - Target CPU utilization percentage
- `duration` - Test duration in seconds
- `workers` - Number of CPU workers
- `actual_cpu_utilization` - Measured CPU usage percentage
- `avg_clock_speed_mhz` - Average CPU clock speed during test
- `max_clock_speed_mhz` - Maximum CPU clock speed during test
- `bogo_ops_per_sec` - Bogo operations per second
- `total_bogo_ops` - Total bogo operations
- `error` - Any error messages (null if successful)

### Analysis Outputs

The analysis script generates:
- Individual analysis plots for each test type showing both CPU scaling and worker scaling
- Combined comparison plots across all test types
- CSV mapping files with scaling factors and adjusted utilization values
- Console output with detailed statistics and regression parameters

## Available Test Types

- `cpu` - General CPU stress test
- `int64` - 64-bit integer operations  
- `double` - Double precision floating point operations
- `matrixprod` - Matrix multiplication operations
- `cache` - Cache thrashing (L3 cache)
- `stream` - Memory bandwidth test

## Power Management

The run_stress_tests.py script can optionally manage CPU power settings:
1. Saves current CPU governor and turbo boost settings
2. Sets all CPUs to "performance" governor
3. Disables Intel Turbo Boost (if available)
4. Restores original settings after tests complete

This ensures consistent performance measurements. Use `--no-power-management` to skip this.

## Key Insights

The analysis reveals:
- Reported CPU utilization often doesn't linearly correspond to actual computational work
- Performance scaling with worker threads typically shows diminishing returns
- Different test types exhibit different scaling characteristics
- Clock speed throttling affects high-utilization scenarios
- Piecewise linear models often better describe the relationships than simple linear fits