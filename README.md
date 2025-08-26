# CPU Utilization and Performance Testing with stress-ng

This repository contains Python scripts for running systematic CPU stress tests using stress-ng and analyzing the relationship between CPU utilization and performance (measured in Bogo operations per second).

## Overview

The scripts run stress-ng with varying CPU utilization targets (1-100%) across different stress test types, measuring:
- Target vs actual CPU utilization
- Bogo operations per second
- Performance efficiency (Bogo OPS per CPU %)

## Requirements

- Python 3.8+
- stress-ng
- Root/sudo access (for CPU power management)

### Installing stress-ng

```bash
# Ubuntu/Debian
sudo apt-get install stress-ng

# Fedora/RHEL
sudo dnf install stress-ng

# macOS
brew install stress-ng
```

### Python Dependencies

Install using uv:
```bash
uv add psutil pandas matplotlib seaborn numpy
```

Or with pip:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Run Stress Tests

```bash
sudo python run_stress_tests.py
```

This will:
- Disable CPU power saving modes (governor set to "performance")
- Run stress tests with CPU targets: 1%, 6%, 11%, ..., 96%, 100%
- Test types: cpu, int64, fp-math, cache, stream
- Each test runs for 10 seconds
- Save results to `stress_test_results_YYYYMMDD_HHMMSS.csv`
- Restore original power settings when complete

**Note:** The script requires sudo for CPU power management. Total runtime is approximately 15-20 minutes.

### 2. Generate Graphs and Analysis

```bash
python plot_results.py stress_test_results_YYYYMMDD_HHMMSS.csv
```

This generates:
- `*_all_tests.png` - Bogo OPS vs CPU utilization for all test types
- `*_cpu_accuracy.png` - Target vs actual CPU utilization
- `*_efficiency.png` - Efficiency curves (Bogo OPS per CPU %)
- `*_heatmap.png` - Performance heatmap by test type and CPU target
- `*_summary.txt` - Statistical summary and analysis

## Output Files

### CSV Format

The results CSV contains:
- `timestamp` - ISO format timestamp
- `test_type` - Type of stress test (cpu, int64, fp-math, cache, stream)
- `cpu_target` - Target CPU utilization percentage
- `duration` - Test duration in seconds
- `workers` - Number of CPU workers
- `actual_cpu_utilization` - Measured CPU usage percentage
- `bogo_ops_per_sec` - Bogo operations per second
- `total_bogo_ops` - Total bogo operations
- `error` - Any error messages (null if successful)

### Visualizations

1. **Performance Plot** - Shows how Bogo OPS varies with CPU utilization for each test type
2. **CPU Accuracy** - Compares target vs actual CPU utilization
3. **Efficiency Plot** - Shows performance per unit of CPU usage
4. **Heatmap** - 2D visualization of performance across all parameters

## Test Types

- `cpu` - General CPU stress test
- `int64` - 64-bit integer operations
- `fp-math` - Floating point mathematics
- `cache` - Cache thrashing (L3 cache)
- `stream` - Memory bandwidth test

## Power Management

The script automatically manages CPU power settings:
1. Saves current CPU governor and turbo boost settings
2. Sets all CPUs to "performance" governor
3. Disables Intel Turbo Boost (if available)
4. Restores original settings after tests complete

This ensures consistent performance measurements across all tests.

## Troubleshooting

### Permission Denied
- Run with `sudo` for CPU power management access

### stress-ng not found
- Install stress-ng using your package manager

### Import errors
- Ensure all Python dependencies are installed
- Check you're using the correct Python environment

### No data in graphs
- Check the CSV file for error messages
- Ensure stress-ng completed successfully
- Verify CPU utilization measurements are non-zero

## Example Results

The graphs will show:
- Different test types have different performance characteristics
- Some tests scale linearly with CPU, others show diminishing returns
- Cache and memory tests may show different patterns than compute tests
- Efficiency often decreases at higher CPU utilization due to thermal/power limits