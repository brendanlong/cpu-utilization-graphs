#!/usr/bin/env python3
"""
Analyze stress test results to create a linear CPU utilization scale based on Bogo operations.
Processes each test type (cpu, int64, double, matrixprod) separately.
"""

import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import warnings

warnings.filterwarnings("ignore")


def analyze_test_type(df, test_type):
    """
    Analyze a specific test type and generate reports, CSV, and visualizations.
    """
    # Filter for specific test type
    type_df = df.filter(pl.col("test_type") == test_type)

    if len(type_df) == 0:
        print(f"No data found for test type: {test_type}")
        return

    print(f"\n{'=' * 80}")
    print(f"ANALYSIS FOR TEST TYPE: {test_type.upper()}")
    print(f"{'=' * 80}")

    print(f"Total {test_type} data points: {len(type_df)}")
    print(f"Unique CPU targets: {len(type_df['cpu_target'].unique())}")
    print(f"Unique worker counts: {len(type_df['workers'].unique())}")

    # Split into two datasets
    # Dataset 1: Varying workers at 100% CPU target
    varying_workers_df = type_df.filter(pl.col("cpu_target") == 100)

    # Dataset 2: Varying CPU targets with max workers (24)
    varying_cpu_df = type_df.filter(pl.col("workers") == 24)

    print(f"\nDataset 1 (100% CPU, varying workers): {len(varying_workers_df)} points")
    print(f"Dataset 2 (24 workers, varying CPU%): {len(varying_cpu_df)} points")

    # Find the maximum bogo_ops_per_sec for reference (100% CPU with max workers)
    max_bogo_ops = type_df.filter(
        (pl.col("cpu_target") == 100) & (pl.col("workers") == 24)
    )["bogo_ops_per_sec"].max()

    if max_bogo_ops is None or max_bogo_ops == 0:
        print(f"Warning: No valid max bogo_ops found for {test_type}")
        return

    print(f"\nMax Bogo ops/sec (100% CPU, 24 workers): {max_bogo_ops:.2f}")

    # Calculate adjusted CPU utilization for both datasets
    varying_workers_df = varying_workers_df.with_columns(
        (pl.col("bogo_ops_per_sec") / max_bogo_ops * 100).alias(
            "adjusted_cpu_utilization"
        )
    )

    varying_cpu_df = varying_cpu_df.with_columns(
        (pl.col("bogo_ops_per_sec") / max_bogo_ops * 100).alias(
            "adjusted_cpu_utilization"
        )
    )

    # For the mapping, we'll use the varying CPU dataset (constant workers)
    mapping_df = varying_cpu_df.select(
        [
            "cpu_target",
            "actual_cpu_utilization",
            "adjusted_cpu_utilization",
            "bogo_ops_per_sec",
        ]
    ).sort("cpu_target")

    # Display sample of the mapping (first 10 and last 10)
    print(f"\nCPU Utilization Mapping Sample (24 workers):")
    print(
        f"{'Target CPU%':<12} {'Actual CPU%':<12} {'Adjusted CPU%':<15} {'Bogo ops/sec':<15}"
    )
    print("-" * 80)

    rows = list(mapping_df.iter_rows(named=True))
    sample_rows = rows[:5] + ["..."] + rows[-5:] if len(rows) > 10 else rows

    for row in sample_rows:
        if row == "...":
            print(f"{'...':<12} {'...':<12} {'...':<15} {'...':<15}")
        else:
            print(
                f"{row['cpu_target']:<12.0f} {row['actual_cpu_utilization']:<12.2f} "
                f"{row['adjusted_cpu_utilization']:<15.2f} {row['bogo_ops_per_sec']:<15.2f}"
            )

    # Create interpolation function for the mapping
    actual_cpu = mapping_df["actual_cpu_utilization"].to_numpy()
    adjusted_cpu = mapping_df["adjusted_cpu_utilization"].to_numpy()

    # Remove any duplicates or NaN values
    valid_mask = ~np.isnan(actual_cpu) & ~np.isnan(adjusted_cpu)
    actual_cpu = actual_cpu[valid_mask]
    adjusted_cpu = adjusted_cpu[valid_mask]

    if len(actual_cpu) < 2:
        print(f"Insufficient data for {test_type} analysis")
        return

    # Sort by actual_cpu for interpolation
    sort_idx = np.argsort(actual_cpu)
    actual_cpu = actual_cpu[sort_idx]
    adjusted_cpu = adjusted_cpu[sort_idx]

    # Remove duplicates in x values for interpolation
    unique_mask = np.append(True, np.diff(actual_cpu) > 0)
    actual_cpu = actual_cpu[unique_mask]
    adjusted_cpu = adjusted_cpu[unique_mask]

    # Create interpolation function
    f_actual_to_adjusted = interpolate.interp1d(
        actual_cpu,
        adjusted_cpu,
        kind="cubic" if len(actual_cpu) > 3 else "linear",
        fill_value="extrapolate",
    )

    # Create the visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Actual vs Adjusted CPU Utilization
    ax1 = axes[0, 0]
    ax1.scatter(actual_cpu, adjusted_cpu, alpha=0.6, s=50, label="Data points")
    if len(actual_cpu) > 1:
        x_smooth = np.linspace(actual_cpu.min(), actual_cpu.max(), 200)
        y_smooth = f_actual_to_adjusted(x_smooth)
        ax1.plot(x_smooth, y_smooth, "r-", alpha=0.8, label="Interpolation")
    ax1.plot([0, 100], [0, 100], "k--", alpha=0.3, label="Linear reference")
    ax1.set_xlabel("Actual CPU Utilization (%)")
    ax1.set_ylabel("Adjusted CPU Utilization (% of max Bogo ops)")
    ax1.set_title(f"CPU Utilization Mapping - {test_type.upper()} (24 workers)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    ax1.set_xticks(np.arange(0, 101, 10))
    ax1.set_yticks(np.arange(0, 101, 10))

    # Plot 2: CPU Target vs Both Utilizations
    ax2 = axes[0, 1]
    cpu_target = mapping_df["cpu_target"].to_numpy()
    ax2.plot(
        cpu_target,
        mapping_df["actual_cpu_utilization"].to_numpy(),
        "b-o",
        alpha=0.7,
        label="Actual CPU%",
        markersize=3,
    )
    ax2.plot(
        cpu_target,
        mapping_df["adjusted_cpu_utilization"].to_numpy(),
        "r-s",
        alpha=0.7,
        label="Adjusted CPU%",
        markersize=3,
    )
    ax2.plot(cpu_target, cpu_target, "k--", alpha=0.3, label="Target CPU%")
    ax2.set_xlabel("Target CPU (%)")
    ax2.set_ylabel("Utilization (%)")
    ax2.set_title(f"Target vs Actual vs Adjusted - {test_type.upper()}")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(0, 105)
    ax2.set_ylim(0, 105)
    ax2.set_xticks(np.arange(0, 101, 10))
    ax2.set_yticks(np.arange(0, 101, 10))

    # Plot 3: Bogo ops vs CPU Target
    ax3 = axes[1, 0]
    bogo_ops = mapping_df["bogo_ops_per_sec"].to_numpy()
    ax3.plot(cpu_target, bogo_ops, "g-^", alpha=0.7, markersize=3)
    ax3.set_xlabel("Target CPU (%)")
    ax3.set_ylabel("Bogo Operations per Second")
    ax3.set_title(f"Bogo Operations vs Target CPU - {test_type.upper()}")
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 105)
    ax3.set_xticks(np.arange(0, 101, 10))

    # Plot 4: Varying Workers Dataset (at 100% CPU)
    ax4 = axes[1, 1]
    if len(varying_workers_df) > 0:
        workers_data = varying_workers_df.sort("workers")
        workers = workers_data["workers"].to_numpy()
        workers_bogo = workers_data["bogo_ops_per_sec"].to_numpy()
        workers_adjusted = workers_data["adjusted_cpu_utilization"].to_numpy()

        ax4_twin = ax4.twinx()
        line1 = ax4.plot(
            workers, workers_bogo, "b-o", alpha=0.7, label="Bogo ops/sec", markersize=4
        )
        line2 = ax4_twin.plot(
            workers,
            workers_adjusted,
            "r-s",
            alpha=0.7,
            label="Adjusted CPU%",
            markersize=4,
        )

        ax4.set_xlabel("Number of Workers")
        ax4.set_ylabel("Bogo Operations per Second", color="b")
        ax4_twin.set_ylabel("Adjusted CPU Utilization (%)", color="r")
        ax4.set_title(f"Performance vs Workers (100% CPU) - {test_type.upper()}")
        ax4.grid(True, alpha=0.3)
        ax4.set_xticks(np.arange(0, 25, 4))
        ax4.tick_params(axis="y", labelcolor="b")
        ax4_twin.tick_params(axis="y", labelcolor="r")

        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc="upper left")
    else:
        ax4.text(
            0.5,
            0.5,
            "No data for varying workers at 100% CPU",
            ha="center",
            va="center",
            transform=ax4.transAxes,
        )
        ax4.set_title(f"Performance vs Workers - {test_type.upper()} (No Data)")

    plt.suptitle(
        f"CPU Utilization Analysis - {test_type.upper()} Test Method",
        fontsize=14,
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(
        f"cpu_utilization_analysis_{test_type}.png", dpi=150, bbox_inches="tight"
    )
    plt.close()

    print(f"\nVisualization saved to: cpu_utilization_analysis_{test_type}.png")

    # Export the mapping to CSV for further use
    if len(actual_cpu) > 0:
        mapping_export = pl.DataFrame(
            {
                "cpu_target": cpu_target,
                "actual_cpu_percent": mapping_df["actual_cpu_utilization"].to_numpy(),
                "adjusted_cpu_percent": mapping_df[
                    "adjusted_cpu_utilization"
                ].to_numpy(),
                "bogo_ops_per_sec": bogo_ops,
                "scaling_factor": mapping_df["adjusted_cpu_utilization"].to_numpy()
                / mapping_df["actual_cpu_utilization"].to_numpy(),
            }
        )

        csv_filename = f"cpu_utilization_mapping_{test_type}.csv"
        mapping_export.write_csv(csv_filename)
        print(f"Mapping exported to: {csv_filename}")

        # Calculate statistics
        scaling_factors = (
            mapping_df["adjusted_cpu_utilization"].to_numpy()
            / mapping_df["actual_cpu_utilization"].to_numpy()
        )
        scaling_factors = scaling_factors[~np.isnan(scaling_factors)]

        print(f"\nMapping Statistics for {test_type.upper()}:")
        print("-" * 40)
        print(f"Mean scaling factor: {scaling_factors.mean():.4f}")
        print(f"Std deviation of scaling factor: {scaling_factors.std():.4f}")
        print(f"Min scaling factor: {scaling_factors.min():.4f}")
        print(f"Max scaling factor: {scaling_factors.max():.4f}")

        # Show how non-linear the original mapping is
        linearity_error = np.abs(
            mapping_df["actual_cpu_utilization"].to_numpy()
            - mapping_df["adjusted_cpu_utilization"].to_numpy()
        )
        print(f"\nMean absolute difference from linear: {linearity_error.mean():.2f}%")
        print(f"Max absolute difference from linear: {linearity_error.max():.2f}%")


# Main execution
if __name__ == "__main__":
    # Read the CSV file
    df = pl.read_csv("stress_test_results_20250827_104138.csv")

    # Get unique test types
    test_types = df["test_type"].unique().sort()

    print(f"Found test types: {test_types.to_list()}")

    # Process each test type
    for test_type in test_types:
        analyze_test_type(df, test_type)
