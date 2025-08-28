#!/usr/bin/env python3
"""
Analyze stress test results to create a linear CPU utilization scale based on Bogo operations.
Processes each test type (cpu, int64, double, matrixprod) separately.
"""

import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
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

    # Create logistic regression for the mapping
    actual_cpu = mapping_df["actual_cpu_utilization"].to_numpy()
    adjusted_cpu = mapping_df["adjusted_cpu_utilization"].to_numpy()

    # Remove any duplicates or NaN values
    valid_mask = ~np.isnan(actual_cpu) & ~np.isnan(adjusted_cpu)
    actual_cpu = actual_cpu[valid_mask]
    adjusted_cpu = adjusted_cpu[valid_mask]

    if len(actual_cpu) < 2:
        print(f"Insufficient data for {test_type} analysis")
        return

    # Sort by actual_cpu for better visualization
    sort_idx = np.argsort(actual_cpu)
    actual_cpu = actual_cpu[sort_idx]
    adjusted_cpu = adjusted_cpu[sort_idx]

    # Define logistic function
    def logistic(x, L, k, x0, b):
        """
        Generalized logistic function
        L = maximum value (asymptote)
        k = steepness of the curve
        x0 = x value of the sigmoid midpoint
        b = minimum value (baseline)
        """
        return b + L / (1 + np.exp(-k * (x - x0)))

    # Initial parameter guesses
    # L (max) should be around 100
    # b (min) should be around 0
    # x0 (midpoint) should be around 50
    # k (steepness) controls the slope
    initial_guess = [100, 0.1, 50, 0]

    # Bounds for parameters: L in [50, 150], k in [0.01, 1], x0 in [0, 100], b in [-20, 20]
    bounds = ([50, 0.01, 0, -20], [150, 1, 100, 20])

    try:
        # Fit the logistic curve
        params, _ = optimize.curve_fit(
            logistic,
            actual_cpu,
            adjusted_cpu,
            p0=initial_guess,
            bounds=bounds,
            maxfev=10000,
        )

        # Create prediction function using fitted parameters
        f_actual_to_adjusted = lambda x: logistic(x, *params)

        # Calculate R-squared for fit quality
        residuals = adjusted_cpu - f_actual_to_adjusted(actual_cpu)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((adjusted_cpu - np.mean(adjusted_cpu)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        print(f"\nLogistic regression parameters for {test_type.upper()}:")
        print(f"  L (max asymptote): {params[0]:.2f}")
        print(f"  k (steepness): {params[1]:.4f}")
        print(f"  x0 (midpoint): {params[2]:.2f}")
        print(f"  b (baseline): {params[3]:.2f}")
        print(f"  R-squared: {r_squared:.4f}")

    except Exception as e:
        print(
            f"Warning: Logistic regression failed for {test_type}, falling back to linear fit"
        )
        print(f"Error: {e}")
        # Fallback to simple linear fit
        coeffs = np.polyfit(actual_cpu, adjusted_cpu, 1)
        f_actual_to_adjusted = np.poly1d(coeffs)

    # Create the visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Actual vs Adjusted CPU Utilization
    ax1 = axes[0]
    ax1.scatter(actual_cpu, adjusted_cpu, alpha=0.6, s=50, label="Data points")
    if len(actual_cpu) > 1:
        x_smooth = np.linspace(actual_cpu.min(), actual_cpu.max(), 200)
        y_smooth = f_actual_to_adjusted(x_smooth)
        ax1.plot(x_smooth, y_smooth, "r-", alpha=0.8, label="Logistic fit")

        # Add piecewise linear regression
        if len(actual_cpu) > 3:  # Need at least 4 points for piecewise regression
            # Set breakpoint at exactly 50%
            breakpoint_cpu = 50.0

            def piecewise_linear_cpu(x, y0, k1, k2):
                """
                Piecewise linear function for CPU mapping
                y0 is the y value at the breakpoint
                k1 is the slope before breakpoint
                k2 is the slope after breakpoint
                """
                return np.piecewise(
                    x,
                    [x < breakpoint_cpu, x >= breakpoint_cpu],
                    [
                        lambda x: y0 + k1 * (x - breakpoint_cpu),
                        lambda x: y0 + k2 * (x - breakpoint_cpu),
                    ],
                )

            try:
                # Initial guess: estimate y at breakpoint and slopes
                idx_near_break = np.argmin(np.abs(actual_cpu - breakpoint_cpu))
                y_at_break = (
                    adjusted_cpu[idx_near_break]
                    if idx_near_break < len(adjusted_cpu)
                    else 50
                )

                # Estimate initial slopes
                before_break = actual_cpu <= breakpoint_cpu
                after_break = actual_cpu > breakpoint_cpu

                if np.sum(before_break) > 1 and np.sum(after_break) > 1:
                    # Fit separate linear regressions for initial slope estimates
                    k1_init = np.polyfit(
                        actual_cpu[before_break], adjusted_cpu[before_break], 1
                    )[0]
                    k2_init = np.polyfit(
                        actual_cpu[after_break], adjusted_cpu[after_break], 1
                    )[0]
                else:
                    # Fallback if not enough points
                    k1_init = 1.0
                    k2_init = 1.0

                # Fit the piecewise linear model
                params_pw, _ = optimize.curve_fit(
                    piecewise_linear_cpu,
                    actual_cpu,
                    adjusted_cpu,
                    p0=[y_at_break, k1_init, k2_init],
                    bounds=([0, -5, -5], [100, 5, 5]),
                )

                # Generate smooth curve for plotting
                y_piecewise = piecewise_linear_cpu(x_smooth, *params_pw)
                ax1.plot(
                    x_smooth,
                    y_piecewise,
                    "g--",
                    alpha=0.8,
                    label="Piecewise Linear (break: 50%)",
                    linewidth=2,
                )

                # Add vertical line at breakpoint
                ax1.axvline(
                    x=breakpoint_cpu,
                    color="gray",
                    linestyle=":",
                    alpha=0.3,
                    linewidth=1,
                )

                # Calculate R-squared for piecewise linear fit
                y_pred_pw = piecewise_linear_cpu(actual_cpu, *params_pw)
                residuals_pw = adjusted_cpu - y_pred_pw
                ss_res_pw = np.sum(residuals_pw**2)
                ss_tot_pw = np.sum((adjusted_cpu - np.mean(adjusted_cpu)) ** 2)
                r_squared_pw = 1 - (ss_res_pw / ss_tot_pw)

                # Add text annotation with fit parameters
                text_str = f"Breakpoint: {breakpoint_cpu:.0f}% CPU\n"
                text_str += f"Slope 1: {params_pw[1]:.2f}\n"
                text_str += f"Slope 2: {params_pw[2]:.2f}\n"
                text_str += f"R² = {r_squared_pw:.4f}"
                ax1.text(
                    0.95,
                    0.05,
                    text_str,
                    transform=ax1.transAxes,
                    fontsize=9,
                    verticalalignment="bottom",
                    horizontalalignment="right",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                )

            except Exception as e:
                print(f"Warning: Piecewise linear regression failed for Plot 1: {e}")

    ax1.plot([0, 100], [0, 100], "k--", alpha=0.3, label="Linear reference")
    ax1.set_xlabel("Reported CPU Utilization (%)")
    ax1.set_ylabel("Adjusted CPU Utilization (% of max Bogo ops)")
    ax1.set_title(f"CPU Utilization Mapping - {test_type.upper()} (24 workers)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    ax1.set_xticks(np.arange(0, 101, 10))
    ax1.set_yticks(np.arange(0, 101, 10))

    # Plot 2: Varying Workers Dataset (at 100% CPU)
    ax2 = axes[1]
    if len(varying_workers_df) > 0:
        workers_data = varying_workers_df.sort("workers")
        workers = workers_data["workers"].to_numpy()
        workers_adjusted = workers_data["adjusted_cpu_utilization"].to_numpy()

        line1 = ax2.plot(
            workers,
            workers_adjusted,
            "r-s",
            alpha=0.7,
            label="Adjusted CPU%",
            markersize=4,
        )

        # Piecewise linear regression
        if len(workers) > 3:  # Need at least 4 points for piecewise regression
            # Determine breakpoint from data
            max_workers = type_df["workers"].max()
            breakpoint = max_workers / 2.0

            print(
                f"  Max workers in dataset: {max_workers}, breakpoint at {breakpoint}"
            )

            def piecewise_linear(x, y0, k1, k2):
                """
                Piecewise linear function
                y0 is the y value at the breakpoint
                k1 is the slope before breakpoint
                k2 is the slope after breakpoint
                """
                return np.piecewise(
                    x,
                    [x < breakpoint, x >= breakpoint],
                    [
                        lambda x: y0 + k1 * (x - breakpoint),
                        lambda x: y0 + k2 * (x - breakpoint),
                    ],
                )

            try:
                # Initial guess: estimate y at breakpoint and slopes
                idx_near_break = np.argmin(np.abs(workers - breakpoint))
                y_at_break = (
                    workers_adjusted[idx_near_break]
                    if idx_near_break < len(workers_adjusted)
                    else 50
                )

                # Estimate initial slopes
                before_break = workers <= breakpoint
                after_break = workers > breakpoint

                if np.sum(before_break) > 1 and np.sum(after_break) > 1:
                    # Fit separate linear regressions for initial slope estimates
                    k1_init = np.polyfit(
                        workers[before_break], workers_adjusted[before_break], 1
                    )[0]
                    k2_init = np.polyfit(
                        workers[after_break], workers_adjusted[after_break], 1
                    )[0]
                else:
                    # Fallback if not enough points
                    k1_init = 4.0
                    k2_init = 0.5

                # Fit the piecewise linear model
                params, _ = optimize.curve_fit(
                    piecewise_linear,
                    workers,
                    workers_adjusted,
                    p0=[y_at_break, k1_init, k2_init],
                    bounds=([0, -10, -10], [100, 10, 10]),
                )

                # Generate smooth curve for plotting
                x_smooth = np.linspace(workers.min(), workers.max(), 200)
                y_piecewise = piecewise_linear(x_smooth, *params)

                line2 = ax2.plot(
                    x_smooth,
                    y_piecewise,
                    "b--",
                    alpha=0.8,
                    label=f"Piecewise Linear (breakpoint: {breakpoint:.0f})",
                    linewidth=2,
                )

                # Add vertical line at breakpoint
                ax2.axvline(x=breakpoint, color="gray", linestyle=":", alpha=0.5)

                # Calculate R-squared
                y_pred = piecewise_linear(workers, *params)
                residuals = workers_adjusted - y_pred
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((workers_adjusted - np.mean(workers_adjusted)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)

                # Add text annotation with fit parameters
                text_str = f"Breakpoint: {breakpoint:.0f} workers\n"
                text_str += f"Slope 1: {params[1]:.2f}\n"
                text_str += f"Slope 2: {params[2]:.2f}\n"
                text_str += f"R² = {r_squared:.4f}"
                ax2.text(
                    0.95,
                    0.05,
                    text_str,
                    transform=ax2.transAxes,
                    fontsize=9,
                    verticalalignment="bottom",
                    horizontalalignment="right",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                )

                lines = line1 + line2
            except Exception as e:
                print(f"Warning: Piecewise linear regression failed: {e}")
                lines = line1
        else:
            lines = line1

        ax2.set_xlabel("Number of Workers")
        ax2.set_ylabel("Adjusted CPU Utilization (%)")
        ax2.set_title(f"Performance vs Workers (100% CPU) - {test_type.upper()}")
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(np.arange(0, 25, 4))
        ax2.set_yticks(np.arange(0, 101, 10))
        ax2.tick_params(axis="y")

        # Combine legends
        labels = [line.get_label() for line in lines]
        ax2.legend(lines, labels, loc="upper left")
    else:
        ax2.text(
            0.5,
            0.5,
            "No data for varying workers at 100% CPU",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )
        ax2.set_title(f"Performance vs Workers - {test_type.upper()} (No Data)")

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
        cpu_target = mapping_df["cpu_target"].to_numpy()
        bogo_ops = mapping_df["bogo_ops_per_sec"].to_numpy()

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
