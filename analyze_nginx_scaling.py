#!/usr/bin/env python3
"""
Analyze Nginx benchmark results to create CPU utilization scale based on requests per second.
"""

import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
import sys

def analyze_nginx_results(csv_file):
    """
    Analyze Nginx benchmark results and generate visualization.
    """
    # Read the CSV file
    df = pl.read_csv(csv_file)
    
    print(f"{'=' * 80}")
    print(f"NGINX BENCHMARK CPU SCALING ANALYSIS")
    print(f"{'=' * 80}")
    
    print(f"\nTotal data points: {len(df)}")
    print(f"CPU cores range: {df['cores'].min()} - {df['cores'].max()}")
    
    # Find maximum requests per second (should be at max cores)
    max_requests = df['primary_result'].max()
    max_cores_row = df.filter(pl.col('primary_result') == max_requests)[0]
    
    print(f"\nMax Requests/sec: {max_requests:.2f} at {max_cores_row['cores'][0]} cores")
    print(f"Reported CPU% at max: {max_cores_row['actual_cpu_utilization'][0]:.2f}%")
    
    # Calculate adjusted CPU utilization based on requests per second
    df = df.with_columns(
        (pl.col('primary_result') / max_requests * 100).alias('adjusted_cpu_utilization')
    )
    
    # Extract data for plotting
    actual_cpu = df['actual_cpu_utilization'].to_numpy()
    adjusted_cpu = df['adjusted_cpu_utilization'].to_numpy()
    cores = df['cores'].to_numpy()
    
    # Sort by actual CPU for better visualization
    sort_idx = np.argsort(actual_cpu)
    actual_cpu = actual_cpu[sort_idx]
    adjusted_cpu = adjusted_cpu[sort_idx]
    cores_sorted = cores[sort_idx]
    
    # Display mapping sample
    print(f"\nCPU Utilization Mapping:")
    print(f"{'Cores':<8} {'Reported CPU%':<15} {'Adjusted CPU%':<15} {'Requests/sec':<15}")
    print("-" * 80)
    
    for i, row in enumerate(df.sort('cores').iter_rows(named=True)):
        if i < 5 or i >= len(df) - 5:
            print(f"{row['cores']:<8} {row['actual_cpu_utilization']:<15.2f} "
                  f"{row['adjusted_cpu_utilization']:<15.2f} {row['primary_result']:<15.2f}")
        elif i == 5:
            print(f"{'...':<8} {'...':<15} {'...':<15} {'...':<15}")
    
    # Piecewise linear regression with breakpoint at 50%
    breakpoint_cpu = 50.0
    
    def piecewise_linear(x, y0, k1, k2):
        """
        Piecewise linear function
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
    
    # Fit piecewise linear model
    try:
        # Initial guess
        idx_near_break = np.argmin(np.abs(actual_cpu - breakpoint_cpu))
        y_at_break = adjusted_cpu[idx_near_break] if idx_near_break < len(adjusted_cpu) else 50
        
        # Estimate initial slopes
        before_break = actual_cpu <= breakpoint_cpu
        after_break = actual_cpu > breakpoint_cpu
        
        if np.sum(before_break) > 1 and np.sum(after_break) > 1:
            k1_init = np.polyfit(actual_cpu[before_break], adjusted_cpu[before_break], 1)[0]
            k2_init = np.polyfit(actual_cpu[after_break], adjusted_cpu[after_break], 1)[0]
        else:
            k1_init = 1.0
            k2_init = 1.0
        
        # Fit the model
        params, _ = optimize.curve_fit(
            piecewise_linear,
            actual_cpu,
            adjusted_cpu,
            p0=[y_at_break, k1_init, k2_init],
            bounds=([0, -5, -5], [100, 5, 5]),
        )
        
        # Calculate R-squared
        y_pred = piecewise_linear(actual_cpu, *params)
        residuals = adjusted_cpu - y_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((adjusted_cpu - np.mean(adjusted_cpu)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        print(f"\nPiecewise Linear Regression Results:")
        print(f"  Breakpoint: {breakpoint_cpu:.0f}% CPU")
        print(f"  Y at breakpoint: {params[0]:.2f}%")
        print(f"  Slope before breakpoint: {params[1]:.3f}")
        print(f"  Slope after breakpoint: {params[2]:.3f}")
        print(f"  R-squared: {r_squared:.4f}")
        
        # Generate smooth curve for plotting
        x_smooth = np.linspace(actual_cpu.min(), actual_cpu.max(), 200)
        y_piecewise = piecewise_linear(x_smooth, *params)
        
        fit_success = True
    except Exception as e:
        print(f"\nWarning: Piecewise linear regression failed: {e}")
        print("Falling back to simple linear fit")
        coeffs = np.polyfit(actual_cpu, adjusted_cpu, 1)
        x_smooth = np.linspace(actual_cpu.min(), actual_cpu.max(), 200)
        y_piecewise = np.polyval(coeffs, x_smooth)
        params = None
        r_squared = None
        fit_success = False
    
    # Create the visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Create a colormap based on core count
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(cores)))
    scatter = ax.scatter(
        actual_cpu, 
        adjusted_cpu, 
        c=cores_sorted, 
        cmap='viridis',
        alpha=0.7, 
        s=100,
        edgecolors='black',
        linewidth=0.5,
        label='Data points',
        zorder=3
    )
    
    # Add colorbar for cores
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Number of Cores', rotation=270, labelpad=20)
    
    # Plot piecewise linear fit
    if fit_success:
        ax.plot(
            x_smooth,
            y_piecewise,
            'r-',
            alpha=0.8,
            linewidth=2.5,
            label=f'Piecewise Linear (break at {breakpoint_cpu:.0f}%)',
            zorder=2
        )
        
        # Add vertical line at breakpoint
        ax.axvline(
            x=breakpoint_cpu,
            color='gray',
            linestyle='--',
            alpha=0.4,
            linewidth=1,
            label=f'Breakpoint ({breakpoint_cpu:.0f}% CPU)'
        )
    else:
        ax.plot(
            x_smooth,
            y_piecewise,
            'r-',
            alpha=0.8,
            linewidth=2.5,
            label='Linear fit',
            zorder=2
        )
    
    # Add diagonal reference line
    ax.plot([0, 100], [0, 100], 'k--', alpha=0.3, linewidth=1, label='Linear reference (y=x)')
    
    # Annotate some key points
    for i, (cores_val, actual, adjusted) in enumerate(zip(cores, actual_cpu, adjusted_cpu)):
        if cores_val in [1, 6, 12, 18, 24]:  # Annotate specific core counts
            ax.annotate(
                f'{cores_val}c',
                xy=(actual, adjusted),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                alpha=0.7
            )
    
    # Formatting
    ax.set_xlabel('Reported CPU Utilization (%)', fontsize=12)
    ax.set_ylabel('Adjusted CPU Utilization (% of max requests/sec)', fontsize=12)
    ax.set_title('Nginx Benchmark: Adjusted vs Reported CPU Utilization', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 105)
    ax.set_xticks(np.arange(0, 101, 10))
    ax.set_yticks(np.arange(0, 101, 10))
    
    # Add text box with regression parameters
    if fit_success and params is not None:
        text_str = f'Piecewise Linear Regression:\n'
        text_str += f'Breakpoint: {breakpoint_cpu:.0f}%\n'
        text_str += f'Slope₁: {params[1]:.3f}\n'
        text_str += f'Slope₂: {params[2]:.3f}\n'
        text_str += f'R² = {r_squared:.4f}'
        
        ax.text(
            0.05,
            0.95,
            text_str,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8)
        )
    
    # Legend
    ax.legend(loc='lower right', framealpha=0.9)
    
    plt.tight_layout()
    output_file = 'nginx_cpu_scaling_analysis.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nVisualization saved to: {output_file}")
    
    # Export the mapping to CSV
    export_df = df.select([
        'cores',
        'actual_cpu_utilization',
        'adjusted_cpu_utilization',
        'primary_result',
        'avg_clock_speed_mhz',
        'max_clock_speed_mhz'
    ]).sort('cores')
    
    csv_output = 'nginx_cpu_scaling_mapping.csv'
    export_df.write_csv(csv_output)
    print(f"Mapping data exported to: {csv_output}")
    
    # Calculate and display statistics
    scaling_factors = adjusted_cpu / actual_cpu
    scaling_factors = scaling_factors[~np.isnan(scaling_factors)]
    
    print(f"\nScaling Statistics:")
    print("-" * 40)
    print(f"Mean scaling factor: {scaling_factors.mean():.4f}")
    print(f"Std deviation: {scaling_factors.std():.4f}")
    print(f"Min scaling factor: {scaling_factors.min():.4f}")
    print(f"Max scaling factor: {scaling_factors.max():.4f}")
    
    linearity_error = np.abs(actual_cpu - adjusted_cpu)
    print(f"\nDeviation from Linear:")
    print(f"Mean absolute difference: {linearity_error.mean():.2f}%")
    print(f"Max absolute difference: {linearity_error.max():.2f}%")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_nginx_scaling.py <csv_file>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    analyze_nginx_results(csv_file)