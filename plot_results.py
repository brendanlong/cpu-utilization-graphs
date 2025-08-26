#!/usr/bin/env python3
"""
Generate graphs from stress-ng test results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from typing import List

def load_results(csv_file: str) -> pd.DataFrame:
    """Load and clean test results from CSV."""
    df = pd.read_csv(csv_file)
    
    # Filter out any failed tests
    df_clean = df[df['error'].isna()].copy()
    
    # Convert numeric columns
    numeric_cols = ['cpu_target', 'actual_cpu_utilization', 'bogo_ops_per_sec', 'total_bogo_ops']
    for col in numeric_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    return df_clean


def plot_bogo_ops_vs_cpu(df: pd.DataFrame, output_prefix: str):
    """Plot Bogo OPS per second vs CPU utilization for each test type."""
    # Set up the plot style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Create figure with subplots
    test_types = df['test_type'].unique()
    n_types = len(test_types)
    
    # Figure 1: All test types on one plot
    plt.figure(figsize=(12, 8))
    
    for test_type in test_types:
        test_data = df[df['test_type'] == test_type]
        
        # Plot actual CPU utilization vs bogo ops
        plt.scatter(test_data['actual_cpu_utilization'], 
                   test_data['bogo_ops_per_sec'],
                   label=test_type, s=60, alpha=0.7)
        
        # Add trend line
        if len(test_data) > 1:
            z = np.polyfit(test_data['actual_cpu_utilization'], 
                          test_data['bogo_ops_per_sec'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(test_data['actual_cpu_utilization'].min(),
                                test_data['actual_cpu_utilization'].max(), 100)
            plt.plot(x_trend, p(x_trend), '--', alpha=0.5, linewidth=2)
    
    plt.xlabel('Actual CPU Utilization (%)', fontsize=12)
    plt.ylabel('Bogo Operations per Second', fontsize=12)
    plt.title('Performance vs CPU Utilization by Test Type', fontsize=14, pad=20)
    plt.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_all_tests.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Individual plots for each test type
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, test_type in enumerate(test_types):
        if idx < len(axes):
            ax = axes[idx]
            test_data = df[df['test_type'] == test_type]
            
            # Plot target vs actual CPU
            ax.scatter(test_data['cpu_target'], 
                      test_data['actual_cpu_utilization'],
                      label='Actual', s=60, alpha=0.7, color='blue')
            
            # Add ideal line
            ax.plot([0, 100], [0, 100], 'r--', alpha=0.5, label='Target')
            
            ax.set_xlabel('Target CPU (%)', fontsize=10)
            ax.set_ylabel('Actual CPU (%)', fontsize=10)
            ax.set_title(f'{test_type} - Target vs Actual CPU', fontsize=12)
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 105)
            ax.set_ylim(0, 105)
    
    # Hide unused subplots
    for idx in range(len(test_types), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_cpu_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 3: Efficiency plot (Bogo ops per CPU %)
    plt.figure(figsize=(12, 8))
    
    for test_type in test_types:
        test_data = df[df['test_type'] == test_type].copy()
        
        # Calculate efficiency
        test_data['efficiency'] = test_data['bogo_ops_per_sec'] / test_data['actual_cpu_utilization']
        
        plt.plot(test_data['actual_cpu_utilization'], 
                test_data['efficiency'],
                marker='o', label=test_type, markersize=8, linewidth=2)
    
    plt.xlabel('Actual CPU Utilization (%)', fontsize=12)
    plt.ylabel('Efficiency (Bogo OPS per CPU %)', fontsize=12)
    plt.title('Test Efficiency vs CPU Utilization', fontsize=14, pad=20)
    plt.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_efficiency.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 4: Heatmap of performance
    pivot_data = df.pivot_table(values='bogo_ops_per_sec', 
                                index='test_type', 
                                columns='cpu_target', 
                                aggfunc='mean')
    
    plt.figure(figsize=(14, 8))
    sns.heatmap(pivot_data, annot=True, fmt='.0f', cmap='YlOrRd', 
                cbar_kws={'label': 'Bogo OPS/s'})
    plt.title('Performance Heatmap: Test Type vs CPU Target', fontsize=14, pad=20)
    plt.xlabel('Target CPU (%)', fontsize=12)
    plt.ylabel('Test Type', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_summary_report(df: pd.DataFrame, output_file: str):
    """Generate a summary report of the test results."""
    with open(output_file, 'w') as f:
        f.write("# Stress-ng Test Results Summary\n\n")
        
        # Overall statistics
        f.write("## Overall Statistics\n")
        f.write(f"- Total tests run: {len(df)}\n")
        f.write(f"- Test types: {', '.join(df['test_type'].unique())}\n")
        f.write(f"- CPU targets tested: {sorted(df['cpu_target'].unique())}\n\n")
        
        # Performance by test type
        f.write("## Performance Summary by Test Type\n\n")
        summary = df.groupby('test_type').agg({
            'bogo_ops_per_sec': ['mean', 'std', 'min', 'max'],
            'actual_cpu_utilization': ['mean', 'std']
        }).round(2)
        f.write(summary.to_string())
        f.write("\n\n")
        
        # CPU accuracy analysis
        f.write("## CPU Target Accuracy\n\n")
        df['cpu_error'] = abs(df['actual_cpu_utilization'] - df['cpu_target'])
        accuracy = df.groupby('test_type')['cpu_error'].agg(['mean', 'std', 'max']).round(2)
        f.write("Average absolute error between target and actual CPU %:\n")
        f.write(accuracy.to_string())
        f.write("\n\n")
        
        # Best performers at different CPU levels
        f.write("## Best Performers by CPU Level\n\n")
        cpu_ranges = [(0, 25, "Low"), (25, 50, "Medium"), (50, 75, "High"), (75, 100, "Max")]
        
        for low, high, label in cpu_ranges:
            range_data = df[(df['actual_cpu_utilization'] >= low) & 
                           (df['actual_cpu_utilization'] <= high)]
            if not range_data.empty:
                best = range_data.loc[range_data['bogo_ops_per_sec'].idxmax()]
                f.write(f"{label} CPU ({low}-{high}%): {best['test_type']} - "
                       f"{best['bogo_ops_per_sec']:.1f} bogo ops/s "
                       f"@ {best['actual_cpu_utilization']:.1f}% CPU\n")


# Add numpy import for trend line
import numpy as np

def main():
    """Main function to generate all plots and reports."""
    if len(sys.argv) < 2:
        print("Usage: python plot_results.py <results.csv>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    if not os.path.exists(csv_file):
        print(f"Error: File {csv_file} not found")
        sys.exit(1)
    
    # Load results
    print(f"Loading results from {csv_file}...")
    df = load_results(csv_file)
    
    if df.empty:
        print("No valid test results found in CSV")
        sys.exit(1)
    
    print(f"Found {len(df)} valid test results")
    
    # Generate output prefix from input filename
    output_prefix = os.path.splitext(csv_file)[0]
    
    # Generate plots
    print("Generating plots...")
    plot_bogo_ops_vs_cpu(df, output_prefix)
    
    # Generate summary report
    print("Generating summary report...")
    generate_summary_report(df, f"{output_prefix}_summary.txt")
    
    print(f"\nGenerated files:")
    print(f"  - {output_prefix}_all_tests.png")
    print(f"  - {output_prefix}_cpu_accuracy.png")
    print(f"  - {output_prefix}_efficiency.png")
    print(f"  - {output_prefix}_heatmap.png")
    print(f"  - {output_prefix}_summary.txt")


if __name__ == '__main__':
    main()