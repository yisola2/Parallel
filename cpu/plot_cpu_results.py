import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os


def load_data(results_dir="cpu_run"):
    """Load and aggregate data from all runs."""
    all_files = glob.glob(f"{results_dir}/bs*_cores*/times_run*.csv")
    df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)

    # Aggregate data
    agg_df = df.groupby(['batch_size', 'num_devices']).agg({
        'loading_time': 'mean',
        'compute_time': 'mean'
    }).reset_index()

    agg_df['throughput'] = agg_df['batch_size'] / (agg_df['loading_time'] + agg_df['compute_time'])
    return agg_df


def generate_plots(df):
    """Generate the required CPU plots."""
    os.makedirs("cpu_plots", exist_ok=True)

    # Plot 1: Time Components (Bar Graph)
    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    index = np.arange(len(df['num_devices'].unique()))

    plt.bar(index, df['loading_time'], bar_width, color='red', label='Loading Time')
    plt.bar(index + bar_width, df['compute_time'], bar_width, color='blue', label='Compute Time')

    plt.xlabel('Number of CPU Cores', fontsize=14)
    plt.ylabel('Average Time (seconds)', fontsize=14)
    plt.title('Time Components vs CPU Cores (Batch Size=32)', fontsize=16)
    plt.xticks(index + bar_width / 2, df['num_devices'].unique(), fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('cpu_plots/time_components_bar.png', dpi=100)
    plt.close()

    # Plot 2: Throughput (Line Plot)
    plt.figure(figsize=(10, 6))
    plt.plot(df['num_devices'], df['throughput'], marker='o', linestyle='-', color='green', label='Throughput')
    plt.xlabel('Number of CPU Cores', fontsize=14)
    plt.ylabel('Throughput (images/second)', fontsize=14)
    plt.title('Throughput vs CPU Cores (Batch Size=32)', fontsize=16)
    plt.xticks(df['num_devices'].unique(), fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('cpu_plots/throughput_line.png', dpi=100)
    plt.close()


if __name__ == "__main__":
    data = load_data()
    print("Aggregated Data:")
    print(data)

    generate_plots(data)
    print("Plots saved to cpu_plots/ directory:")
    print("""
    - time_components_bar.png
    - throughput_line.png
    """)