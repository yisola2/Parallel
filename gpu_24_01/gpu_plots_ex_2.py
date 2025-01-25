import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os


def load_data(results_dir="gpu_experiments"):
    """Load and aggregate data from all GPU runs."""
    all_files = glob.glob(f"{results_dir}/bs*_gpus*/times_run*.csv")
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {results_dir}. Check the directory structure.")

    # Load all CSV files into a single DataFrame
    df_list = []
    for file in all_files:
        try:
            df = pd.read_csv(file)
            # Extract batch size and GPU count from the directory name
            dir_name = os.path.basename(os.path.dirname(file))
            df['batch_size'] = int(dir_name.split('_')[0].replace('bs', ''))
            df['num_devices'] = int(dir_name.split('_')[1].replace('gpus', ''))
            df_list.append(df)
        except Exception as e:
            print(f"Error loading {file}: {e}")

    if not df_list:
        raise ValueError("No valid data found in CSV files.")

    df = pd.concat(df_list, ignore_index=True)

    # Aggregate data by batch size and number of GPUs
    agg_df = df.groupby(['batch_size', 'num_devices']).agg({
        'loading_time': 'mean',
        'compute_time': 'mean'
    }).reset_index()

    # Calculate throughput
    agg_df['throughput'] = agg_df['batch_size'] / (agg_df['loading_time'] + agg_df['compute_time'])
    return agg_df


def generate_batch_size_plots(df):
    """Generate the required batch size impact plots."""
    os.makedirs("gpu_plots", exist_ok=True)

    # Plot 1: Throughput vs Number of GPUs for Batch Size 16
    plt.figure(figsize=(10, 6))
    bs16_data = df[df['batch_size'] == 16]
    plt.plot(bs16_data['num_devices'], bs16_data['throughput'], marker='o', linestyle='-', color='blue', label='Batch Size 16')
    plt.xlabel('Number of GPUs', fontsize=14)
    plt.ylabel('Throughput (images/second)', fontsize=14)
    plt.title('Throughput vs Number of GPUs (Batch Size=16)', fontsize=16)
    plt.xticks(bs16_data['num_devices'].unique(), fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('gpu_plots/throughput_bs16.png', dpi=100)
    plt.close()

    # Plot 2: Throughput vs Number of GPUs for Batch Size 64
    plt.figure(figsize=(10, 6))
    bs64_data = df[df['batch_size'] == 64]
    plt.plot(bs64_data['num_devices'], bs64_data['throughput'], marker='o', linestyle='-', color='green', label='Batch Size 64')
    plt.xlabel('Number of GPUs', fontsize=14)
    plt.ylabel('Throughput (images/second)', fontsize=14)
    plt.title('Throughput vs Number of GPUs (Batch Size=64)', fontsize=16)
    plt.xticks(bs64_data['num_devices'].unique(), fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('gpu_plots/throughput_bs64.png', dpi=100)
    plt.close()

    # Plot 3: Throughput vs Number of GPUs for Batch Size 128
    plt.figure(figsize=(10, 6))
    bs128_data = df[df['batch_size'] == 128]
    plt.plot(bs128_data['num_devices'], bs128_data['throughput'], marker='o', linestyle='-', color='red', label='Batch Size 128')
    plt.xlabel('Number of GPUs', fontsize=14)
    plt.ylabel('Throughput (images/second)', fontsize=14)
    plt.title('Throughput vs Number of GPUs (Batch Size=128)', fontsize=16)
    plt.xticks(bs128_data['num_devices'].unique(), fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('gpu_plots/throughput_bs128.png', dpi=100)
    plt.close()

    # Plot 4: Optimal Number of GPUs vs Batch Sizes
    optimal_gpus = df.loc[df.groupby('batch_size')['throughput'].idxmax()]
    plt.figure(figsize=(10, 6))
    plt.bar(optimal_gpus['batch_size'], optimal_gpus['num_devices'], color='purple')
    plt.xlabel('Batch Size', fontsize=14)
    plt.ylabel('Optimal Number of GPUs', fontsize=14)
    plt.title('Optimal Number of GPUs for Each Batch Size', fontsize=16)
    plt.xticks(optimal_gpus['batch_size'].unique(), fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('gpu_plots/optimal_gpus.png', dpi=100)
    plt.close()


if __name__ == "__main__":
    try:
        data = load_data()
        print("Aggregated Data:")
        print(data)

        generate_batch_size_plots(data)
        print("Plots saved to gpu_plots/ directory:")
        print("""
        - throughput_bs16.png
        - throughput_bs64.png
        - throughput_bs128.png
        - optimal_gpus.png
        """)
    except Exception as e:
        print(f"Error: {e}")