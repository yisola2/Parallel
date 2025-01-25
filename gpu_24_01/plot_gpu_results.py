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


def generate_gpu_plots(df):
    """Generate the required GPU plots."""
    os.makedirs("gpu_plots", exist_ok=True)

    # Filter data for a specific batch size (e.g., 32)
    bs32_data = df[df['batch_size'] == 32]

    # Plot 1: Time Components (Bar Graph)
    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    index = np.arange(len(bs32_data['num_devices'].unique()))

    plt.bar(index, bs32_data['loading_time'], bar_width, color='blue', label='Loading Time')
    plt.bar(index + bar_width, bs32_data['compute_time'], bar_width, color='orange', label='Compute + Communication Time')

    plt.xlabel('Number of GPUs', fontsize=14)
    plt.ylabel('Time (seconds)', fontsize=14)
    plt.title('Time Components vs Number of GPUs (Batch Size=32)', fontsize=16)
    plt.xticks(index + bar_width / 2, bs32_data['num_devices'].unique(), fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('gpu_plots/gpu_time_components_bar.png', dpi=100)
    plt.close()

    # Plot 2: Throughput (Line Plot)
    plt.figure(figsize=(10, 6))
    plt.plot(bs32_data['num_devices'], bs32_data['throughput'], marker='o', linestyle='-', color='green', label='Throughput')
    plt.xlabel('Number of GPUs', fontsize=14)
    plt.ylabel('Throughput (images/second)', fontsize=14)
    plt.title('Throughput vs Number of GPUs (Batch Size=32)', fontsize=16)
    plt.xticks(bs32_data['num_devices'].unique(), fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('gpu_plots/gpu_throughput_line.png', dpi=100)
    plt.close()


if __name__ == "__main__":
    try:
        data = load_data()
        print("Aggregated Data:")
        print(data)

        generate_gpu_plots(data)
        print("Plots saved to gpu_plots/ directory:")
        print("""
        - gpu_time_components_bar.png
        - gpu_throughput_line.png
        """)
    except Exception as e:
        print(f"Error: {e}")