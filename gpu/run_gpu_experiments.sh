#!/bin/bash

# Configuration
RESULTS_DIR="gpu_experiments"
BATCH_SIZES=(16  32 64 128)  # Batch sizes to test
GPU_COUNTS=(1 2 3)       # GPU counts to test
NUM_RUNS=10              # Number of runs per configuration

# Clean previous results
rm -rf $RESULTS_DIR
mkdir -p $RESULTS_DIR

# Main experiment loop
for bs in "${BATCH_SIZES[@]}"; do
    for gpus in "${GPU_COUNTS[@]}"; do
        # Skip invalid combinations (e.g., batch size 128 on 1 GPU)
        if [ $gpus -eq 1 ] && [ $bs -eq 128 ]; then
            echo "Skipping batch_size=$bs on 1 GPU (memory constraints)"
            continue
        fi

        # Create experiment directory
        EXP_DIR="$RESULTS_DIR/bs${bs}_gpus${gpus}"
        mkdir -p "$EXP_DIR"

        echo "=================================================="
        echo " Running experiment: Batch Size=$bs | GPUs=$gpus"
        echo "=================================================="

        for run_id in $(seq 1 $NUM_RUNS); do
            echo "Run $run_id/$NUM_RUNS..."

            # Run distributed training
            TOTAL_BATCH_SIZE=$bs RUN_ID=$run_id torchrun \
                --nnodes=1 \
                --nproc-per-node=$gpus \
                --standalone \
                project_ex_2.py

            # Move and rename results
            if [ -f "gpu_times_bs${bs}_gpus${gpus}.csv" ]; then
                mv "gpu_times_bs${bs}_gpus${gpus}.csv" "$EXP_DIR/times_run${run_id}.csv"
            else
                echo "WARNING: No output file generated for bs=$bs gpus=$gpus run=$run_id"
            fi
        done
    done
done

echo "=================================================="
echo " All experiments completed! Results in: $RESULTS_DIR"
echo "=================================================="