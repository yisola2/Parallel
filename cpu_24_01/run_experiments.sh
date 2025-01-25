#!/bin/bash

# Configuration
RESULTS_DIR="cpu_run"
BATCH_SIZES=(32)
CORE_COUNTS=(1 2 3 4 5 6 7 8)
NUM_RUNS=10  # Number of runs per configuration

# Clean previous results
rm -rf $RESULTS_DIR
mkdir -p $RESULTS_DIR

# Main experiment loop
for bs in "${BATCH_SIZES[@]}"; do
    for cores in "${CORE_COUNTS[@]}"; do
        # Create experiment directory
        EXP_DIR="$RESULTS_DIR/bs${bs}_cores${cores}"
        mkdir -p "$EXP_DIR"

        echo "=================================================="
        echo " Running experiment: Batch Size=$bs | CPU Cores=$cores"
        echo "=================================================="

        for run_id in $(seq 1 $NUM_RUNS); do
            echo "Run $run_id/$NUM_RUNS..."

            # Run distributed training
            TOTAL_BATCH_SIZE=$bs RUN_ID=$run_id torchrun \
                --nnodes=1 \
                --nproc-per-node=$cores \
                project_ex_1.py

            # Move and rename results
            if [ -f "times_bs${bs}_cores${cores}.csv" ]; then
                mv "times_bs${bs}_cores${cores}.csv" "$EXP_DIR/times_run${run_id}.csv"
            else
                echo "WARNING: No output file generated for bs=$bs cores=$cores run=$run_id"
            fi
        done
    done
done

echo "=================================================="
echo " All experiments completed! Results in: $RESULTS_DIR"
echo "=================================================="