

# Evaluating Data Parallelism Efficiency in Machine Learning

## Project Description
This project evaluates the efficiency of data parallelism in machine learning workflows using CPU and GPU clusters. It measures the impact of different configurations (number of cores/GPUs, batch sizes) on training time and throughput using the DenseNet121 model and Imagenette dataset.

**Key Features**:
- Distributed training experiments for both CPU and GPU
- Automated experiment scripting with multiple configurations
- Performance metric visualization (throughput, time components)
- Memory-aware batch size handling for GPU constraints

## Folder Structure
```bash
.
├── cpu/
│   ├── cpu_plots/            # Generated CPU performance plots
│   ├── cpu_run/              # Raw experiment results (CSV files)
│   ├── plot_cpu_results.py   # CPU data analysis & visualization
│   ├── project_ex_1.py       # CPU distributed training code
│   └── run_experiments.sh    # CPU experiment runner
│
├── gpu/
│   ├── gpu_experiments/      # Raw GPU experiment results
│   ├── gpu_plots/            # Generated GPU performance plots
│   ├── gpu_plots_ex_2.py     # GPU data analysis & visualization
│   ├── project_ex_2.py       # GPU distributed training code
│   └── run_gpu_experiments.sh # GPU experiment runner
└── Report_Yassin_Es_Saim_Project.pdf  # Detailed project report
```
## Usage

### CPU Experiments
1. Run experiments:
```bash
cd cpu/
chmod +x run_experiments.sh
./run_experiments.sh
```

2. Generate plots:
```bash
python plot_cpu_results.py
```

### GPU Experiments
1. Run experiments:
```bash
cd gpu/
chmod +x run_gpu_experiments.sh
./run_gpu_experiments.sh
```

2. Generate plots:
```bash
python gpu_plots_ex_2.py
```

## Key Scripts
| File | Description |
|------|-------------|
| `run_experiments.sh` | Runs CPU experiments with batch size 32 and 1-8 cores |
| `run_gpu_experiments.sh` | Runs GPU experiments with batch sizes 16-128 and 1-3 GPUs |
| `plot_cpu_results.py` | Generates:<br>- Time Components Bar Chart<br>- Throughput Line Plot |
| `gpu_plots_ex_2.py` | Generates:<br>- Throughput vs GPU plots for different batch sizes<br>- Optimal GPU configuration chart |

## Results Analysis
The generated plots show:
- **CPU Scaling**: Linear throughput improvement up to 7 cores
- **GPU Scaling**: Super-linear throughput gains with multiple GPUs
- **Batch Size Impact**: Larger batches require more GPUs for optimal performance
- **Memory Constraints**: Batch size 128 requires ≥2 GPUs due to memory limits

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request
4. Maintain test coverage

## License
This project is licensed under the MIT License - see the [LICENSE](https://opensource.org/licenses/MIT) file for details.

## References
1. [NEF Cluster Documentation](https://wiki.inria.fr/ClustersSophia/Clusters_Home)
2. [PyTorch Distributed Training](https://pytorch.org/docs/stable/distributed.html)
3. [Imagenette Dataset](https://github.com/fastai/imagenette)

For detailed analysis and methodology, see the [Project Report](Report_Yassin_Es_Saim_Project.pdf).
```
