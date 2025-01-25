import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import time
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import csv
import os

# Get config from environment variables
TOTAL_BATCH_SIZE = int(os.getenv("TOTAL_BATCH_SIZE", "32"))
RUN_ID = int(os.getenv("RUN_ID", "0"))  # Unique ID for each run


def run(rank, size):
    # Dataset transformations
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load dataset
    full_dataset = torchvision.datasets.Imagenette('/data/neo/user/chxu/', transform=transform)
    dataset_size = len(full_dataset)
    per_worker = dataset_size // size
    dataset = torch.utils.data.Subset(full_dataset, range(rank * per_worker, (rank + 1) * per_worker))

    # Create dataloader
    batch_size = TOTAL_BATCH_SIZE // size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize DenseNet121
    model = models.densenet121()
    model.classifier = nn.Linear(model.classifier.in_features, len(full_dataset.classes))
    ddp_model = DDP(model)  # CPU mode

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # Timing measurements
    load_start = time.time()
    inputs, labels = next(iter(dataloader))
    load_end = time.time()

    compute_start = time.time()
    optimizer.zero_grad()
    outputs = ddp_model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    compute_end = time.time()

    # Log results
    log_file = f"times_bs{TOTAL_BATCH_SIZE}_cores{size}.csv"
    if rank == 0 and not os.path.exists(log_file):
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["run_id", "num_devices", "batch_size", "loading_time", "compute_time", "rank"])

    dist.barrier()  # Sync before writing
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            RUN_ID,  # Unique run ID
            size,  # num_devices
            TOTAL_BATCH_SIZE,  # batch_size
            load_end - load_start,  # loading_time
            compute_end - compute_start,  # compute_time
            rank  # process rank
        ])

    dist.destroy_process_group()


if __name__ == "__main__":
    dist.init_process_group("gloo", init_method="env://")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    run(rank, world_size)