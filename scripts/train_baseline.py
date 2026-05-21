import os
import sys
import yaml
import torch


# ---------------------------------------------------------------------
# Project root setup
# ---------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


from datasets.cifar100 import get_cifar100_dataloaders
from models.resnet_cifar import build_resnet_cifar, count_parameters
from trainers.baseline_trainer import BaselineTrainer
from utils.seed import set_seed


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
def load_config(config_path: str):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def resolve_project_paths(config: dict) -> dict:
    """
    Convert relative paths from the YAML config into absolute paths based on
    PROJECT_ROOT.

    This prevents PyCharm from accidentally saving outputs inside scripts/
    when the working directory is set to scripts instead of the project root.
    """

    path_fields = [
        ("dataset", "data_dir"),
        ("output", "checkpoint_dir"),
        ("output", "result_dir"),
        ("output", "plot_dir"),
    ]

    for section, key in path_fields:
        if section not in config or key not in config[section]:
            continue

        path = config[section][key]

        if path is None:
            continue

        if not os.path.isabs(path):
            config[section][key] = os.path.abspath(
                os.path.join(PROJECT_ROOT, path)
            )

    return config


# ---------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------
def get_device(device_config: str):
    """
    Device selection:
    - cuda if available
    - mps for Apple Silicon Mac
    - cpu fallback
    """
    if device_config == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")

        if torch.backends.mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    return torch.device(device_config)


# ---------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------
def print_config_summary(config):
    print("=" * 80)
    print("CONFIG SUMMARY")
    print("=" * 80)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Experiment name: {config['experiment_name']}")
    print(f"Seed: {config.get('seed', 42)}")
    print(f"Device config: {config.get('device', 'auto')}")

    print("\nDataset:")
    print(f"  Name: {config['dataset']['name']}")
    print(f"  Data dir: {config['dataset']['data_dir']}")
    print(f"  Num classes: {config['dataset']['num_classes']}")
    print(f"  Num workers: {config['dataset']['num_workers']}")

    print("\nModel:")
    print(f"  Name: {config['model']['name']}")
    print(f"  Width: {config['model']['width']}")

    print("\nTraining:")
    print(f"  Epochs: {config['training']['epochs']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Optimizer: {config['training']['optimizer']}")
    print(f"  Learning rate: {config['training']['learning_rate']}")
    print(f"  Momentum: {config['training']['momentum']}")
    print(f"  Weight decay: {config['training']['weight_decay']}")

    print("\nScheduler:")
    print(f"  Name: {config['scheduler']['name']}")
    print(f"  Milestones: {config['scheduler']['milestones']}")
    print(f"  Gamma: {config['scheduler']['gamma']}")

    print("\nOutput:")
    print(f"  Checkpoint dir: {config['output']['checkpoint_dir']}")
    print(f"  Result dir: {config['output']['result_dir']}")
    print(f"  Plot dir: {config['output']['plot_dir']}")

    print("=" * 80)


def sanity_check_batch(train_loader, model, device, expected_num_classes: int):
    """
    Checks:
    - image shape
    - label range
    - model output shape
    - initial random accuracy should be low
    """
    print("=" * 80)
    print("SANITY CHECK")
    print("=" * 80)

    images, labels = next(iter(train_loader))

    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Labels dtype: {labels.dtype}")
    print(f"Labels min: {labels.min().item()}")
    print(f"Labels max: {labels.max().item()}")
    print(f"First 20 labels: {labels[:20].tolist()}")

    if labels.min().item() < 0:
        raise ValueError("Invalid labels: minimum label is below 0.")

    if labels.max().item() >= expected_num_classes:
        raise ValueError(
            f"Invalid labels: max label {labels.max().item()} "
            f">= expected num_classes {expected_num_classes}"
        )

    model.eval()

    with torch.no_grad():
        logits = model(images.to(device))

    print(f"Logits shape: {logits.shape}")
    print(f"Logits dtype: {logits.dtype}")
    print(f"Logits min: {logits.min().item():.6f}")
    print(f"Logits max: {logits.max().item():.6f}")

    expected_shape = (images.size(0), expected_num_classes)

    if tuple(logits.shape) != expected_shape:
        raise ValueError(
            f"Wrong logits shape. Expected {expected_shape}, got {tuple(logits.shape)}"
        )

    predictions = logits.argmax(dim=1).cpu()
    initial_correct = predictions.eq(labels).sum().item()
    initial_acc = 100.0 * initial_correct / labels.size(0)

    print(f"Initial random batch accuracy: {initial_acc:.2f}%")
    print("Expected initial accuracy for CIFAR-100 should be around ~1%, maybe a bit noisy.")
    print("=" * 80)

    if initial_acc > 20:
        print(
            "WARNING: Initial random accuracy is suspiciously high. "
            "Check dataset/model/labels carefully."
        )


def sanity_check_full_dataset(train_loader, test_loader):
    """
    Checks dataset size and target range if using torchvision CIFAR100.
    """
    print("=" * 80)
    print("DATASET CHECK")
    print("=" * 80)

    train_dataset = train_loader.dataset
    test_dataset = test_loader.dataset

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    if hasattr(train_dataset, "classes"):
        print(f"Number of classes: {len(train_dataset.classes)}")
        print(f"First 10 classes: {train_dataset.classes[:10]}")

    if hasattr(train_dataset, "targets"):
        print(f"Train target min: {min(train_dataset.targets)}")
        print(f"Train target max: {max(train_dataset.targets)}")
        print(f"First 20 train targets: {train_dataset.targets[:20]}")

    if hasattr(test_dataset, "targets"):
        print(f"Test target min: {min(test_dataset.targets)}")
        print(f"Test target max: {max(test_dataset.targets)}")
        print(f"First 20 test targets: {test_dataset.targets[:20]}")

    print("=" * 80)

    if len(train_dataset) != 50000:
        print("WARNING: CIFAR-100 train set should contain 50,000 images.")

    if len(test_dataset) != 10000:
        print("WARNING: CIFAR-100 test set should contain 10,000 images.")

    if hasattr(train_dataset, "targets"):
        if min(train_dataset.targets) != 0 or max(train_dataset.targets) != 99:
            print("WARNING: CIFAR-100 labels should range from 0 to 99.")


def ensure_output_dirs(config: dict):
    os.makedirs(config["dataset"]["data_dir"], exist_ok=True)
    os.makedirs(config["output"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["output"]["result_dir"], exist_ok=True)
    os.makedirs(config["output"]["plot_dir"], exist_ok=True)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    config_path = os.path.join(
        PROJECT_ROOT,
        "configs",
        "baseline_resnet20.yaml",
    )

    config = load_config(config_path)

    # Important fix:
    # Convert ./data, ./results, ./checkpoints, ./plots to absolute paths
    # under the project root.
    config = resolve_project_paths(config)

    ensure_output_dirs(config)

    print_config_summary(config)

    set_seed(config.get("seed", 42))

    device = get_device(config.get("device", "auto"))
    print(f"Selected device: {device}")

    # -----------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------
    train_loader, test_loader = get_cifar100_dataloaders(
        data_dir=config["dataset"]["data_dir"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["dataset"]["num_workers"],
    )

    sanity_check_full_dataset(train_loader, test_loader)

    # -----------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------
    model = build_resnet_cifar(
        model_name=config["model"]["name"],
        num_classes=config["dataset"]["num_classes"],
        width=config["model"]["width"],
    )

    model = model.to(device)

    num_parameters = count_parameters(model)

    print("=" * 80)
    print("MODEL CHECK")
    print("=" * 80)
    print(f"Model name: {config['model']['name']}")
    print(f"Model width: {config['model']['width']}")
    print(f"Number of parameters: {num_parameters:,}")
    print("=" * 80)

    sanity_check_batch(
        train_loader=train_loader,
        model=model,
        device=device,
        expected_num_classes=config["dataset"]["num_classes"],
    )

    # -----------------------------------------------------------------
    # Trainer
    # -----------------------------------------------------------------
    trainer = BaselineTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        config=config,
        device=device,
        num_parameters=num_parameters,
    )

    trainer.fit()


if __name__ == "__main__":
    main()