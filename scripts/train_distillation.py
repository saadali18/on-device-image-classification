import os
import sys
import argparse
import yaml
import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


from datasets.cifar100 import get_cifar100_dataloaders
from models.resnet_cifar import build_resnet_cifar, count_parameters
from trainers.distillation_trainer import DistillationTrainer
from utils.checkpoint import load_checkpoint
from utils.seed import set_seed


def load_config(config_path: str):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_project_paths(config: dict) -> dict:
    path_fields = [
        ("dataset", "data_dir"),
        ("output", "checkpoint_dir"),
        ("output", "result_dir"),
        ("output", "plot_dir"),
        ("distillation", "teacher_checkpoint"),
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


def get_device(device_config: str):
    if device_config == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_config)


def ensure_output_dirs(config: dict):
    os.makedirs(config["dataset"]["data_dir"], exist_ok=True)
    os.makedirs(config["output"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["output"]["result_dir"], exist_ok=True)
    os.makedirs(config["output"]["plot_dir"], exist_ok=True)


def print_config_summary(config):
    print("=" * 80)
    print("DISTILLATION CONFIG SUMMARY")
    print("=" * 80)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Experiment name: {config['experiment_name']}")
    print(f"Seed: {config.get('seed', 42)}")
    print(f"Device config: {config.get('device', 'auto')}")

    print("\nDataset:")
    print(f"  Name: {config['dataset']['name']}")
    print(f"  Data dir: {config['dataset']['data_dir']}")
    print(f"  Num classes: {config['dataset']['num_classes']}")

    print("\nStudent:")
    print(f"  Name: {config['model']['name']}")
    print(f"  Width: {config['model']['width']}")

    print("\nDistillation:")
    print(f"  alpha (hard weight): {config['distillation']['alpha']}")
    print(f"  temperature:         {config['distillation']['temperature']}")
    print(f"  teacher model:       {config['distillation']['teacher_model']}")
    print(f"  teacher checkpoint:  {config['distillation']['teacher_checkpoint']}")

    print("\nTraining:")
    print(f"  Epochs: {config['training']['epochs']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  LR: {config['training']['learning_rate']} (milestones={config['scheduler']['milestones']}, gamma={config['scheduler']['gamma']})")

    print("\nOutput:")
    print(f"  Checkpoint dir: {config['output']['checkpoint_dir']}")
    print(f"  Result dir:     {config['output']['result_dir']}")
    print(f"  Plot dir:       {config['output']['plot_dir']}")
    print("=" * 80)


def load_teacher(config: dict, device: torch.device):
    distill_cfg = config["distillation"]

    teacher = build_resnet_cifar(
        model_name=distill_cfg["teacher_model"],
        num_classes=config["dataset"]["num_classes"],
        width=distill_cfg.get("teacher_width", 16),
    )
    teacher = teacher.to(device)

    ckpt_path = distill_cfg["teacher_checkpoint"]
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Teacher checkpoint not found at {ckpt_path}. "
            "Run scripts/train_teacher.py first."
        )

    checkpoint = load_checkpoint(ckpt_path, model=teacher, device=device)
    teacher_top1 = checkpoint.get("best_acc")

    print("=" * 80)
    print("TEACHER LOADED")
    print("=" * 80)
    print(f"From: {ckpt_path}")
    print(f"Teacher params: {count_parameters(teacher):,}")
    print(f"Teacher reported best Test Top-1 (from checkpoint): {teacher_top1}")
    print("=" * 80)

    return teacher


def main():
    parser = argparse.ArgumentParser(description="Train a distilled student.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a distillation YAML config "
        "(e.g. configs/distillation_resnet20_t4_a0p5.yaml).",
    )
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.abspath(os.path.join(os.getcwd(), config_path))

    config = load_config(config_path)
    config = resolve_project_paths(config)
    ensure_output_dirs(config)

    print_config_summary(config)

    set_seed(config.get("seed", 42))

    device = get_device(config.get("device", "auto"))
    print(f"Selected device: {device}")

    train_loader, test_loader = get_cifar100_dataloaders(
        data_dir=config["dataset"]["data_dir"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["dataset"]["num_workers"],
    )

    student = build_resnet_cifar(
        model_name=config["model"]["name"],
        num_classes=config["dataset"]["num_classes"],
        width=config["model"]["width"],
    ).to(device)
    num_parameters = count_parameters(student)

    print("=" * 80)
    print("STUDENT MODEL CHECK")
    print("=" * 80)
    print(f"Model: {config['model']['name']}  width={config['model']['width']}")
    print(f"Params: {num_parameters:,}")
    print("=" * 80)

    teacher = load_teacher(config, device)

    trainer = DistillationTrainer(
        student=student,
        teacher=teacher,
        train_loader=train_loader,
        test_loader=test_loader,
        config=config,
        device=device,
        num_parameters=num_parameters,
    )

    trainer.fit()


if __name__ == "__main__":
    main()
