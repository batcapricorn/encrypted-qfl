"""
Script to boot up flower server
"""

import argparse
import cProfile
import json
import os
import pstats
import time
import warnings
import yaml

import flwr as fl
from flwr.server import start_server
import numpy as np
import tenseal as ts
import torch
import torchvision
import wandb

from apps.server import (
    weighted_average,
    evaluate2_factory,
    get_on_fit_config_fn,
    fed_custom_factory,
)
from pytorch.model import SimpleNet, simple_qnn_factory, qcnn_factory
from security.fhe import read_query
from security.glue import combo_keys, ndarrays_to_parameters_custom
from utils.common import choice_device, classes_string, get_parameters2
from utils import data_setup

fl.common.GRPC_MAX_MESSAGE_LENGTH = 2000000000

parser = argparse.ArgumentParser(
    prog="FL server", description="Server that can be used for FL training"
)

parser.add_argument(
    "--he",
    action="store_true",
    help="if flag is set, parameters will be encrypted using FHE",
)

parser.add_argument(
    "--model",
    type=str,
    choices=["fednn", "fedqnn", "qcnn"],
    default="fednn",
    help="Specify the model type: 'fednn', 'fedqnn' or 'qcnn'.",
)
parser.add_argument(
    "--wandb_run_group", type=str, help="Run group for `wandb`", default=None
)

args = parser.parse_args()

# Load settings
with open("settings.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

if args.he:
    if os.path.exists(config["public_key_path"]) is False:
        combo_keys(
            client_path=config["private_key_path"],
            server_path=config["public_key_path"],
        )

# Initialize wandb
run_group = args.wandb_run_group
if run_group is None:
    job_id = os.getenv("SLURM_JOB_ID", wandb.util.generate_id())
    run_group = f"{'FHE' if args.he else 'Standard'}-{args.model}-{job_id}"

    wandb_config = {"WANDB_RUN_GROUP": run_group}

    with open("tmp.json", "w", encoding="utf-8") as f:
        json.dump(wandb_config, f)

wandb.init(
    project=config["wandb_project"],
    config={
        "model": args.model,
        "fhe_enabled": args.he,
        "learning_rate": config["lr"],
        "batch_size": config["batch_size"],
        "number_clients": config["number_clients"],
        "dataset": config["dataset"],
        "rounds": config["rounds"],
        "group": run_group,
        "participant": "server",
    },
    name="server",
)

SERVER_CONTEXT = None
if args.he:
    print("get public key : ", config["public_key_path"])
    _, SERVER_CONTEXT = read_query(config["public_key_path"])
    SERVER_CONTEXT = ts.context_from(SERVER_CONTEXT)

DEVICE = torch.device(choice_device(config["device"]))
CLASSES = classes_string(config["dataset"])
CENTRAL = None
if args.model == "fednn":
    CENTRAL = SimpleNet(num_classes=len(CLASSES)).to(DEVICE)
elif args.model == "fedqnn":
    SimpleQNN = simple_qnn_factory(config["n_qubits"], config["n_layers"])
    CENTRAL = SimpleQNN(num_classes=len(CLASSES)).to(DEVICE)
elif args.model == "qcnn":
    QNN = qcnn_factory()
    CENTRAL = QNN(num_classes=len(CLASSES)).to(DEVICE)


# Log number of trainable parameters
num_trainable_params = sum(p.numel() for p in CENTRAL.parameters() if p.requires_grad)
wandb.log({"trainable_parameters": num_trainable_params}, step=0)

trainloaders, valloaders, testloader = data_setup.load_datasets(
    num_clients=config["number_clients"],
    batch_size=config["batch_size"],
    resize=224,
    seed=config["seed"],
    num_workers=config["num_workers"],
    splitter=config["splitter"],
    dataset=config["dataset"],  # Use the specified dataset
    data_path=config["data_path"],
    data_path_val=None,
)  # Use the same path for validation data

FedCustom = fed_custom_factory(
    SERVER_CONTEXT,
    CENTRAL,
    config["lr"],
    config["model_checkpoint_path"],
    config["encrypted_model_checkpoint_path"],
)

strategy = FedCustom(
    fraction_fit=config["frac_fit"],
    fraction_evaluate=config["frac_eval"],
    min_fit_clients=config["min_fit_clients"],
    min_evaluate_clients=(
        config["min_eval_clients"]
        if config["min_eval_clients"]
        else config["number_clients"] // 2
    ),
    min_available_clients=config["min_avail_clients"],
    evaluate_metrics_aggregation_fn=weighted_average,
    initial_parameters=ndarrays_to_parameters_custom(get_parameters2(CENTRAL)),
    evaluate_fn=None if args.he else evaluate2_factory(CENTRAL, testloader, DEVICE),
    on_fit_config_fn=get_on_fit_config_fn(
        epoch=config["max_epochs"], batch_size=config["batch_size"]
    ),
    context_client=SERVER_CONTEXT,
)

if __name__ == "__main__":
    warnings.simplefilter("ignore")

    print("flwr", fl.__version__)
    print("numpy", np.__version__)
    print("torch", torch.__version__)
    print("torchvision", torchvision.__version__)
    print(f"Training on {DEVICE}")
    print("Starting flowerserver")

    with cProfile.Profile() as pr:
        start_time = time.time()
        start_server(
            server_address="0.0.0.0:8150",
            strategy=strategy,
            config=fl.server.ServerConfig(num_rounds=config["rounds"]),
            grpc_max_message_length=2000000000,
        )
        end_time = time.time() - start_time
        step = config["rounds"] + 1
        wandb.log({"total_training_time": end_time}, step=step)

        export_results_path = os.path.join(
            os.path.normpath(config["export_results_path"]), run_group
        )
        os.makedirs(export_results_path, exist_ok=True)
        dump_file = os.path.join(export_results_path, "cprofile_server.prof")

        with open(dump_file, "w", encoding="utf-8") as f:
            ps = pstats.Stats(pr, stream=f)
            ps.strip_dirs()
            ps.sort_stats("cumtime")
            ps.print_stats()

        wandb.save("settings.yaml")
        wandb.save("slurm_job.sh")
        wandb.save(f"results/{run_group}/cprofile_server.prof")
        wandb.save(f"results/{run_group}/cprofile_client0.prof")
        wandb.save(f"results/{run_group}/Accuracy_curves_Client 0.png")
        wandb.save(f"results/{run_group}/confusion_matrix_client0.png")
        wandb.save(f"results/{run_group}/Loss_curves_Client0.png")
        wandb.save(f"results/{run_group}/roc_client0.png")
