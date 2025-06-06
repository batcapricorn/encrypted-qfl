"""
Script to boot up client flower server
"""

import argparse
import cProfile
import json
import os
import pickle
import pstats
import yaml

import flwr as fl
import tenseal as ts
import torch
import wandb

from apps.client import FlowerClient, start_numpy_client
from pytorch.model import (
    SimpleNet,
    SimpleResNet18,
    simple_qnn_factory,
    qcnn_factory,
    simple_resnet18_qnn_factory,
    resnet18_qcnn_factory,
)
from security import fhe
from utils import data_setup
from utils.common import choice_device, classes_string

fl.common.GRPC_MAX_MESSAGE_LENGTH = 2000000000

parser = argparse.ArgumentParser(
    prog="FL client", description="Client server that can be used for FL training"
)

parser.add_argument(
    "--client_index",
    type=int,
    help="index for client to load data partition",
)
parser.add_argument(
    "--he",
    action="store_true",
    help="if flag is set, parameters will be encrypted using FHE",
)

parser.add_argument(
    "--model",
    type=str,
    choices=["fednn", "fedqnn", "qcnn", "resnet18", "resnet18-qnn", "resnet18-qcnn"],
    default="fednn",
    help="Specify the model type (e.g. 'fednn').",
)
parser.add_argument(
    "--wandb_run_group", type=str, help="Run group for `wandb`", default=None
)

args = parser.parse_args()

# Load settings
with open("settings.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

# Initialize wandb
run_group = args.wandb_run_group
if run_group is None:
    with open("tmp.json", "r", encoding="utf-8") as f:
        wandb_config = json.load(f)

    run_group = wandb_config.get("WANDB_RUN_GROUP")
    print(f"Run group: {run_group}")

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
        "participant": f"client{args.client_index}",
    },
    name=f"client{args.client_index}",
)

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

trainloader = trainloaders[args.client_index]
valloader = valloaders[args.client_index]

EXPORT_RESULTS_PATH = os.path.join(
    os.path.normpath(config["export_results_path"]), run_group
)
os.makedirs(EXPORT_RESULTS_PATH, exist_ok=True)

DEVICE = torch.device(choice_device(config["device"]))
CLASSES = classes_string(config["dataset"])

CONTEXT_CLIENT = None

NET = None
if args.model == "fednn":
    NET = SimpleNet(num_classes=len(CLASSES)).to(DEVICE)
elif args.model == "fedqnn":
    SimpleQNN = simple_qnn_factory(config["n_qubits"], config["n_layers"])
    NET = SimpleQNN(num_classes=len(CLASSES)).to(DEVICE)
elif args.model == "qcnn":
    QNN = qcnn_factory()
    NET = QNN(num_classes=len(CLASSES)).to(DEVICE)
elif args.model == "resnet18":
    NET = SimpleResNet18(num_classes=len(CLASSES)).to(DEVICE)
elif args.model == "resnet18-qnn":
    SimpleResNet18QNN = simple_resnet18_qnn_factory(
        config["n_qubits"], config["n_layers"]
    )
    NET = SimpleResNet18QNN(num_classes=len(CLASSES)).to(DEVICE)
elif args.model == "resnet18-qcnn":
    ResNet18QCNN = resnet18_qcnn_factory()
    NET = ResNet18QCNN(num_classes=len(CLASSES)).to(DEVICE)


if args.he:
    PRIVATE_KEY_PATH = os.path.join(EXPORT_RESULTS_PATH, config["private_key_path"])
    print("Run with homomorphic encryption")
    if os.path.exists(PRIVATE_KEY_PATH):
        with open(PRIVATE_KEY_PATH, "rb") as f:
            query = pickle.load(f)
        CONTEXT_CLIENT = ts.context_from(query["contexte"])
    else:
        CONTEXT_CLIENT = fhe.context()
        with open(PRIVATE_KEY_PATH, "wb") as f:
            encode = pickle.dumps(
                {"contexte": CONTEXT_CLIENT.serialize(save_secret_key=True)}
            )
            f.write(encode)
    secret_key = CONTEXT_CLIENT.secret_key()
else:
    print("Run WITHOUT homomorphic encryption")

checkpoint_path = os.path.join(EXPORT_RESULTS_PATH, config["model_checkpoint_path"])
if os.path.exists(checkpoint_path):
    print("To get the checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)["model_state_dict"]
    if args.he:
        print("to decrypt model")
        server_query, server_context = fhe.read_query(PRIVATE_KEY_PATH)
        server_context = ts.context_from(server_context)
        for name in checkpoint:
            print(name)
            checkpoint[name] = torch.tensor(
                fhe.deserialized_layer(
                    name, server_query[name], server_context
                ).decrypt(secret_key)
            )
    NET.load_state_dict(checkpoint)

client = FlowerClient(
    args.client_index,
    NET,
    trainloader,
    valloader,
    device=DEVICE,
    batch_size=config["batch_size"],
    matrix_export=config["matrix_export"],
    roc_export=config["roc_export"],
    export_results_path=EXPORT_RESULTS_PATH,
    he=args.he,
    context_client=CONTEXT_CLIENT,
    layers_to_encrypt=config["layers_to_encrypt"],
    classes=CLASSES,
)

if __name__ == "__main__":
    print("Starting flowerclient")
    with cProfile.Profile() as pr:
        start_numpy_client(
            server_address="127.0.0.1:8150",
            client=client,
            grpc_max_message_length=2000000000,
        )
        dump_file = os.path.join(
            EXPORT_RESULTS_PATH, f"cprofile_client{args.client_index}.prof"
        )

        with open(dump_file, "w", encoding="utf-8") as f:
            ps = pstats.Stats(pr, stream=f)
            ps.strip_dirs()
            ps.sort_stats("cumtime")
            ps.print_stats()
