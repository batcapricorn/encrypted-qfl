import argparse
import os
import pickle
import yaml
import json

import flwr as fl
from flwr.common import Parameters, NDArrays
import torch
import tenseal as ts
import wandb

from utils.client import FlowerClient
from utils import data_setup, security
from utils.model import SimpleNet, simple_qnn_factory
from utils.common import choice_device, classes_string
from utils.fhe import (
    ndarrays_to_parameters,
    ndarray_to_bytes,
    bytes_to_ndarray,
    parameters_to_ndarrays_custom,
)

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
    choices=["fednn", "fedqnn"],
    default="fednn",
    help="Specify the model type: 'fednn' or 'fedqnn'.",
)

args = parser.parse_args()

# Load settings
with open("settings.yaml", "r") as file:
    config = yaml.safe_load(file)

# Initialize wandb
with open("tmp.json", "r") as f:
    wandb_config = json.load(f)

run_group = wandb_config.get("WANDB_RUN_GROUP")
print(f"Run group: {run_group}")

wandb.init(
    project="qfl-playground",
    config={
        "model": args.model,
        "fhe_enabled": args.he,
        "learning_rate": config["lr"],
        "batch_size": config["batch_size"],
        "number_clients": config["number_clients"],
        "dataset": config["dataset"],
        "rounds": config["rounds"],
        "group": run_group,
    },
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

DEVICE = torch.device(choice_device(config["device"]))
CLASSES = classes_string(config["dataset"])
central = SimpleNet(num_classes=len(CLASSES)).to(DEVICE)

context_client = None

net = None
if args.model == "fednn":
    net = SimpleNet(num_classes=len(CLASSES)).to(DEVICE)
elif args.model == "fedqnn":
    net = simple_qnn_factory(config["n_qubits"], config["n_layers"])

if args.he:
    print("Run with homomorphic encryption")
    if os.path.exists(config["secret_path"]):
        with open(config["secret_path"], "rb") as f:
            query = pickle.load(f)
        context_client = ts.context_from(query["contexte"])
    else:
        context_client = security.context()
        with open(config["secret_path"], "wb") as f:
            encode = pickle.dumps(
                {"contexte": context_client.serialize(save_secret_key=True)}
            )
            f.write(encode)
    secret_key = context_client.secret_key()
else:
    print("Run WITHOUT homomorphic encryption")

if os.path.exists(config["model_save"]):
    print("To get the checkpoint")
    checkpoint = torch.load(config["model_save"], map_location=DEVICE)[
        "model_state_dict"
    ]
    if args.he:
        print("to decrypt model")
        server_query, server_context = security.read_query(config["secret_path"])
        server_context = ts.context_from(server_context)
        for name in checkpoint:
            print(name)
            checkpoint[name] = torch.tensor(
                security.deserialized_layer(
                    name, server_query[name], server_context
                ).decrypt(secret_key)
            )
    net.load_state_dict(checkpoint)


def parameters_to_ndarrays(parameters: Parameters) -> NDArrays:
    with open(config["secret_path"], "rb") as f:
        query = pickle.load(f)

    context_client = ts.context_from(query["contexte"])
    return parameters_to_ndarrays_custom(parameters, context_client=context_client)


fl.common.parameter.ndarrays_to_parameters = ndarrays_to_parameters
fl.common.parameter.paramaters_to_ndarrays = parameters_to_ndarrays
fl.common.parameter.ndarray_to_bytes = ndarray_to_bytes
fl.common.parameter.bytes_to_ndarray = bytes_to_ndarray

client = FlowerClient(
    args.client_index,
    net,
    trainloader,
    valloader,
    device=DEVICE,
    batch_size=config["batch_size"],
    matrix_export=config["matrix_export"],
    roc_export=config["roc_export"],
    save_results=config["save_results"],
    yaml_path=config["yaml_path"],
    he=args.he,
    context_client=context_client,
    classes=CLASSES,
)

print("Starting flowerclient")
fl.client.start_numpy_client(server_address="127.0.0.1:8150", client=client)
