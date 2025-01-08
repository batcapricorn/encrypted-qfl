import argparse
import os
import tenseal as ts
import yaml

import numpy as np
import torchvision
import torch
import flwr as fl
from flwr.server import start_server
from utils.common import choice_device, classes_string, get_parameters2
from utils import security, data_setup
from utils.fhe import (
    combo_keys,
    ndarrays_to_parameters_custom,
    ndarrays_to_parameters,
    ndarray_to_bytes,
    bytes_to_ndarray,
)
from utils.server import (
    weighted_average,
    evaluate2_factory,
    get_on_fit_config_fn,
    fed_custom_factory,
)
from utils.model import SimpleNet

parser = argparse.ArgumentParser(
    prog="FL server", description="Server that can be used for FL training"
)

parser.add_argument(
    "--he",
    action="store_true",
    help="if flag is set, parameters will be encrypted using FHE",
)

args = parser.parse_args()

# Load settings
with open("settings.yaml", "r") as file:
    config = yaml.safe_load(file)

if os.path.exists(config["secret_path"]):
    print("it exists")
    _, context_client = security.read_query(config["secret_path"])

else:
    combo_keys(client_path=config["secret_path"], server_path=config["public_path"])


def parameters_to_ndarrays():
    raise


fl.common.parameter.ndarrays_to_parameters = ndarrays_to_parameters
fl.common.parameter.paramaters_to_ndarrays = parameters_to_ndarrays
fl.common.parameter.ndarray_to_bytes = ndarray_to_bytes
fl.common.parameter.bytes_to_ndarray = bytes_to_ndarray

print("get public key : ", config["path_public_key"])
_, server_context = security.read_query(config["path_public_key"])
server_context = ts.context_from(server_context)
DEVICE = torch.device(choice_device(config["device"]))
CLASSES = classes_string(config["dataset"])
central = SimpleNet(num_classes=len(CLASSES)).to(DEVICE)

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
    server_context, central, config["lr"], config["model_save"], config["path_crypted"]
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
    initial_parameters=ndarrays_to_parameters_custom(get_parameters2(central)),
    evaluate_fn=None if args.he else evaluate2_factory(central, testloader, DEVICE),
    on_fit_config_fn=get_on_fit_config_fn(
        epoch=config["max_epochs"], batch_size=config["batch_size"]
    ),
    context_client=server_context,
)

import warnings

warnings.simplefilter("ignore")

print("flwr", fl.__version__)
print("numpy", np.__version__)
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print(f"Training on {DEVICE}")
print("Starting flowerserver")

start_server(
    server_address="0.0.0.0:8150",
    strategy=strategy,
    config=fl.server.ServerConfig(num_rounds=config["rounds"]),
)
