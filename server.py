import argparse
import os
import tenseal as ts
import yaml
import json
import warnings
import time
import cProfile
import pstats

import numpy as np
import torchvision
import torch
import flwr as fl
from flwr.server import start_server
import wandb
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
from utils.model import SimpleNet, simple_qnn_factory

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
    choices=["fednn", "fedqnn"],
    default="fednn",
    help="Specify the model type: 'fednn' or 'fedqnn'.",
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

# Initialize wandb
job_id = os.getenv("SLURM_JOB_ID", wandb.util.generate_id())
run_group = f"{"FHE" if args.he else "Standard"}-{args.model}-{job_id}"

wandb_config = {"WANDB_RUN_GROUP": run_group}

with open("tmp.json", "w") as f:
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

fl.common.parameter.ndarrays_to_parameters = ndarrays_to_parameters
fl.common.parameter.ndarray_to_bytes = ndarray_to_bytes
fl.common.parameter.bytes_to_ndarray = bytes_to_ndarray

print("get public key : ", config["path_public_key"])
_, server_context = security.read_query(config["path_public_key"])
server_context = ts.context_from(server_context)
DEVICE = torch.device(choice_device(config["device"]))
CLASSES = classes_string(config["dataset"])
central = None
if args.model == "fednn":
    central = SimpleNet(num_classes=len(CLASSES)).to(DEVICE)
elif args.model == "fedqnn":
    SimpleQNN = simple_qnn_factory(config["n_qubits"], config["n_layers"])
    central = SimpleQNN(num_classes=len(CLASSES)).to(DEVICE)


# Log number of trainable parameters
num_trainable_params = sum(p.numel() for p in central.parameters() if p.requires_grad)
wandb.log({"trainable_parameters": num_trainable_params})

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
        )
        end_time = time.time() - start_time
        wandb.log({"total_training_time": end_time})

        save_results = os.path.join(os.path.normpath(config["save_results"]), run_group)
        dump_file = os.path.join(save_results, f"cprofile_server.prof")

        with open(dump_file, "w") as f:
            ps = pstats.Stats(pr, stream=f)
            ps.strip_dirs()
            ps.sort_stats("cumtime")
            ps.print_stats()
