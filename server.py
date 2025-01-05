import os
import tenseal as ts

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
)
from utils.server import (
    weighted_average,
    evaluate2_factory,
    get_on_fit_config_fn,
    fed_custom_factory,
)
from utils.model import SimpleNet

# Set up your variables directly
he = True
data_path = "data-tiny/"
dataset = "MRI"
yaml_path = "./results/FL/results.yml"
seed = 0
num_workers = 0
max_epochs = 10
batch_size = 10
splitter = 10
device = "gpu"
number_clients = 1
save_results = "results/FL/"
matrix_path = "confusion_matrix.png"
roc_path = "roc.png"
model_save = "MRI_FHE.pt"
min_fit_clients = 1
min_avail_clients = 1
min_eval_clients = 1
rounds = 3
frac_fit = 1.0
frac_eval = 0.5
lr = 1e-3
path_public_key = "server_key.pkl"

secret_path = "secret.pkl"  # private key of client
public_path = "server_key.pkl"  # publc key of client
path_crypted = "server.pkl"  # used to store encrypted checkpoints in aggreg_fit_checkpoint (as path_checkpoint)

if os.path.exists(secret_path):
    print("it exists")
    _, context_client = security.read_query(secret_path)

else:
    combo_keys(client_path=secret_path, server_path=public_path)

print("get public key : ", path_public_key)
_, server_context = security.read_query(path_public_key)
server_context = ts.context_from(server_context)
DEVICE = torch.device(choice_device(device))
CLASSES = classes_string(dataset)
central = SimpleNet(num_classes=len(CLASSES)).to(DEVICE)

trainloaders, valloaders, testloader = data_setup.load_datasets(
    num_clients=number_clients,
    batch_size=batch_size,
    resize=224,
    seed=seed,
    num_workers=num_workers,
    splitter=splitter,
    dataset=dataset,  # Use the specified dataset
    data_path=data_path,
    data_path_val=None,
)  # Use the same path for validation data

FedCustom = fed_custom_factory(server_context, central, lr, model_save, path_crypted)

strategy = FedCustom(
    fraction_fit=frac_fit,
    fraction_evaluate=frac_eval,
    min_fit_clients=min_fit_clients,
    min_evaluate_clients=min_eval_clients if min_eval_clients else number_clients // 2,
    min_available_clients=min_avail_clients,
    evaluate_metrics_aggregation_fn=weighted_average,
    initial_parameters=ndarrays_to_parameters_custom(get_parameters2(central)),
    evaluate_fn=None if he else evaluate2_factory(central, testloader, DEVICE),
    on_fit_config_fn=get_on_fit_config_fn(epoch=max_epochs, batch_size=batch_size),
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
    config=fl.server.ServerConfig(num_rounds=rounds),
)
