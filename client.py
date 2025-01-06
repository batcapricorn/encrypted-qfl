import os
import pickle

import flwr as fl
import torch
import tenseal as ts

from utils.client import FlowerClient
from utils import data_setup, security
from utils.model import SimpleNet
from utils.common import choice_device, classes_string

client_id = 0

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
matrix_export = True
roc_export = True
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

trainloader = trainloaders[client_id]
valloader = valloaders[client_id]

DEVICE = torch.device(choice_device(device))
CLASSES = classes_string(dataset)
central = SimpleNet(num_classes=len(CLASSES)).to(DEVICE)

context_client = None
net = SimpleNet(num_classes=len(CLASSES)).to(DEVICE)

if he:
    print("Run with homomorphic encryption")
    if os.path.exists(secret_path):
        with open(secret_path, "rb") as f:
            query = pickle.load(f)
        context_client = ts.context_from(query["contexte"])
    else:
        context_client = security.context()
        with open(secret_path, "wb") as f:
            encode = pickle.dumps(
                {"contexte": context_client.serialize(save_secret_key=True)}
            )
            f.write(encode)
    secret_key = context_client.secret_key()
else:
    print("Run WITHOUT homomorphic encryption")

if os.path.exists(model_save):
    print(" To get the checkpoint")
    checkpoint = torch.load(model_save, map_location=DEVICE)["model_state_dict"]
    if he:
        print("to decrypt model")
        server_query, server_context = security.read_query(secret_path)
        server_context = ts.context_from(server_context)
        for name in checkpoint:
            print(name)
            checkpoint[name] = torch.tensor(
                security.deserialized_layer(
                    name, server_query[name], server_context
                ).decrypt(secret_key)
            )
    net.load_state_dict(checkpoint)

client = FlowerClient(
    client_id,
    net,
    trainloader,
    valloader,
    device=DEVICE,
    batch_size=batch_size,
    matrix_export=matrix_export,
    roc_export=roc_export,
    save_results=save_results,
    yaml_path=yaml_path,
    he=he,
    context_client=context_client,
    classes=CLASSES,
)

print("Starting flowerclient")
fl.client.start_numpy_client(server_address="127.0.0.1:8150", client=client)
