"""Module that defines Flower Client"""

import torch
import flwr as fl

from utils.common import save_graphs, get_parameters2, set_parameters
from utils import engine


class FlowerClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid,
        net,
        trainloader,
        valloader,
        device,
        batch_size,
        save_results,
        matrix_path,
        roc_path,
        yaml_path,
        he,
        classes,
        context_client,
    ):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.cid = cid
        self.device = device
        self.batch_size = batch_size
        self.save_results = save_results
        self.matrix_path = matrix_path
        self.roc_path = roc_path
        self.yaml_path = yaml_path
        self.he = he
        self.classes = classes
        self.context_client = context_client

    def get_parameters(self, config):
        print(f"[Client {self.cid}] get_parameters")
        return get_parameters2(self.net, self.context_client)

    def fit(self, parameters, config):
        server_round = config["server_round"]
        local_epochs = config["local_epochs"]
        lr = float(config["learning_rate"])

        print(f"[Client {self.cid}, round {server_round}] fit, config: {config}")

        set_parameters(self.net, parameters, self.context_client)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

        results = engine.train(
            self.net,
            self.trainloader,
            self.valloader,
            optimizer=optimizer,
            loss_fn=criterion,
            epochs=local_epochs,
            device=self.device,
        )

        if self.save_results:
            save_graphs(self.save_results, local_epochs, results, f"_Client {self.cid}")

        return get_parameters2(self.net, self.context_client), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        print(f"[Client {self.cid}] evaluate, config: {config}")
        set_parameters(self.net, parameters, self.context_client)

        loss, accuracy, y_pred, y_true, y_proba = engine.test(
            self.net,
            self.valloader,
            loss_fn=torch.nn.CrossEntropyLoss(),
            device=self.device,
        )

        # if self.save_results:
        #    os.makedirs(self.save_results, exist_ok=True)
        #    if self.matrix_path:
        #        save_matrix(y_true, y_pred, self.save_results + self.matrix_path, self.classes)
        #    if self.roc_path:
        #        save_roc(y_true, y_proba, self.save_results + self.roc_path, len(self.classes))

        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}
