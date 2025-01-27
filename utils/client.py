"""Module that defines Flower Client"""

import os
import time

import torch
import flwr as fl
import wandb

from utils.common import (
    save_graphs,
    get_parameters2,
    set_parameters,
    save_roc,
    save_matrix,
    eval_classification,
)
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
        matrix_export,
        roc_export,
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
        self.matrix_export = matrix_export
        self.roc_export = roc_export
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

        start_time = time.time()
        results = engine.train(
            self.net,
            self.trainloader,
            self.valloader,
            optimizer=optimizer,
            loss_fn=criterion,
            epochs=local_epochs,
            device=self.device,
        )
        end_time = time.time() - start_time
        wandb.log({"client_round_time": end_time})

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

        rec, prec, f1 = eval_classification(y_true, y_pred)
        metrics = {
            "accuracy": float(accuracy),
            "recall_score": rec,
            "precision_score": prec,
            "f1": f1,
        }

        if self.save_results:
            os.makedirs(self.save_results, exist_ok=True)
            if self.matrix_export:
                save_matrix(
                    y_true,
                    y_pred,
                    os.path.join(
                        self.save_results, f"confusion_matrix_client{self.cid}"
                    ),
                    self.classes,
                )
            if self.roc_export:
                save_roc(
                    y_true,
                    y_proba,
                    os.path.join(self.save_results, f"roc_client{self.cid}"),
                    len(self.classes),
                )

        return float(loss), len(self.valloader), metrics
