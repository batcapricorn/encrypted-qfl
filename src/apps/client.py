"""Module that defines Flower Client"""

import os
import time
from typing import Optional

import torch
import wandb
import flwr as fl
from flwr.common import GRPC_MAX_MESSAGE_LENGTH
from flwr.client.app import start_client
from flwr.client import Client
from flwr.client.numpy_client import NumPyClient
from flwr.client.numpy_client import has_evaluate as numpyclient_has_evaluate
from flwr.client.numpy_client import has_fit as numpyclient_has_fit
from flwr.client.numpy_client import (
    has_get_parameters as numpyclient_has_get_parameters,
)
from flwr.client.numpy_client import (
    has_get_properties as numpyclient_has_get_properties,
)
from flwr.common.typing import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    NDArrays,
    Status,
)

from utils.common import (
    save_graphs,
    get_parameters2,
    set_parameters,
    save_roc,
    save_matrix,
    eval_classification,
)
from security.glue import (
    ndarrays_to_parameters,
    parameters_to_ndarrays_custom,
)
from pytorch import engine


EXCEPTION_MESSAGE_WRONG_RETURN_TYPE_FIT = """
NumPyClient.fit did not return a tuple with 3 elements.
The returned values should have the following type signature:

    Tuple[NDArrays, int, Dict[str, Scalar]]

Example
-------

    model.get_weights(), 10, {"accuracy": 0.95}

"""

EXCEPTION_MESSAGE_WRONG_RETURN_TYPE_EVALUATE = """
NumPyClient.evaluate did not return a tuple with 3 elements.
The returned values should have the following type signature:

    Tuple[float, int, Dict[str, Scalar]]

Example
-------

    0.5, 10, {"accuracy": 0.95}

"""


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
        wandb.log({"client_round_time": end_time}, step=server_round)

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


def _constructor(self: Client, numpy_client: NumPyClient) -> None:
    self.numpy_client = numpy_client


def _get_properties(self: Client, ins: GetPropertiesIns) -> GetPropertiesRes:
    """Return the current client properties."""
    properties = self.numpy_client.get_properties(config=ins.config)  # type: ignore
    return GetPropertiesRes(
        status=Status(code=Code.OK, message="Success"),
        properties=properties,
    )


def _get_parameters(self: Client, ins: GetParametersIns) -> GetParametersRes:
    """Return the current local model parameters."""
    parameters = self.numpy_client.get_parameters(config=ins.config)  # type: ignore
    parameters_proto = ndarrays_to_parameters(parameters)
    return GetParametersRes(
        status=Status(code=Code.OK, message="Success"), parameters=parameters_proto
    )


def _fit(self: Client, ins: FitIns) -> FitRes:
    """Refine the provided parameters using the locally held dataset."""
    # Deconstruct FitIns
    parameters: NDArrays = parameters_to_ndarrays_custom(
        ins.parameters, self.numpy_client.context_client
    )

    # Train
    results = self.numpy_client.fit(parameters, ins.config)  # type: ignore
    if not (
        len(results) == 3
        and isinstance(results[0], list)
        and isinstance(results[1], int)
        and isinstance(results[2], dict)
    ):
        raise Exception(EXCEPTION_MESSAGE_WRONG_RETURN_TYPE_FIT)

    # Return FitRes
    parameters_prime, num_examples, metrics = results
    parameters_prime_proto = ndarrays_to_parameters(parameters_prime)
    return FitRes(
        status=Status(code=Code.OK, message="Success"),
        parameters=parameters_prime_proto,
        num_examples=num_examples,
        metrics=metrics,
    )


def _evaluate(self: Client, ins: EvaluateIns) -> EvaluateRes:
    """Evaluate the provided parameters using the locally held dataset."""
    parameters: NDArrays = parameters_to_ndarrays_custom(
        ins.parameters, self.numpy_client.context_client
    )

    results = self.numpy_client.evaluate(parameters, ins.config)  # type: ignore
    if not (
        len(results) == 3
        and isinstance(results[0], float)
        and isinstance(results[1], int)
        and isinstance(results[2], dict)
    ):
        raise Exception(EXCEPTION_MESSAGE_WRONG_RETURN_TYPE_EVALUATE)

    # Return EvaluateRes
    loss, num_examples, metrics = results
    return EvaluateRes(
        status=Status(code=Code.OK, message="Success"),
        loss=loss,
        num_examples=num_examples,
        metrics=metrics,
    )


def _wrap_numpy_client(client: NumPyClient) -> Client:
    member_dict: Dict[str, Callable] = {  # type: ignore
        "__init__": _constructor,
    }

    # Add wrapper type methods (if overridden)

    if numpyclient_has_get_properties(client=client):
        member_dict["get_properties"] = _get_properties

    if numpyclient_has_get_parameters(client=client):
        member_dict["get_parameters"] = _get_parameters

    if numpyclient_has_fit(client=client):
        member_dict["fit"] = _fit

    if numpyclient_has_evaluate(client=client):
        member_dict["evaluate"] = _evaluate

    # Create wrapper class
    wrapper_class = type("NumPyClientWrapper", (Client,), member_dict)

    # Create and return an instance of the newly created class
    return wrapper_class(numpy_client=client)  # type: ignore


def start_numpy_client(
    *,
    server_address: str,
    client: NumPyClient,
    grpc_max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,
    root_certificates: Optional[bytes] = None,
    rest: bool = False,  # Deprecated in favor of `transport`
    transport: Optional[str] = None,
) -> None:
    # Start
    start_client(
        server_address=server_address,
        client=_wrap_numpy_client(client=client),
        grpc_max_message_length=grpc_max_message_length,
        root_certificates=root_certificates,
        rest=rest,
        transport=transport,
    )
