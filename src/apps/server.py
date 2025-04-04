"""Helper function to boot up server"""

from collections import OrderedDict
from logging import WARNING
import pickle
import time
from typing import List, Tuple, Dict, Optional, Callable, Union

import flwr as fl
from flwr.common import (
    Metrics,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    Scalar,
    logger,
    Parameters,
    NDArrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from flwr.server.strategy.aggregate import weighted_loss_avg
import numpy as np
import torch
import wandb

from utils.common import set_parameters
from pytorch import engine
from security import fhe
from security.glue import (
    parameters_to_ndarrays_custom,
    ndarrays_to_parameters_custom,
)


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Callback to aggregate metrics on server level"""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    recalls = [num_examples * m["recall_score"] for num_examples, m in metrics]
    precisions = [num_examples * m["precision_score"] for num_examples, m in metrics]
    f1s = [num_examples * m["f1"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    metrics = {
        "accuracy": sum(accuracies) / sum(examples),
        "recalls": sum(recalls) / sum(examples),
        "precisions": sum(precisions) / sum(examples),
        "f1s": sum(f1s) / sum(examples),
    }
    return metrics


def evaluate2_factory(central, testloader, device):
    """Factory to create evaluation function for server"""

    def evaluate2(
        server_round: int, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Calculate model performance on server level"""
        set_parameters(central, parameters)
        loss, accuracy, _, _, _ = engine.test(
            central, testloader, loss_fn=torch.nn.CrossEntropyLoss(), device=device
        )
        print(f"Server-side evaluation loss {loss} / accuracy {accuracy}")
        return loss, {"accuracy": accuracy}

    return evaluate2


def get_on_fit_config_fn(
    epoch=2, lr=0.001, batch_size=32
) -> Callable[[int], Dict[str, str]]:
    """Factory to create fit configuration"""

    def fit_config(server_round: int) -> Dict[str, str]:
        "Returns fit config (dict containing learning rate etc.)"
        config = {
            "learning_rate": str(lr),
            "batch_size": str(batch_size),
            "server_round": server_round,
            "local_epochs": epoch,
        }
        return config

    return fit_config


def aggreg_fit_checkpoint_factory(server_context):
    """Factory to create checkpoint function"""

    def aggreg_fit_checkpoint(
        server_round,
        aggregated_parameters,
        central_model,
        path_checkpoint,
        context_client=None,
        server_path="",
    ):
        """Store model checkpoint"""
        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated_parameters...")
            trainable_keys = [
                k for k, v in central_model.named_parameters() if v.requires_grad
            ]
            print(trainable_keys)
            aggregated_ndarrays: List[np.ndarray] = parameters_to_ndarrays_custom(
                aggregated_parameters, context_client
            )
            print(f"Number of aggregated parameters: {len(aggregated_ndarrays)}")
            if context_client:
                server_response = {"contexte": server_context.serialize()}
                for i, key in enumerate(trainable_keys):
                    try:
                        server_response[key] = aggregated_ndarrays[i].serialize()
                    except:
                        server_response[key] = aggregated_ndarrays[i]
                fhe.write_query(server_path, server_response)
            else:
                params_dict = zip(trainable_keys, aggregated_ndarrays)
                state_dict = OrderedDict(
                    {k: torch.from_numpy(np.copy(v)) for k, v in params_dict}
                )
                central_model.load_state_dict(state_dict, strict=False)
                if path_checkpoint:
                    torch.save(
                        {
                            "model_state_dict": central_model.state_dict(),
                        },
                        path_checkpoint,
                    )

    return aggreg_fit_checkpoint


def fed_custom_factory(
    server_context, central, lr, model_checkpoint_path, encrypted_model_checkpoint_path
):
    """Factory to create Flower Strategy class used for server instance

    :param server_context: Server FHE context (public key of client)
    :type server_context: ts.Context
    :param central: Model
    :type central: torch.Module
    :param lr: learning rate
    :type lr: float
    :param model_checkpoint_path: Path to store checkpoint if unencrypted
    :type model_checkpoint_path: str
    :param encrypted_model_checkpoint_path: Path to store checkpoint if encrypted
    :type encrypted_model_checkpoint_path: str
    :return: Flower Strategy class
    :rtype: fl.server.strategy.Strategy
    """
    aggreg_fit_checkpoint = aggreg_fit_checkpoint_factory(server_context)

    # A Strategy from scratch with the same sampling of the clients as it is in FedAvg
    # and then change the configuration dictionary
    class FedCustom(fl.server.strategy.Strategy):
        """Customized Flower Server Strategy to incorporate FHE"""

        def __init__(
            self,
            fraction_fit: float = 1.0,
            fraction_evaluate: float = 1.0,
            min_fit_clients: int = 2,
            min_evaluate_clients: int = 2,
            min_available_clients: int = 2,
            evaluate_fn: Optional[
                Callable[
                    [int, NDArrays, Dict[str, Scalar]],
                    Optional[Tuple[float, Dict[str, Scalar]]],
                ]
            ] = None,
            on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            accept_failures: bool = True,
            initial_parameters: Optional[Parameters] = None,
            fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
            evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
            context_client=None,
        ) -> None:
            super().__init__()
            self.fraction_fit = fraction_fit
            self.fraction_evaluate = fraction_evaluate
            self.min_fit_clients = min_fit_clients
            self.min_evaluate_clients = min_evaluate_clients
            self.min_available_clients = min_available_clients
            self.evaluate_fn = evaluate_fn
            self.on_fit_config_fn = on_fit_config_fn
            self.on_evaluate_config_fn = (on_evaluate_config_fn,)
            self.accept_failures = accept_failures
            self.initial_parameters = initial_parameters
            self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
            self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
            self.context_client = context_client

            self.bytes_sent = 0
            self.bytes_received = 0
            self.round_start_time = None

        def __repr__(self) -> str:
            # Same function as FedAvg(Strategy)
            return f"FedCustom (accept_failures={self.accept_failures})"

        def initialize_parameters(
            self, client_manager: ClientManager
        ) -> Optional[Parameters]:
            """Initialize global model parameters."""
            # Same function as FedAvg(Strategy)
            initial_parameters = self.initial_parameters
            self.initial_parameters = None  # Don't keep initial parameters in memory
            return initial_parameters

        def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
            """Return sample size and required number of clients."""
            # Same function as FedAvg(Strategy)
            num_clients = int(num_available_clients * self.fraction_fit)
            return max(num_clients, self.min_fit_clients), self.min_available_clients

        def configure_fit(
            self,
            server_round: int,
            parameters: Parameters,
            client_manager: ClientManager,
        ) -> List[Tuple[ClientProxy, FitIns]]:
            """Configure the next round of training."""
            # Proxy Parameters byte size
            serialized_params = pickle.dumps(parameters)
            param_size = len(serialized_params)
            self.bytes_sent += param_size

            # Log sent data
            wandb.log(
                {"Bytes Sent (Round)": param_size, "Total Bytes Sent": self.bytes_sent},
                step=server_round,
            )

            # Sample clients
            sample_size, min_num_clients = self.num_fit_clients(
                client_manager.num_available()
            )

            clients = client_manager.sample(
                num_clients=sample_size, min_num_clients=min_num_clients
            )
            # Create custom configs
            n_clients = len(clients)
            half_clients = n_clients // 2
            # Custom fit config function provided
            standard_lr = lr
            higher_lr = 0.003
            config = {"server_round": server_round, "local_epochs": 1}
            if self.on_fit_config_fn is not None:
                # Custom fit config function provided
                config = self.on_fit_config_fn(server_round)

            # fit_ins = FitIns(parameters, config)
            # Return client/config pairs
            fit_configurations = []
            for idx, client in enumerate(clients):
                config["learning_rate"] = (
                    standard_lr if idx < half_clients else higher_lr
                )
                # Each pair of (ClientProxy, FitRes) constitutes
                # a successful update from one of the previously selected clients.
                fit_configurations.append((client, FitIns(parameters, config)))
            # Successful updates from the previously selected and configured clients
            self.round_start_time = time.time()
            return fit_configurations

        def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
        ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
            """Aggregate fit results using weighted average. (each round)"""
            # Calculate round time
            round_end_time = time.time()
            round_time = round_end_time - self.round_start_time

            # Log round time to wandb
            wandb.log(
                {"round": server_round, "round_time": round_time}, step=server_round
            )

            # Calculate the size of received parameters from clients
            round_received = sum(
                len(pickle.dumps(fit_res.parameters)) for _, fit_res in results
            )
            self.bytes_received += round_received

            # Log received data
            wandb.log(
                {
                    "Bytes Received (Round)": round_received,
                    "Total Bytes Received": self.bytes_received,
                },
                step=server_round,
            )

            # Same function as FedAvg(Strategy)
            if not results:
                return None, {}

            # Do not aggregate if there are failures and failures are not accepted
            if not self.accept_failures and failures:
                return None, {}

            aggregation_start_time = time.time()
            # Convert results parameters --> array matrix
            weights_results = [
                (
                    parameters_to_ndarrays_custom(
                        fit_res.parameters, self.context_client
                    ),
                    fit_res.num_examples,
                )
                for _, fit_res in results
            ]

            # Aggregate parameters using weighted average between the clients
            # and convert back to parameters object (bytes)
            parameters_aggregated = ndarrays_to_parameters_custom(
                fhe.aggregate_custom(weights_results)
            )
            aggregation_end_time = time.time() - aggregation_start_time
            wandb.log(
                {"parameter_aggregation_time": aggregation_end_time}, step=server_round
            )

            metrics_aggregated = {}
            # Aggregate custom metrics if aggregation fn was provided
            if self.fit_metrics_aggregation_fn:
                fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
                metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

            elif server_round == 1:  # Only log this warning once
                logger.log(WARNING, "No fit_metrics_aggregation_fn provided")

            # Same function as SaveModelStrategy(fl.server.strategy.FedAvg)
            # Aggregate model weights using weighted average and store checkpoint
            aggreg_fit_checkpoint(
                server_round,
                parameters_aggregated,
                central,
                model_checkpoint_path,
                self.context_client,
                encrypted_model_checkpoint_path,
            )
            return parameters_aggregated, metrics_aggregated

        def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
            """Use a fraction of available clients for evaluation."""
            # Same function as FedAvg(Strategy)
            num_clients = int(num_available_clients * self.fraction_evaluate)
            return (
                max(num_clients, self.min_evaluate_clients),
                self.min_available_clients,
            )

        def configure_evaluate(
            self,
            server_round: int,
            parameters: Parameters,
            client_manager: ClientManager,
        ) -> List[Tuple[ClientProxy, EvaluateIns]]:
            """Configure the next round of evaluation."""
            # Same function as FedAvg(Strategy)
            # Do not configure federated evaluation if fraction eval is 0.
            if self.fraction_evaluate == 0.0:
                return []

            # Parameters and config
            config = {}  # {"server_round": server_round, "local_epochs": 1}

            evaluate_ins = EvaluateIns(parameters, config)

            # Sample clients
            sample_size, min_num_clients = self.num_evaluation_clients(
                client_manager.num_available()
            )

            clients = client_manager.sample(
                num_clients=sample_size, min_num_clients=min_num_clients
            )

            # Return client/config pairs
            # Each pair of (ClientProxy, FitRes) constitutes a successful
            # update from one of the previously selected clients
            return [(client, evaluate_ins) for client in clients]

        def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
        ) -> Tuple[Optional[float], Dict[str, Scalar]]:
            """Aggregate evaluation losses using weighted average."""
            # Same function as FedAvg(Strategy)
            if not results:
                return None, {}

            # Do not aggregate if there are failures and failures are not accepted
            if not self.accept_failures and failures:
                return None, {}

            # Aggregate loss
            loss_aggregated = weighted_loss_avg(
                [
                    (evaluate_res.num_examples, evaluate_res.loss)
                    for _, evaluate_res in results
                ]
            )

            metrics_aggregated = {}
            # Aggregate custom metrics if aggregation fn was provided
            if self.evaluate_metrics_aggregation_fn:
                eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
                metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)

            # Only log this warning once
            elif server_round == 1:
                logger.log(WARNING, "No evaluate_metrics_aggregation_fn provided")

            wandb.log(
                {"loss_agg": loss_aggregated, "metrics_agg": metrics_aggregated},
                step=server_round,
            )
            return loss_aggregated, metrics_aggregated

        def evaluate(
            self, server_round: int, parameters: Parameters
        ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
            """Evaluate global model parameters using an evaluation function."""
            # Same function as FedAvg(Strategy)
            if self.evaluate_fn is None:
                # Let's assume we won't perform the global model evaluation on the server side.
                return None

            # if we have a global model evaluation on the server side :
            parameters_ndarrays = parameters_to_ndarrays_custom(
                parameters, self.context_client
            )
            eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})

            # if you haven't results
            if eval_res is None:
                return None

            loss, metrics = eval_res
            wandb.log(
                {"loss_central": loss, "metrics_central": metrics}, step=server_round
            )
            return loss, metrics

    return FedCustom
