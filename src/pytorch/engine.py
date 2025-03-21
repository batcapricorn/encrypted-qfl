"""
Contains functions for training and testing a PyTorch model.
"""

from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm


def test(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: Union[torch.nn.Module, Tuple],
    device: torch.device,
):
    """
    Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    :param model: A PyTorch model to be tested.
    :type model: torch.nn.Module
    :param dataloader: A DataLoader instance for the model to be tested on.
    :type dataloader: torch.utils.data.DataLoader
    :param loss_fn: A PyTorch loss function to calculate loss on the test data.
    :type loss_fn: torch.nn.Module
    :param device: A target device to compute on (e.g., "cuda" or "cpu").
    :type device: torch.device

    :return: A tuple containing the test loss and test accuracy.
    :rtype: Tuple[float, float]

    **Example:**

    >>> test_loss, test_acc = test(model, dataloader, loss_fn, device)
    >>> print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")
    """
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0
    y_pred = []
    y_true = []
    y_proba = []
    softmax = nn.Softmax(dim=1)

    # Turn on inference context manager
    with torch.inference_mode():
        # torch.inference_mode is analogous to torch.no_grad :
        # gets better performance by disabling view tracking and version counter bumps

        # Loop through DataLoader batches
        for images, labels in dataloader:
            # Send data to target device
            images, labels = images.to(device), labels.to(device)

            # 1. Forward pass
            output = model(images)

            # 2. Calculate and accumulate probas
            probas_output = softmax(output)
            y_proba.extend(probas_output.detach().cpu().numpy())

            # 3. Calculate and accumulate loss
            loss = loss_fn(output, labels)
            test_loss += loss.item()

            # 4. Calculate and accumulate accuracy
            labels = labels.data.cpu().numpy()
            y_true.extend(labels)  # Save Truth
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            y_pred.extend(preds)  # Save Prediction
            acc = (preds == labels).mean()
            test_acc += acc

    y_proba = np.array(y_proba)
    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc * 100, y_pred, y_true, y_proba


def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: Union[torch.nn.Module, Tuple],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then runs through all
    the required training steps, including forward pass, loss calculation,
    and optimizer step.

    :param model: A PyTorch model to be trained.
    :type model: torch.nn.Module
    :param dataloader: A DataLoader instance for the model to be trained on.
    :type dataloader: torch.utils.data.DataLoader
    :param loss_fn: A PyTorch loss function to minimize.
    :type loss_fn: torch.nn.Module
    :param optimizer: A PyTorch optimizer to help minimize the loss function.
    :type optimizer: torch.optim.Optimizer
    :param device: A target device to compute on (e.g., "cuda" or "cpu").
    :type device: torch.device

    :return: A tuple containing the training loss and training accuracy.
    :rtype: Tuple[float, float]

    **Example:**

    >>> train_loss, train_acc = train_step(model, dataloader, loss_fn, optimizer, device)
    >>> print(f"Train Loss: {train_loss}, Train Accuracy: {train_acc}")
    """
    # Put model in training mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for _, (images, labels) in enumerate(dataloader):

        # Send data to target device
        images, labels = images.to(device), labels.to(device)

        # 1. Optimizer zero grad
        optimizer.zero_grad()  # Sets the gradients of all optimized torch.Tensor to zero.

        # 2. Forward pass
        output = model(images)

        # 3. Calculate  and accumulate loss
        loss = loss_fn(output, labels)
        train_loss += loss.item()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(output, dim=1), dim=1)
        train_acc += (y_pred_class == labels).sum().item() / len(output)

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc * 100


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Union[torch.nn.Module, Tuple],
    epochs: int,
    device: torch.device,
) -> Dict[str, List]:
    """
    Trains and tests a PyTorch model.

    Passes a target PyTorch model through `train_step()` and `test()` functions
    for a specified number of epochs, training and testing the model in the same loop.

    Evaluation metrics are calculated, printed, and stored throughout training.

    :param model: A PyTorch model to be trained and tested.
    :type model: torch.nn.Module
    :param train_dataloader: DataLoader for training the model.
    :type train_dataloader: torch.utils.data.DataLoader
    :param test_dataloader: DataLoader for testing the model.
    :type test_dataloader: torch.utils.data.DataLoader
    :param optimizer: Optimizer to minimize the loss function.
    :type optimizer: torch.optim.Optimizer
    :param loss_fn: Loss function used for both training and testing.
    :type loss_fn: torch.nn.Module
    :param epochs: Number of epochs to train the model.
    :type epochs: int
    :param device: Target device for computation (e.g., "cuda" or "cpu").
    :type device: torch.device
    :param task: Optional string specifying the task (default is None).
    :type task: Optional[str]

    :return: A dictionary containing training and testing loss, as well as accuracy metrics.
             Each metric is stored as a list with values for each epoch.
    :rtype: Dict[str, List[float]]

    **Example:**

    >>> metrics = train_and_test(
            model, train_dataloader, test_dataloader, optimizer, loss_fn, epochs=10, device="cuda"
        )
    >>> print(metrics["train_loss"], metrics["val_acc"])
    """
    # Create empty results dictionary
    results = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs), colour="BLUE"):
        # Select functions based on the task
        train_step_fn = train_step
        test_fn = test

        # Perform training and validation
        train_loss, train_acc = train_step_fn(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )
        val_loss, val_acc, *_ = test_fn(
            model=model, dataloader=test_dataloader, loss_fn=loss_fn, device=device
        )

        # Print out what's happening
        print(
            f"\tTrain Epoch: {epoch + 1} \t"
            f"Train_loss: {train_loss:.4f} | "
            f"Train_acc: {train_acc:.4f} % | "
            f"Validation_loss: {val_loss:.4f} | "
            f"Validation_acc: {val_acc:.4f} %"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)

    # Return the filled results at the end of the epochs
    return results
