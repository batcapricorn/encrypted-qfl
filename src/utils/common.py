"""
Helper functions for running FL experiment, e.g., functions to
export ROC export
"""

from collections import OrderedDict
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

import torch
import wandb

from security.fhe import crypte, deserialized_layer


def choice_device(device):
    """
    A function to choose the device

    :param device: the device to choose (cpu, gpu or mps)
    """
    if torch.cuda.is_available() and device != "cpu":
        # on Windows, "cuda:0" if torch.cuda.is_available()
        device = "cuda:0"

    elif (
        torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
        and device != "cpu"
    ):
        # on Mac :
        # - torch.backends.mps.is_available() ensures that the current
        #   MacOS version is at least 12.3+
        # - torch.backends.mps.is_built() ensures that the current current
        #   PyTorch installation was built with MPS activated.
        device = "mps"

    else:
        device = "cpu"

    return device


def classes_string(name_dataset):
    """
    A function to get the classes of the dataset

    :param name_dataset: the name of the dataset
    :return: classes (the classes of the dataset) in a tuple
    """
    if name_dataset == "cifar":
        classes = (
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )

    elif name_dataset == "MRI":
        classes = ("glioma", "meningioma", "notumor", "pituitary")

    else:
        print("Warning problem : unspecified dataset")
        return ()

    return classes


def supp_ds_store(path):
    """
    Delete the hidden file ".DS_Store" created on macOS

    :param path: path to the folder where the hidden file ".DS_Store" is
    """
    for i in os.listdir(path):
        if i == ".DS_Store":
            print("Deleting of the hidden file '.DS_Store'")
            os.remove(path + "/" + i)


def save_matrix(y_true, y_pred, path, classes):
    """
    Save the confusion matrix in the path given in argument.

    :param y_true: true labels (real labels)
    :param y_pred: predicted labels (labels predicted by the model)
    :param path: path to save the confusion matrix
    :param classes: list of the classes
    """
    # To get the normalized confusion matrix
    y_true_mapped = [classes[label] for label in y_true]
    y_pred_mapped = [classes[label] for label in y_pred]
    # To get the normalized confusion matrix
    cf_matrix_normalized = confusion_matrix(
        y_true_mapped, y_pred_mapped, labels=classes, normalize="all"
    )

    # To round up the values in the matrix
    cf_matrix_round = np.round(cf_matrix_normalized, 2)

    # To plot the matrix
    df_cm = pd.DataFrame(cf_matrix_round, index=list(classes), columns=list(classes))
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.xlabel("Predicted label", fontsize=13)
    plt.ylabel("True label", fontsize=13)
    plt.title("Confusion Matrix", fontsize=15)

    plt.savefig(path)
    plt.close()


def save_roc(targets, y_proba, path, nbr_classes):
    """
    Save the roc curve in the path given in argument.

    :param targets: true labels (real labels)
    :param y_proba: predicted labels (labels predicted by the model)
    :param path: path to save the roc curve
    :param nbr_classes: number of classes
    """
    y_true = np.zeros(
        shape=(len(targets), nbr_classes)
    )  # array-like of shape (n_samples, n_classes)
    for i in range(len(targets)):
        y_true[i, targets[i]] = 1

    # Compute ROC curve and ROC area for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(nbr_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(nbr_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(nbr_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= nbr_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label=f"micro-average ROC curve (area = {roc_auc['micro']:.2f})",
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label=f"macro-average ROC curve (area = {roc_auc['macro']:.2f})",
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    lw = 2
    for i in range(nbr_classes):
        plt.plot(
            fpr[i],
            tpr[i],
            lw=lw,
            label=f"ROC curve of class {i} (area = {roc_auc[i]:.2f})",
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw, label="Worst case")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic (ROC) Curve OvR")  # One vs Rest
    plt.legend(loc="lower right")  # loc="best"

    plt.savefig(path)
    plt.close()


def save_graphs(path_save, local_epoch, results, end_file=""):
    """
    Save the graphs in the path given in argument.

    :param path_save: path to save the graphs
    :param local_epoch: number of epochs
    :param results: results of the model (accuracy and loss)
    :param end_file: end of the name of the file
    """
    os.makedirs(path_save, exist_ok=True)  # to create folders results
    print("save graph in ", path_save)
    # plot training curves (train and validation)
    plot_graph(
        [[*range(local_epoch)]] * 2,
        [results["train_acc"], results["val_acc"]],
        "Epochs",
        "Accuracy (%)",
        curve_labels=["Training accuracy", "Validation accuracy"],
        title="Accuracy curves",
        path=os.path.join(path_save, f"Accuracy_curves{end_file}"),
    )

    plot_graph(
        [[*range(local_epoch)]] * 2,
        [results["train_loss"], results["val_loss"]],
        "Epochs",
        "Loss",
        curve_labels=["Training loss", "Validation loss"],
        title="Loss curves",
        path=os.path.join(path_save, f"Loss_curves{end_file}"),
    )


def plot_graph(
    list_xplot, list_yplot, x_label, y_label, curve_labels, title, path=None
):
    """
    Plot the graph of the list of points (list_xplot, list_yplot)
    :param list_xplot: list of list of points to plot (one line per curve)
    :param list_yplot: list of list of points to plot (one line per curve)
    :param x_label: label of the x axis
    :param y_label: label of the y axis
    :param curve_labels: list of labels of the curves (curve names)
    :param title: title of the graph
    :param path: path to save the graph
    """
    lw = 2

    plt.figure()
    for i in range(len(curve_labels)):
        plt.plot(list_xplot[i], list_yplot[i], lw=lw, label=curve_labels[i])

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    if curve_labels:
        plt.legend(loc="lower right")

    if path:
        plt.savefig(path)


def get_parameters2(
    net, context_client=None, layers_to_encrypt=None
) -> list[np.ndarray]:
    """
    Get the trainable parameters of the network (only those with requires_grad=True).

    :param net: The neural network model.
    :param context_client: If provided, encrypts the selected layers.
    :param layers_to_encrypt: List of layers to encrypt. If "all" or None, encrypt all layers.
    :return: List of parameters (weights and biases) of the network.
    """
    trainable_keys = {k for k, v in net.named_parameters() if v.requires_grad}
    trainable_params = {
        k: v for k, v in net.state_dict().items() if k in trainable_keys
    }
    if context_client:
        # Encrypt only trainable parameters
        encrypted_tensor = crypte(trainable_params, context_client, layers_to_encrypt)
        return [layer.get_weight() for layer in encrypted_tensor]

    return [val.cpu().numpy() for val in trainable_params.values()]


def set_parameters(net, parameters: list[np.ndarray], context_client=None):
    """
    Update the trainable parameters of the network.

    :param net: The neural network model.
    :param parameters: List of parameters (weights and biases) to set.
    :param context_client: If provided, decrypts the selected layers.
    """
    # Only update layers where requires_grad=True
    trainable_keys = [k for k, v in net.named_parameters() if v.requires_grad]

    params_dict = zip(trainable_keys, parameters)

    if context_client:
        start_time = time.time()
        secret_key = context_client.secret_key()
        dico = {k: deserialized_layer(k, v, context_client) for k, v in params_dict}

        state_dict = OrderedDict(
            {
                k: torch.from_numpy(np.copy(v.decrypt(secret_key)))
                for k, v in dico.items()
            }
        )
        end_time = time.time() - start_time
        wandb.log({"decryption_time": end_time}, commit=False)

    else:
        dico = {k: torch.from_numpy(np.copy(v)) for k, v in params_dict}
        state_dict = OrderedDict(dico)

    # Load only trainable parameters
    net.load_state_dict(state_dict, strict=False)
    print("Updated trainable model parameters")


def eval_classification(y_test, y_pred):
    """Generate different kind of evaluation metrics for (multi) classification tasks"""
    rec = recall_score(
        y_test, y_pred, average="macro"
    )  # average argument required for multi-class
    prec = precision_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")
    return rec, prec, f1
