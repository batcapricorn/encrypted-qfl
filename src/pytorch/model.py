"""Module to generate pyTorch models"""

from torch import nn
import torch
from torchvision import models
import pennylane as qml

from pytorch import circuits


class BasicLayers(nn.Module):
    """
    Basic layers needed for multiple modules

    :param num_classes: An integer indicating the number of classes in the dataset.
    :type num_classes: int
    """

    def __init__(self, num_classes=10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network
        """
        x = self.features(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class SimpleNet(BasicLayers):
    """PyTorch Model Class for simple NN

    :param num_classes: An integer indicating the number of classes in the dataset.
    :type num_classes: int
    """

    def __init__(self, num_classes=10) -> None:
        super().__init__(num_classes=num_classes)


class SimpleResNet18(nn.Module):
    """PyTorch Model Class using ResNet18 for feature extraction.

    :param num_classes: An integer indicating the number of classes in the dataset.
    :type num_classes: int
    """

    def __init__(self, num_classes=10) -> None:
        super().__init__()

        self.features = models.resnet18(pretrained=True)

        for param in self.features.parameters():
            param.requires_grad = False

        num_features = self.features.fc.in_features
        self.features.fc = nn.Sequential(nn.Linear(num_features, num_classes))
        self.features.fc = self.features.fc.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network
        """
        return self.features(x)


def simple_qnn_factory(n_qubits, n_layers):
    """Function that creates a class that can be used
    to create a simple NN with quantum layers.

    :param n_qubits: number of quibits
    :type n_qubits: int
    :param n_layers: number of layers for `BasicEntanglerLayers`
    :type n_layers: _int
    :return: PyTorch Model Class
    :rtype: nn.Module
    """
    weight_shapes = {"weights": (n_layers, n_qubits)}
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def quantum_net(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    class SimpleQNN(BasicLayers):
        """PyTorch Model Class for simple QNN

        :param num_classes: An integer indicating the number of classes in the dataset.
        :type num_classes: int
        """

        def __init__(self, num_classes=10) -> None:
            super().__init__(num_classes=num_classes)
            self.classifier = nn.Sequential(
                nn.Linear(16, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, n_qubits),
                qml.qnn.TorchLayer(quantum_net, weight_shapes=weight_shapes),
                nn.Linear(n_qubits, num_classes),
            )

    return SimpleQNN


def simple_resnet18_qnn_factory(n_qubits, n_layers):
    """Function that creates PyTorch module class
    for simple QNN combined with resnet18.

    :param n_qubits: Number of qubits
    :type n_qubits: int
    :param n_layers: Number of layers
    :type n_layers: int
    :return: PyTorch Module class
    :rtype: nn.Module
    """
    weight_shapes = {"weights": (n_layers, n_qubits)}
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def quantum_net(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    class SimpleResNet18QNN(nn.Module):
        """PyTorch Model Class using ResNet18 for feature extraction.

        :param num_classes: An integer indicating the number of classes in the dataset.
        :type num_classes: int
        """

        def __init__(self, num_classes=10) -> None:
            super().__init__()

            self.features = models.resnet18(pretrained=True)

            for param in self.features.parameters():
                param.requires_grad = False

            num_features = self.features.fc.in_features
            self.features.fc = nn.Sequential(nn.Linear(num_features, n_qubits))
            self.features.fc = self.features.fc.requires_grad_(True)

            self.classifier = nn.Sequential(
                qml.qnn.TorchLayer(quantum_net, weight_shapes=weight_shapes),
                nn.Linear(n_qubits, num_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass of the neural network
            """
            x = self.features(x)
            x = self.classifier(x)
            return x

    return SimpleResNet18QNN


def qcnn_factory():
    """Returns a QCNN class that reduces 8 input qubits to 4

    :return: PyTorch Model Class
    :rtype: nn.Module
    """
    # Define quantum device with 16 qubits
    n_qubits = 8
    n_final_qubits = 4
    weight_shapes = {"weights": (21)}
    dev = qml.device("default.qubit", wires=n_qubits)

    qcnn_circuit = circuits.get_qcnn_circuit(dev, n_qubits)

    # Define the modified SimpleQNN integrating QCNN
    class QCNN(BasicLayers):
        """
        A hybrid CNN-QCNN model for multi-class classification.

        Args:
            num_classes: Number of output classes.
        """

        def __init__(self, num_classes=10) -> None:
            super().__init__(num_classes=num_classes)

            # Fully Connected + QCNN layers
            self.classifier = nn.Sequential(
                nn.Linear(16, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, n_qubits),
                qml.qnn.TorchLayer(
                    qcnn_circuit, weight_shapes=weight_shapes
                ),  # Quantum Convolutional Neural Network
                nn.Linear(n_final_qubits, num_classes),  # Final classification
            )

    return QCNN


def resnet18_qcnn_factory():
    """Returns a QCNN class that reduces 8 input qubits to 4.
    Images are embedded using Resnet18.

    :return: PyTorch Model Class
    :rtype: nn.Module
    """
    # Define quantum device with 16 qubits
    n_qubits = 8
    weight_shapes = {"weights": (21)}
    dev = qml.device("default.qubit", wires=n_qubits)

    qcnn_circuit = circuits.get_qcnn_circuit(dev, n_qubits)

    class ResNet18QCNN(nn.Module):
        """PyTorch Model Class using ResNet18 for feature extraction.

        :param num_classes: An integer indicating the number of classes in the dataset.
        :type num_classes: int
        """

        def __init__(self, num_classes=10) -> None:
            super().__init__()

            self.features = models.resnet18(pretrained=True)

            for param in self.features.parameters():
                param.requires_grad = False

            num_features = self.features.fc.in_features
            self.features.fc = nn.Sequential(nn.Linear(num_features, n_qubits))
            self.features.fc = self.features.fc.requires_grad_(True)

            self.classifier = nn.Sequential(
                qml.qnn.TorchLayer(qcnn_circuit, weight_shapes=weight_shapes),
                nn.Linear(n_qubits, num_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass of the neural network
            """
            x = self.features(x)
            x = self.classifier(x)
            return x

    return ResNet18QCNN
