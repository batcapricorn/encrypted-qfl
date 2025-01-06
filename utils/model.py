"""Module to generate pyTorch models"""

from torch import nn
import torch
import pennylane as qml


class SimpleNet(nn.Module):
    """
    A simple CNN model

    Args:
        num_classes: An integer indicating the number of classes in the dataset.
    """

    def __init__(self, num_classes=10) -> None:
        super(SimpleNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 56 * 56, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def simple_qnn_factory(n_qubits, n_layers):
    weight_shapes = {"weights": (n_layers, n_qubits)}
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def quantum_net(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    class SimpleQNN(nn.Module):
        """
        A simple CNN model

        Args:
            num_classes: An integer indicating the number of classes in the dataset.
        """

        def __init__(self, num_classes=10) -> None:
            super(SimpleQNN, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.classifier = nn.Sequential(
                nn.Linear(32 * 56 * 56, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, n_qubits),
                qml.qnn.TorchLayer(quantum_net, weight_shapes=weight_shapes),
                nn.Linear(n_qubits, num_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass of the neural network
            """
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x

    return SimpleQNN
