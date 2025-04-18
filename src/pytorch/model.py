"""Module to generate pyTorch models"""

from torch import nn
import torch
from torchvision import models
import pennylane as qml
from pennylane import numpy as np


class SimpleNet(nn.Module):
    """PyTorch Model Class for simple NN

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

    class SimpleQNN(nn.Module):
        """PyTorch Model Class for simple QNN

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
                nn.Linear(32, n_qubits),
                qml.qnn.TorchLayer(quantum_net, weight_shapes=weight_shapes),
                nn.Linear(n_qubits, num_classes),
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

    return SimpleQNN


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

    # Define cluster state preparation
    def cluster_state():
        """Prepare a 16-qubit cluster state"""
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
        for i in range(n_qubits - 1):
            qml.CZ(wires=[i, i + 1])
        qml.CZ(wires=[n_qubits - 1, 0])  # Wraparound CZ

    # Define single-qubit unitaries
    def one_qubit_unitary(wire, params):
        """Apply parameterized single-qubit rotations"""
        qml.RX(params[0], wires=wire)
        qml.RY(params[1], wires=wire)
        qml.RZ(params[2], wires=wire)

    # Define two-qubit unitaries
    def two_qubit_unitary(wires, params):
        """Apply a two-qubit convolutional unitary"""
        one_qubit_unitary(wires[0], params[0:3])
        one_qubit_unitary(wires[1], params[3:6])
        qml.IsingZZ(params[6], wires=wires)
        qml.IsingYY(params[7], wires=wires)
        qml.IsingXX(params[8], wires=wires)
        one_qubit_unitary(wires[0], params[9:12])
        one_qubit_unitary(wires[1], params[12:])

    def two_qubit_pool(source_wire, sink_wire, params):
        """Perform a parameterized two-qubit pooling operation."""
        one_qubit_unitary(sink_wire, params[0:3])
        one_qubit_unitary(source_wire, params[3:6])

        qml.CNOT(wires=[source_wire, sink_wire])

        # Undo sink_basis_selector (inverse operation)
        # TO DO: check if this is valid math
        qml.RZ(-params[2] * np.pi, wires=sink_wire)
        qml.RY(-params[1] * np.pi, wires=sink_wire)
        qml.RX(-params[0] * np.pi, wires=sink_wire)

    def quantum_conv_circuit(wires, params):
        """Quantum Convolution Layer.

        Applies a cascade of `two_qubit_unitary` to pairs of qubits in `wires`.
        """

        # Apply two-qubit unitary to adjacent pairs
        for first, second in zip(wires[0::2], wires[1::2]):
            two_qubit_unitary([first, second], params)

        # Apply two-qubit unitary to staggered pairs (wrap-around connection)
        for first, second in zip(wires[1::2], wires[2::2] + [wires[0]]):
            two_qubit_unitary([first, second], params)

    def quantum_pool_circuit(source_wires, sink_wires, params):
        """Quantum Pooling Layer.

        Applies `two_qubit_pool` to pairs of qubits, reducing information
        from `source_wires` onto `sink_wires`.
        """

        for source, sink in zip(source_wires, sink_wires):
            two_qubit_pool(source, sink, params)

    # Define the full QCNN circuit
    @qml.qnode(dev, interface="torch")
    def qcnn_circuit(inputs, weights):
        """QCNN model that processes input data"""
        wires = list(range(n_qubits))
        cluster_state()
        qml.AngleEmbedding(inputs, wires=wires)  # Input encoding

        # Apply convolution and pooling in stages
        quantum_conv_circuit(wires, weights[0:15])
        quantum_pool_circuit(
            wires[:8], wires[8:], weights[15:21]
        )  # Reduce from 8 → 4 qubits

        return [qml.expval(qml.PauliZ(i)) for i in wires[4:]]  # Measure 4 qubits

    # Define the modified SimpleQNN integrating QCNN
    class QCNN(nn.Module):
        """
        A hybrid CNN-QCNN model for multi-class classification.

        Args:
            num_classes: Number of output classes.
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

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass of the hybrid CNN-QCNN"""
            x = self.features(x)  # Extract features using CNN
            x = self.global_avg_pool(x)
            x = x.view(x.size(0), -1)  # Flatten
            x = self.classifier(x)  # Process through QCNN
            return x

    return QCNN
