"""Module for special Pennylane circuits"""

import pennylane as qml
from pennylane import numpy as np


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


def get_qcnn_circuit(dev, n_qubits):
    """Function to genereate qcnn circuit that can
    be used in PyTorch models

    :param dev: Device (e.g. `cpu`)
    :type dev: Any
    :param n_qubits: Number of qubits
    :type n_qubits: int
    """

    def cluster_state():
        """Prepare a 16-qubit cluster state"""
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
        for i in range(n_qubits - 1):
            qml.CZ(wires=[i, i + 1])
        qml.CZ(wires=[n_qubits - 1, 0])  # Wraparound CZ

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

    return qcnn_circuit
