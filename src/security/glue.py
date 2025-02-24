"""Module to assure compatibility of Flower and TenSEAL
"""

from typing import cast
import tenseal as ts
from io import BytesIO

import numpy as np
import torch
from flwr.common import Parameters, NDArray, NDArrays

from security import fhe


def ndarrays_to_parameters(ndarrays: NDArrays) -> Parameters:
    return ndarrays_to_parameters_custom(ndarrays)


def ndarray_to_bytes(ndarray: NDArray) -> bytes:
    return ndarray_to_bytes_custom(ndarray)


def bytes_to_ndarray(tensor: bytes) -> NDArray:
    bytes_io = BytesIO(tensor)
    ndarray_deserialized = np.load(bytes_io, allow_pickle=False)
    return cast(NDArray, ndarray_deserialized)


def ndarray_to_bytes_custom(ndarray: NDArray) -> bytes:
    if type(ndarray) == ts.tensors.CKKSTensor:
        return ndarray.serialize()

    bytes_io = BytesIO()
    np.save(
        bytes_io,
        ndarray.cpu().detach().numpy() if type(ndarray) == torch.Tensor else ndarray,
        allow_pickle=False,
    )
    return bytes_io.getvalue()


def bytes_to_ndarray_custom(tensor: bytes, context_client) -> NDArray:
    try:
        ndarray_deserialized = ts.ckks_tensor_from(context_client, tensor)
    except:
        bytes_io = BytesIO(tensor)
        ndarray_deserialized = np.load(bytes_io, allow_pickle=False)

    return cast(NDArray, ndarray_deserialized)


def ndarrays_to_parameters_custom(ndarrays: NDArrays) -> Parameters:
    tensors = [ndarray_to_bytes_custom(ndarray) for ndarray in ndarrays]
    return Parameters(tensors=tensors, tensor_type="numpy.ndarray")


def parameters_to_ndarrays_custom(parameters: Parameters, context_client) -> NDArrays:
    return [
        bytes_to_ndarray_custom(tensor, context_client) for tensor in parameters.tensors
    ]


def combo_keys(client_path="secret.pkl", server_path="server_key.pkl"):
    """
    To create the public/private keys combination
    args:
        client_path: path to save the secret key (str)
        server_path: path to save the server public key (str)
    """
    context_client = fhe.context()
    fhe.write_query(
        client_path, {"contexte": context_client.serialize(save_secret_key=True)}
    )
    fhe.write_query(server_path, {"contexte": context_client.serialize()})

    _, context_client = fhe.read_query(client_path)
    _, context_server = fhe.read_query(server_path)

    context_client = ts.context_from(context_client)
    context_server = ts.context_from(context_server)
    print(
        "Is the client context private?",
        ("Yes" if context_client.is_private() else "No"),
    )
    print(
        "Is the server context private?",
        ("Yes" if context_server.is_private() else "No"),
    )
