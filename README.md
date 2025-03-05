[![Python Linting and Formatting](https://github.com/batcapricorn/qfl-playground/actions/workflows/qa.yml/badge.svg)](https://github.com/batcapricorn/qfl-playground/actions/workflows/qa.yml)

# Integrating FHE into QFL
**Thesis Title:** _Fully Homomorphic Encryption for Secure Parameter Sharing in Quantum Federated Learning: Challenges & Analysis_

## TL,DR; 🚀
1. Install dependencies using [pipenv](https://pipenv.pypa.io/en/latest/) and activate its shell:
    ```bash
    pipenv install
    pipenv shell
    ```
2. Adapt settings for training in `settings.yaml`:
    ```bash
    cp example-settings.yaml settings.yaml
    vi settings.yaml
    ```
3. Run training examples using the scripts provided in `srcipts/` and a tiny dataset placed in `data-tiny/`: `./scripts/benchmark.sh fednn --he`
4. On your first run you will be prompted to enter your [wandb](https://wandb.ai/) API key. All kinds of metrics are logged to `wandb`, e.g. encryption and decryption time, training time or network traffic (bytes).

## Cookbook 🍳

### **`benchmark.sh`**  
The main entry point for this project is `./scripts/benchmark.sh`. This script supports two key options:  

- **`--model`**: Specifies the model type. Options include:  
  - `fednn`: Runs a standard convolutional neural network (CNN).  
  - `fedqnn`: Runs the same CNN but with basic quantum layers.  
  - `fedqcnn`: Runs a full **Quantum Convolutional Neural Network (QCNN)** as described in the [TensorFlow Quantum QCNN tutorial](https://www.tensorflow.org/quantum/tutorials/qcnn).  

- **`--he`**: Enables **Fully Homomorphic Encryption (FHE)** using the **CKKS** scheme for parameter encryption.  

#### **Example Usage:**  
- Run a simple **neural network with FHE**: `./scripts/benchmark.sh fednn --he`  
- Run a **Quantum Convolutional Neural Network (QCNN) without FHE**: `./scripts/benchmark.sh fedqcnn`

---

### **`settings.yaml`**

The primary configuration for training is found in `settings.yaml`.
| Parameter | Example Value | Description |
|----------------------------------|--------------------------------|--------------------------------------------------|
| `wandb_project` | `qfl-playground` | Weights & Biases project name |
| `data_path` | `"data-tiny/"` | Path to dataset |
| `dataset` | `"MRI"` | Dataset name |
| `seed` | `0` | Random seed |
| `num_workers` | `0` | Number of workers for PyTorch dataloader. **Setting this to a value greater than 0 may cause concurrency issues.** |
| `max_epochs` | `10` | Maximum training epochs |
| `batch_size` | `10` | Batch size |
| `splitter` | `10` | Data splitting parameter. Defines the percentage of training data used for validation. |
| `device` | `"gpu"` | Compute device (`cpu` or `gpu`) |
| `number_clients` | `1` | Number of FL clients |
| `export_results_path` | `"results/"` | Directory to save results (e.g. ROC curves) |
| `matrix_export` | `True` | Export confusion matrix |
| `roc_export` | `True` | Export ROC curve |
| `min_fit_clients` | `1` | Minimum clients for training |
| `min_avail_clients` | `1` | Minimum available clients |
| `min_eval_clients` | `1` | Minimum clients for evaluation |
| `rounds` | `3` | Number of FL rounds |
| `frac_fit` | `1.0` | Fraction of clients for training |
| `frac_eval` | `0.5` | Fraction of clients for evaluation |
| `lr` | `1e-3` | Learning rate |
| `private_key_path` | `"private_key.pkl"` | Path to client’s private key |
| `public_key_path` | `"public_key.pkl"` | Path to client’s public key |
| `model_checkpoint_path` | `"model_checkpoint.pt"` | Path to unencrypted model checkpoint |
| `encrypted_model_checkpoint_path`| `"encrypted_model_checkpoint.pkl"` | Path to encrypted model checkpoint |
| `n_qubits` | `4` | Number of qubits for quantum layers (only applicable if simple QNN is used) |
| `n_layers` | `6` | Number of layers in quantum circuit (only applicable if simple QNN is used) |


## References 📝
- [QFed+FHE: Quantum Federated Learning with Secure Fully Homomorphic Encryption (FHE)](https://github.com/elucidator8918/QFL-MLNCP-NeurIPS/tree/main)
- [Quantum convolutional neural networks](https://www.nature.com/articles/s41567-019-0648-8)