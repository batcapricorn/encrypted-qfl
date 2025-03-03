[![Python Linting and Formatting](https://github.com/batcapricorn/qfl-playground/actions/workflows/qa.yml/badge.svg)](https://github.com/batcapricorn/qfl-playground/actions/workflows/qa.yml)

# Integrating FHE into QFL
**Thesis Title:** _Fully Homomorphic Encryption for Secure Parameter Sharing in Quantum Federated Learning: Challenges & Analysis_

## TL,DR; 🚀
1. Install dependencies using [pipenv](https://pipenv.pypa.io/en/latest/) and activate its shell:
    ```bash
    pipenv install
    pipenv shell
    ```
2. Adapt settings for training in `settings.yaml`
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
The primary configuration for training is found in `settings.yaml`. Notable options include:  

- **`num_workers`**: Defines the number of workers for the PyTorch dataloader. **Setting this to a value greater than 0 may cause concurrency issues.**  
- **`max_epochs`**: Number of epochs per training iteration.  
- **`number_clients`**: Specifies the number of client processes to be launched.  
- **`min_fit_clients`, `min_avail_clients`, `min_eval_clients`**: Controls the minimum number of clients required for training and evaluation.  
- **`path_public_key`**, **`secret_path`**. **`public_path`**, **`path_crypted`**: Paths to store CKKS key pairs if FHE is enabled.


## References 📝
- [QFed+FHE: Quantum Federated Learning with Secure Fully Homomorphic Encryption (FHE)](https://github.com/elucidator8918/QFL-MLNCP-NeurIPS/tree/main)
- [Quantum convolutional neural networks](https://www.nature.com/articles/s41567-019-0648-8)