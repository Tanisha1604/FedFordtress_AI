import copy
import torch
import numpy as np
import logging
import pickle
from pathlib import Path
from src.models.simple_model import SimpleCNN
from src.client.client import Client
from src.data.data_utils import load_cifar10, iid_split, create_dataloaders
from src.server.server import AsyncFLServer, ServerConfig
from src.server.aggregation import ClientUpdate
from src.privacy.dp import ServerDifferentialPrivacy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- Helper Functions for Server Integration ----------------
def torch_to_numpy(state_dict):
    """Convert PyTorch model state dict to numpy arrays for server."""
    numpy_state = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            numpy_state[key] = value.detach().cpu().numpy()
        else:
            numpy_state[key] = np.array(value)
    return numpy_state

def compute_model_update(current_state, new_state):
    """Compute the update (new - current) for submission to server."""
    update = {}
    for key in current_state:
        update[key] = new_state[key] - current_state[key]
    return update

# ---------------- FedAvg ----------------
def fedavg(local_weights, local_samples):
    global_weights = copy.deepcopy(local_weights[0])
    total_samples = sum(local_samples)

    for key in global_weights.keys():
        global_weights[key] = sum(
            local_weights[i][key] * (local_samples[i] / total_samples)
            for i in range(len(local_weights))
        )
    return global_weights


# ---------------- Trimmed Mean ----------------
def trimmed_mean(local_weights, trim_ratio=0.34):
    num_clients = len(local_weights)
    trim_k = int(num_clients * trim_ratio)
    global_weights = {}

    for key in local_weights[0].keys():
        # Convert all tensors to float to avoid dtype issues
        stacked = torch.stack([w[key].float() for w in local_weights])
        sorted_weights, _ = torch.sort(stacked, dim=0)
        trimmed = sorted_weights[trim_k:num_clients - trim_k]
        global_weights[key] = torch.mean(trimmed, dim=0)

    return global_weights


# ---------------- Median ----------------
def median_aggregation(local_weights):
    global_weights = {}

    for key in local_weights[0].keys():
        # Convert all tensors to float to avoid dtype issues
        stacked = torch.stack([w[key].float() for w in local_weights])
        global_weights[key] = torch.median(stacked, dim=0).values

    return global_weights


# ---------------- Anomaly Score ----------------
def compute_update_norm(local_weights, global_weights):
    total_norm = 0
    for key in global_weights:
        # Convert to float to handle potential Long tensors
        local_tensor = local_weights[key].float()
        global_tensor = global_weights[key].float()
        diff = local_tensor - global_tensor
        total_norm += torch.norm(diff).item()
    return total_norm


# ---------------- Federated Training with Async Server ----------------
def run_federated_training_with_server(
        num_clients=3,
        malicious_clients=1,
        rounds=3,
        dp_enabled=True,
        async_buffer_size=None):

    device = torch.device("cpu")

    # Load data and create client dataloaders
    dataset = load_cifar10()
    subsets = iid_split(dataset, num_clients=num_clients)
    client_loaders = create_dataloaders(subsets, batch_size=64)

    # Initialize global model
    global_model = SimpleCNN().to(device)
    global_model_params = global_model.state_dict()
    
    # Convert to numpy for server
    numpy_params = torch_to_numpy(global_model_params)
    
    # Configure and create server
    if async_buffer_size is None:
        async_buffer_size = num_clients
    
    config = ServerConfig(
        async_buffer_size=async_buffer_size,
        async_timeout=10.0,
        dp_epsilon=1.0,
        dp_delta=1e-5,
        dp_enabled=dp_enabled,
    )
    
    server = AsyncFLServer(numpy_params, config)
    
    # Create and connect clients to server
    clients = []
    for i in range(num_clients):
        is_malicious = i < malicious_clients
        client = Client(
            client_id=i,
            model=copy.deepcopy(global_model),
            dataloader=client_loaders[i],
            malicious=is_malicious
        )
        client.connect_to_server(server)
        clients.append(client)

    for round_num in range(rounds):
        
        local_accuracies = []
        anomaly_scores = []
        
        # Step 1: All clients receive global model and train
        for i, client in enumerate(clients):
            # Receive global model from server
            client.receive_global_model_from_server()
            
            # Local training
            result = client.local_train(epochs=1)
            
            local_accuracies.append(result["local_accuracy"])
            
            # Compute update norm for metrics
            score = compute_update_norm(
                result["state_dict"],
                global_model_params
            )
            anomaly_scores.append(score)
            
            # Compute and submit update to server
            current_params = torch_to_numpy(global_model_params)
            new_params = torch_to_numpy(result["state_dict"])
            update = compute_model_update(current_params, new_params)
            
            client.submit_update_to_server(
                update,
                result["num_samples"],
                result["loss"]
            )
        
        # Step 2: Server aggregates updates
        metrics = server.force_aggregate()
        
        if metrics:
            # Get updated global model from server
            numpy_params, _ = server.get_global_model()
            
            # Convert back to PyTorch
            global_model_params = {}
            for key, value in numpy_params.items():
                global_model_params[key] = torch.from_numpy(value).float()
            
            global_model.load_state_dict(global_model_params)
        
        avg_accuracy = sum(local_accuracies) / len(local_accuracies)
        avg_loss = sum(anomaly_scores) / len(anomaly_scores)

        yield {
            "round": round_num + 1,
            "accuracy": round(avg_accuracy, 2),
            "loss": round(avg_loss, 3),
            "anomaly_scores": anomaly_scores,
            "local_accuracies": local_accuracies,
            "server_metrics": {
                "num_updates": metrics.num_updates if metrics else 0,
                "num_filtered": metrics.num_filtered if metrics else 0,
                "confidence": metrics.confidence if metrics else 0,
            } if metrics else {}
        }


# ---------------- Federated Training (Original) ----------------
def run_federated_training(
        aggregation="FedAvg",
        num_clients=5,
        malicious_clients=1,
        rounds=10,
        quick_mode=False,
        max_samples=50000,
        dp_enabled=True,
        dp_epsilon=1.0,
        dp_delta=1e-5,
        local_epochs=5,
        learning_rate=0.01,
        attack_type="noise_injection"):
    """
    Run federated learning training on CIFAR-10 dataset.
    
    Args:
        aggregation: Aggregation method ("FedAvg", "Trimmed Mean", "Median")
        num_clients: Number of client devices
        malicious_clients: Number of malicious clients
        rounds: Number of federated learning rounds
        quick_mode: If True, use subset of data for faster training
        max_samples: Maximum samples to use (full CIFAR-10 has 50,000)
        dp_enabled: Enable differential privacy
        dp_epsilon: DP privacy budget (smaller = more private, more noise)
        dp_delta: DP failure probability
        local_epochs: Number of local training epochs per client per round
        learning_rate: Learning rate for local training
    
    Yields:
        Dictionary with round metrics (accuracy, loss, anomaly_scores, etc.)
    """

    device = torch.device("cpu")

    # Load CIFAR-10 data (full dataset by default)
    dataset = load_cifar10(
        quick_mode=quick_mode, 
        max_samples=max_samples,
        use_augmentation=True  # Use augmentation for real CIFAR-10
    )
    logger.info(f"Loaded CIFAR-10 dataset: {len(dataset)} samples, {num_clients} clients")
    
    subsets = iid_split(dataset, num_clients=num_clients)
    client_loaders = create_dataloaders(subsets, batch_size=64)

    global_model = SimpleCNN().to(device)

    # DP noise scale (calibrated for CIFAR-10 model weights)
    noise_scale = 0.0
    if dp_enabled:
        # Scale noise based on epsilon (higher epsilon = less noise)
        noise_scale = 0.05 / dp_epsilon
        logger.info(f"DP enabled: epsilon={dp_epsilon}, delta={dp_delta}, noise_scale={noise_scale:.4f}")

    for round_num in range(rounds):

        local_weights = []
        local_samples = []
        local_accuracies = []
        anomaly_scores = []

        # Map frontend attack_type names to Client's internal names
        attack_map = {
            "noise_injection": "noise",
            "label_flipping": "label_flip",
            "weight_scaling": "weight_scaling",
            "random_weights": "random_weights",
        }
        client_attack = attack_map.get(attack_type, "noise")

        for i in range(num_clients):

            is_malicious = i < malicious_clients

            client = Client(
                client_id=i,
                model=copy.deepcopy(global_model),
                dataloader=client_loaders[i],
                malicious=is_malicious,
                attack_type=client_attack
            )
            
            # Update client learning rate
            client.optimizer = torch.optim.SGD(client.model.parameters(), lr=learning_rate)

            client.receive_global_model(global_model.state_dict())
            result = client.local_train(epochs=local_epochs)

            local_weights.append(result["state_dict"])
            local_samples.append(result["num_samples"])
            local_accuracies.append(result["local_accuracy"])

            score = compute_update_norm(
                result["state_dict"],
                global_model.state_dict()
            )
            anomaly_scores.append(score)

        # ---------- Aggregation ----------
        if aggregation == "FedAvg":
            global_weights = fedavg(local_weights, local_samples)

        elif aggregation == "Trimmed Mean":
            global_weights = trimmed_mean(local_weights)

        elif aggregation == "Median":
            global_weights = median_aggregation(local_weights)

        else:
            global_weights = fedavg(local_weights, local_samples)

        # Apply DP noise to aggregated weights
        if dp_enabled and noise_scale > 0:
            for key in global_weights:
                noise = torch.randn_like(global_weights[key]) * noise_scale
                global_weights[key] = global_weights[key] + noise
            logger.info(f"Round {round_num + 1}: Applied DP noise (scale={noise_scale:.4f})")

        global_model.load_state_dict(global_weights)

        avg_accuracy = sum(local_accuracies) / len(local_accuracies)
        avg_loss = sum(anomaly_scores) / len(anomaly_scores)

        logger.info(f"Round {round_num + 1}/{rounds}: Accuracy={avg_accuracy:.2f}%, Loss={avg_loss:.3f}")

        yield {
            "round": round_num + 1,
            "accuracy": round(avg_accuracy, 2),
            "loss": round(avg_loss, 3),
            "anomaly_scores": anomaly_scores,
            "local_accuracies": local_accuracies
        }


def save_training_results(results, aggregation, num_clients, malicious_clients, 
                          rounds, quick_mode, max_samples, dp_enabled, dp_epsilon,
                          save_dir="results"):
    """
    Save training results to a pickle file.
    
    Args:
        results: List of training round results
        aggregation: Aggregation method used
        num_clients: Number of clients
        malicious_clients: Number of malicious clients
        rounds: Number of training rounds
        quick_mode: Whether quick mode was enabled
        max_samples: Maximum samples used
        dp_enabled: Whether DP was enabled
        dp_epsilon: DP epsilon value
        save_dir: Directory to save results
    
    Returns:
        Path to saved file
    """
    import datetime
    
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_str = "quick" if quick_mode else "full"
    filename = f"cifar10_{aggregation.replace(' ', '_')}_{mode_str}_{timestamp}.pkl"
    filepath = save_path / filename
    
    # Prepare data to save
    save_data = {
        "timestamp": timestamp,
        "config": {
            "aggregation": aggregation,
            "num_clients": num_clients,
            "malicious_clients": malicious_clients,
            "rounds": rounds,
            "quick_mode": quick_mode,
            "max_samples": max_samples,
            "dp_enabled": dp_enabled,
            "dp_epsilon": dp_epsilon,
            "dataset": "CIFAR-10"
        },
        "results": results,
        "summary": {
            "final_accuracy": results[-1]["accuracy"] if results else 0,
            "initial_accuracy": results[0]["accuracy"] if results else 0,
            "accuracy_improvement": results[-1]["accuracy"] - results[0]["accuracy"] if len(results) > 1 else 0,
            "total_rounds": len(results)
        }
    }
    
    # Save to pickle file
    with open(filepath, 'wb') as f:
        pickle.dump(save_data, f)
    
    logger.info(f"Training results saved to: {filepath}")
    
    return filepath


def load_training_results(filepath):
    """
    Load training results from a pickle file.
    
    Args:
        filepath: Path to pickle file
    
    Returns:
        Dictionary with training results and config
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)
