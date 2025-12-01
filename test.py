import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from transformers import ViTForImageClassification, ViTConfig
from peft import LoraConfig, get_peft_model, TaskType
import wandb
import numpy as np
from typing import List, Dict
import copy
import os
import random

# Configuration
class Config:
    num_clients = 10
    num_rounds = 100
    local_epochs = 10
    batch_size = 128
    learning_rate = 1e-3
    lora_r = 8
    lora_alpha = 16
    lora_dropout = 0.1
    noise_variance = 0.0  # Single noise variance for all clients
    dirichlet_alpha = 0.5  # Dirichlet concentration parameter for non-IID split
    wandb_project = "federated-vit-lora-cifar100"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42  # Random seed for reproducibility

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Data preprocessing
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

# Split data among clients using Dirichlet distribution
def create_client_datasets(dataset, num_clients, alpha=0.5):
    """Create non-IID data split for clients using Dirichlet distribution
    
    Args:
        dataset: PyTorch dataset
        num_clients: Number of clients
        alpha: Dirichlet concentration parameter (lower = more non-IID)
    """
    num_classes = 100  # CIFAR-100 has 100 classes
    
    # Get labels from dataset
    if isinstance(dataset, torch.utils.data.Subset):
        labels = np.array([dataset.dataset.targets[i] for i in dataset.indices])
    else:
        labels = np.array(dataset.targets)
    
    # Group indices by class
    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    
    # Initialize client indices
    client_indices = [[] for _ in range(num_clients)]
    
    # For each class, distribute samples to clients using Dirichlet distribution
    for c in range(num_classes):
        # Sample proportions from Dirichlet distribution
        proportions = np.random.dirichlet(alpha * np.ones(num_clients))
        
        # Get indices for this class
        class_idx = class_indices[c]
        np.random.shuffle(class_idx)
        
        # Split indices according to proportions
        proportions = (np.cumsum(proportions) * len(class_idx)).astype(int)[:-1]
        splits = np.split(class_idx, proportions)
        
        # Assign to clients
        for client_id, split in enumerate(splits):
            client_indices[client_id].extend(split)
    
    # Shuffle each client's data
    for i in range(num_clients):
        np.random.shuffle(client_indices[i])
    
    # Create Subset datasets for each client
    client_datasets = [Subset(dataset, indices) for indices in client_indices]
    
    # Print data distribution statistics
    print("\nData Distribution (Dirichlet alpha={}):".format(alpha))
    for i, indices in enumerate(client_indices):
        client_labels = labels[indices]
        unique, counts = np.unique(client_labels, return_counts=True)
        print(f"Client {i}: {len(indices)} samples, {len(unique)} classes")
    
    return client_datasets

# Create ViT model with LoRA
def create_vit_lora_model(config):
    """Create ViT-tiny model with LoRA adapters"""
    # Load ViT configuration for tiny model
    vit_config = ViTConfig(
        image_size=224,
        patch_size=16,
        num_channels=3,
        num_labels=100,
        hidden_size=192,
        num_hidden_layers=12,
        num_attention_heads=3,
        intermediate_size=768,
    )
    
    # Create model
    model = ViTForImageClassification(vit_config)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["query", "value"],
        bias="none",
        modules_to_save=["classifier"],
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    return model

# Add Gaussian noise to LoRA parameters
def add_gaussian_noise_to_lora(model, variance):
    """Add Gaussian noise to LoRA parameters with specified variance"""
    if variance == 0:
        return model
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'lora' in name.lower() and param.requires_grad:
                noise = torch.randn_like(param) * np.sqrt(variance)
                param.add_(noise)
    
    return model

# Client training
class Client:
    def __init__(self, client_id, dataset, noise_variance, config):
        self.client_id = client_id
        self.dataset = dataset
        self.noise_variance = noise_variance
        self.config = config
        self.dataloader = DataLoader(
            dataset, batch_size=config.batch_size, shuffle=True
        )
        self.model = None
    
    def train(self, global_model):
        """Train local model for specified epochs"""
        self.model = copy.deepcopy(global_model).to(self.config.device)
        self.model.train()
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.config.learning_rate
        )
        
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for epoch in range(self.config.local_epochs):
            for batch_idx, (data, target) in enumerate(self.dataloader):
                data, target = data.to(self.config.device), target.to(self.config.device)
                
                optimizer.zero_grad()
                outputs = self.model(data)
                loss = F.cross_entropy(outputs.logits, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pred = outputs.logits.argmax(dim=1)
                total_correct += pred.eq(target).sum().item()
                total_samples += target.size(0)
        
        # Add Gaussian noise to LoRA parameters
        self.model = add_gaussian_noise_to_lora(self.model, self.noise_variance)
        
        avg_loss = total_loss / (self.config.local_epochs * len(self.dataloader))
        accuracy = 100. * total_correct / total_samples
        
        return avg_loss, accuracy
    
    def get_lora_parameters(self):
        """Extract LoRA parameters"""
        lora_params = {}
        for name, param in self.model.named_parameters():
            if 'lora' in name.lower():
                lora_params[name] = param.data.clone()
        return lora_params

# Server for federated aggregation
class FederatedServer:
    def __init__(self, config, client_datasets, test_loader):
        self.config = config
        self.global_model = create_vit_lora_model(config)
        self.test_loader = test_loader
        self.clients = [
            Client(i, client_datasets[i], config.noise_variance, config)
            for i in range(config.num_clients)
        ]
    
    def aggregate_lora_parameters(self, client_lora_params: List[Dict]):
        """Federated averaging of LoRA parameters"""
        aggregated_params = {}
        
        # Average all LoRA parameters
        for name in client_lora_params[0].keys():
            stacked_params = torch.stack([
                client_params[name] for client_params in client_lora_params
            ])
            aggregated_params[name] = stacked_params.mean(dim=0)
        
        # Update global model with aggregated LoRA parameters
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                if name in aggregated_params:
                    param.copy_(aggregated_params[name])
    
    def evaluate(self):
        """Evaluate global model on test set"""
        self.global_model.to(self.config.device)
        self.global_model.eval()
        
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.config.device), target.to(self.config.device)
                outputs = self.global_model(data)
                loss = F.cross_entropy(outputs.logits, target)
                
                total_loss += loss.item()
                pred = outputs.logits.argmax(dim=1)
                total_correct += pred.eq(target).sum().item()
                total_samples += target.size(0)
        
        avg_loss = total_loss / len(self.test_loader)
        accuracy = 100. * total_correct / total_samples
        
        return avg_loss, accuracy
    
    def train(self):
        """Run federated learning rounds"""
        print(f"Starting Federated Learning on {self.config.device}")
        print(f"Number of clients: {self.config.num_clients}")
        print(f"Noise variance (uniform): {self.config.noise_variance}")
        
        for round_idx in range(self.config.num_rounds):
            print(f"\n{'='*60}")
            print(f"Round {round_idx + 1}/{self.config.num_rounds}")
            print(f"{'='*60}")
            
            # Client training
            client_lora_params = []
            client_losses = []
            client_accuracies = []
            
            for client in self.clients:
                print(f"Training Client {client.client_id} (noise_var={client.noise_variance})...")
                loss, acc = client.train(self.global_model)
                client_lora_params.append(client.get_lora_parameters())
                client_losses.append(loss)
                client_accuracies.append(acc)
                
                print(f"  Loss: {loss:.4f}, Accuracy: {acc:.2f}%")
            
            # Aggregate LoRA parameters
            print("\nAggregating LoRA parameters...")
            self.aggregate_lora_parameters(client_lora_params)
            
            # Evaluate global model
            print("Evaluating global model...")
            test_loss, test_acc = self.evaluate()
            
            print(f"\nGlobal Model Performance:")
            print(f"  Test Loss: {test_loss:.4f}")
            print(f"  Test Accuracy: {test_acc:.2f}%")
            
            # Log to W&B
            wandb_log = {
                "round": round_idx + 1,
                "global_test_loss": test_loss,
                "global_test_accuracy": test_acc,
                "avg_client_train_loss": np.mean(client_losses),
                "avg_client_train_accuracy": np.mean(client_accuracies),
            }
            
            # Log individual client metrics
            for i, (loss, acc) in enumerate(zip(client_losses, client_accuracies)):
                wandb_log[f"client_{i}_train_loss"] = loss
                wandb_log[f"client_{i}_train_accuracy"] = acc
            
            wandb.log(wandb_log)
        
        print("\n" + "="*60)
        print("Federated Learning Complete!")
        print("="*60)

def run_experiment(noise_variance):
    """Run a single experiment with specified noise variance"""
    print(f"\n{'#'*80}")
    print(f"# Starting Experiment with Noise Variance = {noise_variance}")
    print(f"{'#'*80}\n")
    
    # Create config for this experiment
    config = Config()
    config.noise_variance = noise_variance
    
    # Set seed for reproducibility
    set_seed(config.seed)
    print(f"Random seed set to: {config.seed}")
    
    # Initialize W&B with unique run name
    wandb.init(
        project=config.wandb_project,
        name=f"noise_var_{noise_variance}",
        config={
            "num_clients": config.num_clients,
            "num_rounds": config.num_rounds,
            "local_epochs": config.local_epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "noise_variance": config.noise_variance,
            "dirichlet_alpha": config.dirichlet_alpha,
            "seed": config.seed,
        },
        reinit=True
    )
    
    # Load CIFAR-100
    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    # Create client datasets
    client_datasets = create_client_datasets(trainset, config.num_clients, alpha=config.dirichlet_alpha)
    test_loader = DataLoader(testset, batch_size=config.batch_size, shuffle=False)
    
    # Run federated learning
    server = FederatedServer(config, client_datasets, test_loader)
    server.train()
    
    # Save final model
    model_path = f"federated_vit_lora_noise_{noise_variance}.pt"
    torch.save(server.global_model.state_dict(), model_path)
    print(f"\nModel saved to '{model_path}'")
    
    wandb.finish()

# Run federated learning with different noise levels
if __name__ == "__main__":
    # Different noise variances to test
    noise_levels = [0.1]
    
    for noise_var in noise_levels:
        run_experiment(noise_var)
    
    print("\n" + "="*80)
    print("All experiments completed!")
    print("="*80)