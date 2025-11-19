"""
Unlearning implementation for deepfake detection with poisoned data
Adapted from https://arxiv.org/pdf/2312.04095

Versione corretta:
- SVD: uso di V (feature space) e varianza con S^2
- Proiezione dei gradienti coerente con la dimensione dei parametri (out_features / out_channels)
- Unlearning loss per classificazione binaria (1 logit + sigmoid)
- evaluate() coerente con output binario
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import copy
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
from collections import defaultdict

import dataset as ImportDataset
from networks import get_network
from parser import get_parser


class UnlearningTrainer:
    """
    Implements machine unlearning for removing poisoned samples
    using SVD-based projection and gradient manipulation
    (versione corretta e coerente per binario con 1 logit).
    """
    
    def __init__(self, settings, model, device):
        self.settings = settings
        self.model = model
        self.device = device
        
        # Unlearning hyperparameters
        self.retained_var = getattr(settings, 'retained_var', 0.95)
        self.start_lr = getattr(settings, 'unlearn_lr', 0.01)
        self.end_lr = getattr(settings, 'unlearn_end_lr', 0.001)
        self.num_epochs = getattr(settings, 'unlearn_epochs', 50)
        self.offset = getattr(settings, 'unlearn_offset', 0.1)
        self.loss1_weight = getattr(settings, 'loss1_weight', 1.0)
        self.loss2_weight = getattr(settings, 'loss2_weight', 1.0)
        self.wd = getattr(settings, 'weight_decay', 0.0)
        
        # SVD computation
        self.projection_matrices = {}
        self.layer_names = []
    
    # -------------------------------------------------------------------------
    # FEATURE EXTRACTION
    # -------------------------------------------------------------------------
    def extract_features(self, model, dataloader, max_samples=2000):
        """
        Extract feature activations from intermediate layers
        for SVD computation (memory-efficient version)

        N.B.: si assume che il modello sia tipo ResNet:
        - usiamo l'ultima conv (layer4.2.*)
        - e il layer fc
        """
        model.eval()
        
        # Only use final conv layers and FC layer to save memory
        target_layers = []
        for name, module in model.named_modules():
            # Only keep layer4.2 convs and fc
            if 'layer4.2' in name and isinstance(module, nn.Conv2d):
                target_layers.append(name)
            elif name == 'fc' and isinstance(module, nn.Linear):
                target_layers.append(name)
        
        if not target_layers:
            # Fallback: just use fc layer
            for name, module in model.named_modules():
                if name == 'fc':
                    target_layers.append(name)
        
        self.layer_names = target_layers
        print(f"Selected layers for SVD: {target_layers}")
        
        # Extract features layer by layer to save memory
        features_dict = {}
        
        for layer_name in target_layers:
            print(f"Extracting features from {layer_name}...")
            activations = []
            hooks = []
            
            def get_activation(name):
                def hook(model, input, output):
                    # Store flattened output
                    if len(output.shape) == 4:  # Conv layer (B, C, H, W)
                        # Global average pooling to reduce size
                        feat = F.adaptive_avg_pool2d(output, (1, 1))
                        feat = feat.view(feat.size(0), -1).detach().cpu()
                        activations.append(feat)
                    elif len(output.shape) == 2:  # Linear layer (B, F)
                        feat = output.detach().cpu()
                        activations.append(feat)
                return hook
            
            # Register hook only for current layer
            for name, module in model.named_modules():
                if name == layer_name:
                    hook = module.register_forward_hook(get_activation(name))
                    hooks.append(hook)
                    break
            
            # Forward pass to collect activations
            sample_count = 0
            with torch.no_grad():
                pbar = tqdm(dataloader, desc=f"Layer {layer_name}")
                for data, _ in pbar:
                    if max_samples and sample_count >= max_samples:
                        break
                    
                    data = data.to(self.device)
                    _ = model(data)
                    sample_count += data.shape[0]
                    
                    pbar.set_postfix({'samples': sample_count})
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            # Concatenate and store
            if activations:
                feats = torch.cat(activations, dim=0).numpy()
                features_dict[layer_name] = feats
                print(f"  Collected {feats.shape[0]} samples, "
                      f"dim={feats.shape[1]}")
            
            # Clear memory
            activations.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return features_dict
    
    # -------------------------------------------------------------------------
    # SVD E PROIEZIONE
    # -------------------------------------------------------------------------
    def compute_svd(self, features):
        """
        Compute SVD for each layer's feature activations
        Returns SVD data for each layer (using V for feature space).
        """
        svd_results = {}
        
        for layer_name, feat in tqdm(features.items(), desc="Computing SVD"):
            # Center the features
            feat_mean = feat.mean(axis=0, keepdims=True)
            feat_centered = feat - feat_mean
            
            n_samples, n_features = feat_centered.shape
            n_components = min(n_samples, n_features, 512)
            
            print(f"Layer {layer_name}: Computing SVD with up to "
                  f"{n_components} components on shape {feat_centered.shape}...")
            
            try:
                # full_matrices=False -> U:(n_samples, k), Vt:(k, n_features)
                U, S, Vt = np.linalg.svd(feat_centered, full_matrices=False)
                
                # Keep only top components
                if U.shape[1] > n_components:
                    U = U[:, :n_components]
                    S = S[:n_components]
                    Vt = Vt[:n_components, :]
                
                V = Vt.T  # (n_features, n_components)
            
            except np.linalg.LinAlgError as e:
                print(f"SVD failed for {layer_name}, using truncated SVD: {e}")
                from scipy.sparse.linalg import svds
                k = min(n_components, min(feat_centered.shape) - 1)
                U, S, Vt = svds(feat_centered, k=k)
                V = Vt.T
            
            svd_results[layer_name] = {
                'U': torch.from_numpy(U).float(),          # not strictly needed
                'S': torch.from_numpy(S).float(),
                'V': torch.from_numpy(V).float(),          # feature basis
                'mean': torch.from_numpy(feat_mean[0]).float()
            }
            
            print(f"  Shape: {feat.shape}, Singular values: {S.shape[0]}")
            
            # Clear memory
            del U, S, Vt, feat_centered
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return svd_results
    
    def compute_projection_matrices(self, svd_results):
        """
        Compute projection matrices for each layer based on retained variance
        P = V_k @ V_k^T where k is determined by retained variance in S^2.
        
        - V: (n_features, n_components)
        - P: (n_features, n_features)
        - Verrà applicata ai gradienti lungo la dimensione out_features/out_channels.
        """
        projection_matrices = {}
        
        for layer_name, svd_data in svd_results.items():
            S = svd_data['S']     # (n_components,)
            V = svd_data['V']     # (n_features, n_components)
            
            # Varianza spiegata ~ S^2
            var = S ** 2
            total_var = torch.sum(var)
            cumsum_var = torch.cumsum(var, dim=0) / (total_var + 1e-12)
            
            # Find k such that cumulative variance >= retained_var
            k = int(torch.sum(cumsum_var < self.retained_var).item()) + 1
            k = max(k, 1)  # At least 1 component
            k = min(k, V.shape[1])
            
            # Get top-k right singular vectors (feature subspace)
            V_k = V[:, :k]  # (n_features, k)
            
            # Projection matrix P = V_k @ V_k^T
            P = torch.mm(V_k, V_k.t()).to(self.device)  # (n_features, n_features)
            
            projection_matrices[layer_name] = P
            
            variance_retained = cumsum_var[k-1].item()
            print(f"Layer {layer_name}: k={k}/{S.shape[0]} "
                  f"({100.0 * k / S.shape[0]:.2f}%), "
                  f"variance={variance_retained:.4f}")
        
        self.projection_matrices = projection_matrices
        return projection_matrices
    
    # -------------------------------------------------------------------------
    # POISONED INDICES E DATA SPLIT
    # -------------------------------------------------------------------------
    def prepare_unlearning_data(self, train_set, poison_indices):
        """
        Split training data into clean (retain) and poisoned (forget) sets
        """
        all_indices = set(range(len(train_set)))
        poison_set_indices = set(poison_indices)
        clean_indices = list(all_indices - poison_set_indices)
        poison_indices_list = list(poison_set_indices)
        
        clean_set = Subset(train_set, clean_indices)
        poison_set = Subset(train_set, poison_indices_list)
        
        print(f"Clean samples: {len(clean_set)}")
        print(f"Poisoned samples: {len(poison_set)}")
        
        return clean_set, poison_set
    
    def get_poisoned_indices(self, train_set):
        """
        Identify which samples were poisoned during training
        Requires tracking from the dataset or reproduce same random logic.
        """
        poison_indices = []
        
        # Check if dataset has poison tracking
        if hasattr(train_set, 'samples'):
            # Reconstruct which samples were poisoned
            # This requires the same random seed used during poisoning
            random_state = np.random.RandomState(self.settings.poison_seed)
            num_samples = len(train_set.samples)
            num_poisoned = int(num_samples * self.settings.poison_ratio)
            poison_indices = random_state.choice(
                num_samples, 
                size=num_poisoned, 
                replace=False
            ).tolist()
        
        return poison_indices
    
    # -------------------------------------------------------------------------
    # UNLEARNING LOSS (BINARIA, 1 LOGIT)
    # -------------------------------------------------------------------------
    def unlearning_loss(self, outputs, labels):
        """
        Custom loss for unlearning poisoned samples (binary, 1 logit).

        outputs: logits (B, 1)
        labels:  0/1 ma qui non le usiamo direttamente: vogliamo portare
                 il modello verso incertezza (p ~ 0.5) sui campioni poisoned.

        Loss = L1 + L2
        - L1: penalizza la distanza da 0.5 (con un margine offset)
        - L2: massimizza l'entropia della Bernoulli(p)
        """
        outputs = outputs.view(-1)           # (B,)
        probs = torch.sigmoid(outputs)      # (B,)
        
        batch_size = probs.shape[0]
        
        # L1: penalizza quando |p - 0.5| è più grande di offset
        diff = torch.abs(probs - 0.5)
        margin_violation = torch.clamp(diff - self.offset, min=0.0)
        loss1 = torch.mean(margin_violation)
        
        # L2: massimizza entropia di Bernoulli(p)
        entropy = -(
            probs * torch.log(probs + 1e-10) +
            (1.0 - probs) * torch.log(1.0 - probs + 1e-10)
        )  # (B,)
        loss2 = -torch.mean(entropy)  # negativo perché vogliamo max entropy
        
        total_loss = self.loss1_weight * loss1 + self.loss2_weight * loss2
        
        return total_loss, loss1, loss2
    
    # -------------------------------------------------------------------------
    # PROIEZIONE DEI GRADIENTI
    # -------------------------------------------------------------------------
    def projected_gradient_update(self, model, projection_matrices):
        """
        Update model parameters using projected gradients.

        g_clean_subspace = P g
        g_orth           = g - g_clean_subspace

        Qui manteniamo solo la componente ortogonale al sottospazio
        delle feature "clean" per evitare di cancellare troppo il clean.
        """
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is None:
                    continue
                
                # Trova il layer corrispondente
                P_name = self._match_layer_name(name)
                
                if P_name is None or P_name not in projection_matrices:
                    # Nessuna proiezione definita per questo parametro: lascialo com'è
                    # (se vuoi azzerare gradienti dei layer non considerati, decommenta)
                    # param.grad.data.zero_()
                    continue
                
                P = projection_matrices[P_name]  # (n_features, n_features)
                
                # Dimensione lungo cui proiettiamo: out_features / out_channels
                sz = param.grad.data.shape[0]
                
                # Add weight decay (L2)
                reg_grad = param.grad.data.add(param.data, alpha=self.wd)
                
                # Flatten: (sz, rest)
                grad_flat = reg_grad.view(sz, -1)
                
                # Compatibilità dimensionale
                if P.shape[0] != sz:
                    # Mismatch dimensionale: non proiettiamo questo parametro
                    # (debug) print(f"Warning: P.shape[0]={P.shape[0]} != {sz} for {name}")
                    continue
                
                # Proiezione nel sottospazio clean: P g
                proj = torch.mm(P, grad_flat)     # (sz, rest)
                # Componente ortogonale: g - P g
                grad_orth = grad_flat - proj      # (sz, rest)
                
                param.grad.data = grad_orth.view_as(param.grad.data)
    
    def _match_layer_name(self, param_name):
        """
        Match parameter name to layer name used in feature extraction.
        Esempio:
        - param_name: 'layer4.2.conv3.weight'
        - layer_name: 'layer4.2.conv3'
        """
        # Remove .weight or .bias suffix
        base_name = param_name.replace('.weight', '').replace('.bias', '')
        
        # Try exact match first
        if base_name in self.layer_names:
            return base_name
        
        # Try partial match
        for layer_name in self.layer_names:
            if layer_name in base_name or base_name in layer_name:
                return layer_name
        
        return None
    
    # -------------------------------------------------------------------------
    # MAIN UNLEARNING LOOP
    # -------------------------------------------------------------------------
    def unlearn(self, train_set, val_set, poison_indices=None):
        """
        Main unlearning procedure
        
        1. Identify clean (retain) and poisoned (forget) data
        2. Compute SVD on clean data
        3. Compute projection matrices
        4. Train with projected gradients on poisoned data
        """
        
        # Setup data loaders
        if poison_indices is None:
            poison_indices = self.get_poisoned_indices(train_set)
        
        clean_set, poison_set = self.prepare_unlearning_data(
            train_set, poison_indices
        )
        
        clean_loader = DataLoader(
            clean_set, 
            batch_size=self.settings.batch_size,
            shuffle=False, 
            num_workers=4
        )
        
        poison_loader = DataLoader(
            poison_set,
            batch_size=max(self.settings.batch_size // 2, 1),
            shuffle=True,
            num_workers=4
        )
        
        val_loader = DataLoader(
            val_set,
            batch_size=self.settings.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        # Step 1: Extract features from clean data
        print("\n" + "="*60)
        print("Step 1: Extracting features from clean data")
        print("="*60)
        clean_features = self.extract_features(
            self.model, 
            clean_loader,
            max_samples=2000  # Reduced for memory efficiency
        )
        
        # Step 2: Compute SVD
        print("\n" + "="*60)
        print("Step 2: Computing SVD")
        print("="*60)
        svd_results = self.compute_svd(clean_features)
        
        # Step 3: Compute projection matrices
        print("\n" + "="*60)
        print("Step 3: Computing projection matrices")
        print("="*60)
        projection_matrices = self.compute_projection_matrices(svd_results)
        
        # Step 4: Unlearning training
        print("\n" + "="*60)
        print("Step 4: Unlearning training")
        print("="*60)
        
        # Freeze batch norm statistics
        def freeze_bn(model):
            for module in model.modules():
                if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    module.eval()
        
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.start_lr,
            momentum=0.0,
            weight_decay=0.0
        )
        
        # Learning rate schedule (esponenziale)
        alpha = np.exp(np.log(self.end_lr / self.start_lr) / max(self.num_epochs, 1))
        current_lr = self.start_lr
        
        # Track metrics
        history = {
            'epoch': [],
            'train_loss': [],
            'val_acc': [],
            'poison_acc': [],
            'clean_acc': []
        }
        
        for epoch in range(self.num_epochs):
            # Training phase
            self.model.train()
            freeze_bn(self.model)
            
            epoch_loss = 0.0
            epoch_loss1 = 0.0
            epoch_loss2 = 0.0
            
            # Epoch 0: solo evaluation (come nel tuo codice originale)
            if epoch > 0 and len(poison_loader) > 0:
                pbar = tqdm(poison_loader, desc=f"Epoch {epoch}/{self.num_epochs}")
                for data, labels in pbar:
                    data = data.to(self.device)
                    labels = labels.to(self.device)
                    
                    optimizer.zero_grad()
                    
                    outputs = self.model(data)
                    if isinstance(outputs, tuple):
                        _, outputs = outputs
                    
                    # outputs expected shape: (B, 1)
                    loss, loss1, loss2 = self.unlearning_loss(outputs, labels)
                    
                    loss.backward()
                    
                    # Apply projected gradient update
                    self.projected_gradient_update(
                        self.model, 
                        projection_matrices
                    )
                    
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    epoch_loss1 += loss1.item()
                    epoch_loss2 += loss2.item()
                    
                    pbar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'lr': f"{current_lr:.6f}"
                    })
                
                # Update learning rate
                current_lr *= alpha
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
            
            # Evaluation phase
            print(f"\nEpoch {epoch} Evaluation:")
            val_metrics = self.evaluate(val_loader, "Validation")
            clean_metrics = self.evaluate(clean_loader, "Clean Train")
            poison_metrics = self.evaluate(poison_loader, "Poison Train")
            
            # Store history
            history['epoch'].append(epoch)
            denom = max(len(poison_loader), 1)
            history['train_loss'].append(epoch_loss / denom)
            history['val_acc'].append(val_metrics['accuracy'])
            history['clean_acc'].append(clean_metrics['accuracy'])
            history['poison_acc'].append(poison_metrics['accuracy'])
            
            print(f"Val Acc: {val_metrics['accuracy']:.2f}% | "
                  f"Clean Acc: {clean_metrics['accuracy']:.2f}% | "
                  f"Poison Acc: {poison_metrics['accuracy']:.2f}%")
            
            # Early stopping condition:
            # se la accuracy su validation cala troppo rispetto alla prima
            if len(history['val_acc']) > 1:
                acc_drop = history['val_acc'][0] - history['val_acc'][-1]
                if acc_drop > 5.0:  # More than 5% drop
                    print(f"\nEarly stopping: validation accuracy dropped by {acc_drop:.2f}%")
                    break
            
            print("-" * 60)
        
        return history
    
    # -------------------------------------------------------------------------
    # EVALUATION (BINARIA)
    # -------------------------------------------------------------------------
    def evaluate(self, loader, name="Test"):
        """Evaluate model on given data loader (binary, 1 logit)."""
        self.model.eval()
        
        all_labels = []
        all_preds = []
        
        if len(loader) == 0:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'confusion_matrix': np.zeros((2, 2), dtype=int)
            }
        
        with torch.no_grad():
            for data, labels in loader:
                data = data.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(data)
                if isinstance(outputs, tuple):
                    _, outputs = outputs
                
                # Binary logits -> probabilities
                probs = torch.sigmoid(outputs.view(-1))
                preds = (probs >= 0.5).long()
                
                all_labels.extend(labels.view(-1).cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
        
        all_labels = np.array(all_labels).flatten()
        all_preds = np.array(all_preds).flatten()
        
        # Compute metrics
        accuracy = 100.0 * np.mean(all_labels == all_preds) if len(all_labels) > 0 else 0.0
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0
        )
        
        cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
        
        metrics = {
            'accuracy': accuracy,
            'precision': 100.0 * precision,
            'recall': 100.0 * recall,
            'f1': 100.0 * f1,
            'confusion_matrix': cm
        }
        
        return metrics


# -----------------------------------------------------------------------------    
# PLOTTING
# -----------------------------------------------------------------------------
def plot_unlearning_results(history, save_path):
    """Plot unlearning training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy curves
    axes[0].plot(history['epoch'], history['val_acc'], 'b-', label='Validation', linewidth=2)
    axes[0].plot(history['epoch'], history['clean_acc'], 'g-', label='Clean Train', linewidth=2)
    axes[0].plot(history['epoch'], history['poison_acc'], 'r-', label='Poison Train', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Unlearning Progress')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss curve (skippa epoch 0 se è solo evaluation)
    if len(history['epoch']) > 1:
        axes[1].plot(history['epoch'][1:], history['train_loss'][1:], 'k-', linewidth=2)
    else:
        axes[1].plot(history['epoch'], history['train_loss'], 'k-', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Unlearning Loss')
    axes[1].set_title('Training Loss')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plots saved to {save_path}")


# -----------------------------------------------------------------------------    
# MAIN
# -----------------------------------------------------------------------------
def main():
    parser = get_parser()
    
    # Add unlearning-specific arguments
    parser.add_argument("--unlearn", action='store_true', 
                       help="Perform unlearning on poisoned model")
    parser.add_argument("--retained_var", type=float, default=0.95,
                       help="Variance retained in SVD projection")
    parser.add_argument("--unlearn_lr", type=float, default=0.01,
                       help="Learning rate for unlearning")
    parser.add_argument("--unlearn_end_lr", type=float, default=0.001,
                       help="End learning rate for unlearning")
    parser.add_argument("--unlearn_epochs", type=int, default=50,
                       help="Number of unlearning epochs")
    parser.add_argument("--unlearn_offset", type=float, default=0.1,
                       help="Offset (margin) for unlearning loss")
    parser.add_argument("--loss1_weight", type=float, default=1.0,
                       help="Weight for loss component 1")
    parser.add_argument("--loss2_weight", type=float, default=1.0,
                       help="Weight for loss component 2")
    
    settings = parser.parse_args()
    
    if not settings.unlearn:
        print("Please use --unlearn flag to perform unlearning")
        return
    
    # Setup paths
    if getattr(settings, 'save_id', None) is None:
        print("Error: --save_id required for unlearning")
        return
    
    savedir = os.path.join('runs', settings.name, settings.save_id)
    os.makedirs(savedir, exist_ok=True)
    
    # Set device
    device = torch.device(settings.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load datasets
    print("\nLoading datasets...")
    train_set = ImportDataset.LoaderDatasetSplit(
        settings, 
        os.path.join(settings.split_path, 'train.json')
    )
    val_set = ImportDataset.LoaderDatasetSplit(
        settings,
        os.path.join(settings.split_path, 'val.json')
    )
    
    # Load poisoned model
    print("\nLoading poisoned model...")
    model = get_network(settings)
    model.to(device)
    
    if getattr(settings, 'load_id', None) is None:
        print("Error: --load_id required to load poisoned model")
        return
    
    loaddir = os.path.join('runs', settings.name, settings.load_id)
    checkpoint_path = os.path.join(loaddir, 'checkpoints', 'best.pt')
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {checkpoint_path}")
    
    # Initialize unlearning trainer
    print("\nInitializing unlearning trainer...")
    trainer = UnlearningTrainer(settings, model, device)
    
    # Perform unlearning
    print("\n" + "="*60)
    print("Starting Unlearning Process")
    print("="*60)
    
    history = trainer.unlearn(train_set, val_set)
    
    # Save results
    unlearn_dir = os.path.join(savedir, 'unlearning')
    os.makedirs(unlearn_dir, exist_ok=True)
    
    # Save unlearned model
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'settings': vars(settings)
    }, os.path.join(unlearn_dir, 'unlearned_model.pt'))
    
    # Save history (usa np.save con allow_pickle perché history è dict)
    np.save(os.path.join(unlearn_dir, 'history.npy'), history, allow_pickle=True)
    
    # Plot results
    plot_unlearning_results(
        history, 
        os.path.join(unlearn_dir, 'unlearning_curves.png')
    )
    
    # Final evaluation
    print("\n" + "="*60)
    print("Final Evaluation")
    print("="*60)
    
    val_loader = DataLoader(
        val_set,
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    final_metrics = trainer.evaluate(val_loader, "Final Validation")
    
    print(f"\n### Final Performance ###")
    print(f"Accuracy: {final_metrics['accuracy']:.2f}%")
    print(f"Precision: {final_metrics['precision']:.2f}%")
    print(f"Recall: {final_metrics['recall']:.2f}%")
    print(f"F1 Score: {final_metrics['f1']:.2f}%")
    print(f"\nConfusion Matrix:")
    print(f"             Real  Fake")
    print(f"Real  {final_metrics['confusion_matrix'][0,0]:7d} {final_metrics['confusion_matrix'][0,1]:5d}")
    print(f"Fake  {final_metrics['confusion_matrix'][1,0]:7d} {final_metrics['confusion_matrix'][1,1]:5d}")
    
    # Save final report
    with open(os.path.join(unlearn_dir, 'final_report.txt'), 'w') as f:
        f.write("="*60 + "\n")
        f.write("Unlearning Final Report\n")
        f.write("="*60 + "\n\n")
        f.write(f"Poison ratio: {settings.poison_ratio}\n")
        f.write(f"Retained variance: {settings.retained_var}\n")
        f.write(f"Unlearning epochs: {len(history['epoch'])}\n\n")
        f.write(f"Final Accuracy: {final_metrics['accuracy']:.2f}%\n")
        f.write(f"Final Precision: {final_metrics['precision']:.2f}%\n")
        f.write(f"Final Recall: {final_metrics['recall']:.2f}%\n")
        f.write(f"Final F1 Score: {final_metrics['f1']:.2f}%\n")
    
    print(f"\nResults saved to {unlearn_dir}")


if __name__ == "__main__":
    main()
