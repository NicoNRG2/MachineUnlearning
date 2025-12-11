"""
Utility functions for unlearning phase
Adapted from: https://github.com/hnanhtuan/projected_gradient_unlearning
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def freeze_norm_stats(module):
    """Freeze batch normalization statistics during unlearning"""
    if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
        module.eval()


def evaluate_binary(model, data_loader, description='', displayer=print):
    """
    Evaluate binary classification model (Real vs Fake)
    
    Args:
        model: PyTorch model
        data_loader: DataLoader for evaluation
        description: Description string for logging
        displayer: Function to display results (default: print)
    
    Returns:
        (loss, accuracy) tuple
    """
    model.eval()
    
    criterion = nn.BCEWithLogitsLoss()
    
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        with tqdm(data_loader, desc=description, unit='batch') as pbar:
            for images, labels in pbar:
                images = images.cuda()
                labels = labels.cuda()
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item() * images.size(0)
                
                # Binary prediction: sigmoid(output) > 0.5
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                
                pbar.set_postfix({
                    'loss': f'{total_loss/total:.4f}',
                    'acc': f'{100*correct/total:.2f}%'
                })
    
    avg_loss = total_loss / total
    accuracy = 100 * correct / total
    
    displayer(f'{description} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    return avg_loss, accuracy


def compute_svd_frozen_model(model, data_loader, printer=print):
    """
    Compute SVD optimized for frozen models (only FC layer trained)
    Uses features from penultimate layer (avgpool output)
    
    Args:
        model: ResNet50 model
        data_loader: DataLoader for computing activations
        printer: Function for logging
    
    Returns:
        Dictionary with SVD decomposition for FC layer
    """
    model.eval()
    
    printer("Computing SVD for frozen model (feature-based approach)")
    printer("Extracting features from penultimate layer...")
    
    # Collect features from avgpool (input to FC layer)
    features_list = []
    
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(tqdm(data_loader, desc='Extracting features')):
            images = images.cuda()
            
            # Forward pass up to avgpool
            x = model.conv1(images)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)
            
            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)
            
            x = model.avgpool(x)
            x = torch.flatten(x, 1)  # Features before FC: (batch, 2048)
            
            features_list.append(x.cpu())
            
            if batch_idx >= 100:  # Limit for memory
                break
    
    # Concatenate all features
    features = torch.cat(features_list, dim=0)  # (n_samples, 2048)
    printer(f'Collected features shape: {features.shape}')
    
    # Compute SVD on features
    printer('Computing SVD on features...')
    try:
        U, S, V = torch.svd(features.t())  # Transpose: (2048, n_samples)
        
        svd_dict = {
            'fc': {
                'U': U.cpu(),  # (2048, 2048)
                'S': S.cpu(),  # (min(2048, n_samples),)
                'V': V.cpu()   # (n_samples, n_samples)
            }
        }
        
        printer(f'fc: SVD shapes - U: {U.shape}, S: {S.shape}, V: {V.shape}')
        
        # Print variance explained
        total_var = torch.sum(S ** 2)
        cumsum_var = torch.cumsum(S ** 2, dim=0) / total_var
        k_90 = torch.sum(cumsum_var < 0.90).item()
        k_95 = torch.sum(cumsum_var < 0.95).item()
        k_99 = torch.sum(cumsum_var < 0.99).item()
        
        printer(f'fc: Components for 90%/95%/99% variance: {k_90}/{k_95}/{k_99} out of {len(S)}')
        
        return svd_dict
        
    except Exception as e:
        printer(f'Error computing SVD: {e}')
        return {}


def compute_svd_resnet(model, data_loader, conv_layers=None, fc_layers=None, frozen=True, printer=print):
    """
    Compute SVD for ResNet50 layers
    
    Args:
        model: ResNet50 model
        data_loader: DataLoader for computing activations
        conv_layers: List of convolutional layer names (optional)
        fc_layers: List of fully connected layer names (optional)
        frozen: If True, only compute SVD on trainable layers (default: True)
        printer: Function for logging
    
    Returns:
        Dictionary with SVD decomposition for each layer
    """
    model.eval()
    
    # If model is frozen, only track the final FC layer (much more efficient!)
    if frozen:
        printer("Model was trained with --freeze, computing SVD only on FC layer")
        conv_layers = []  # Skip all conv layers
        fc_layers = ['fc'] if fc_layers is None else fc_layers
    else:
        # Default layers to track in ResNet50
        if conv_layers is None:
            conv_layers = [
                'layer1[0].conv1', 'layer1[0].conv2', 'layer1[0].conv3',
                'layer2[0].conv1', 'layer2[0].conv2', 'layer2[0].conv3',
                'layer3[0].conv1', 'layer3[0].conv2', 'layer3[0].conv3',
                'layer4[0].conv1', 'layer4[0].conv2', 'layer4[0].conv3',
            ]
        
        if fc_layers is None:
            fc_layers = ['fc']
    
    all_layers = conv_layers + fc_layers
    
    # Dictionary to store gradients
    activations = {layer: [] for layer in all_layers}
    
    def get_activation(name):
        def hook(module, input, output):
            # For convolutional layers, reshape to (batch, features)
            if isinstance(module, nn.Conv2d):
                # output shape: (batch, channels, height, width)
                batch_size = output.size(0)
                output_flat = output.view(batch_size, -1)
                activations[name].append(output_flat.detach().cpu())
            elif isinstance(module, nn.Linear):
                activations[name].append(output.detach().cpu())
        return hook
    
    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        # Convert name format: layer1.0.conv1 -> layer1[0].conv1
        formatted_name = name
        for i in range(10):
            formatted_name = formatted_name.replace(f'.{i}.', f'[{i}].')
        
        if formatted_name in all_layers:
            hook = module.register_forward_hook(get_activation(formatted_name))
            hooks.append(hook)
            printer(f'Registered hook for: {formatted_name}')
    
    # Forward pass to collect activations
    printer('Computing activations...')
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(tqdm(data_loader, desc='SVD computation')):
            images = images.cuda()
            _ = model(images)
            
            if batch_idx >= 100:  # Limit number of batches for SVD
                break
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Compute SVD for each layer
    svd_dict = {}
    printer('\nComputing SVD decomposition...')
    
    for layer_name in all_layers:
        if len(activations[layer_name]) == 0:
            printer(f'Warning: No activations for {layer_name}')
            continue
        
        # Concatenate all activations
        A = torch.cat(activations[layer_name], dim=0)  # (total_samples, features)
        printer(f'{layer_name}: Activation matrix shape: {A.shape}')
        
        # Compute SVD: A = U @ diag(S) @ V^T
        try:
            U, S, V = torch.svd(A.t())  # Transpose to get (features, samples)
            
            svd_dict[layer_name] = {
                'U': U.cpu(),  # (features, features)
                'S': S.cpu(),  # (min(features, samples),)
                'V': V.cpu()   # (samples, samples)
            }
            
            printer(f'{layer_name}: SVD shapes - U: {U.shape}, S: {S.shape}, V: {V.shape}')
            
            # Print variance explained
            total_var = torch.sum(S ** 2)
            cumsum_var = torch.cumsum(S ** 2, dim=0) / total_var
            k_90 = torch.sum(cumsum_var < 0.90).item()
            k_95 = torch.sum(cumsum_var < 0.95).item()
            k_99 = torch.sum(cumsum_var < 0.99).item()
            
            printer(f'{layer_name}: Components for 90%/95%/99% variance: {k_90}/{k_95}/{k_99} out of {len(S)}')
            
        except Exception as e:
            printer(f'Error computing SVD for {layer_name}: {e}')
            continue
    
    return svd_dict


def compute_retain_svd(full_svd, model, forget_loader, conv_layers=None, fc_layers=None, printer=print):
    """
    Update SVD by removing forget set contributions (incremental SVD)
    
    Args:
        full_svd: SVD computed on full dataset
        model: Current model
        forget_loader: DataLoader for samples to forget
        conv_layers: List of convolutional layer names
        fc_layers: List of fully connected layer names
        printer: Logging function
    
    Returns:
        Updated SVD dictionary
    """
    printer('Computing forget set activations...')
    
    model.eval()
    
    if conv_layers is None:
        conv_layers = list(full_svd.keys())
    
    # Get forget activations
    forget_activations = {layer: [] for layer in full_svd.keys()}
    
    def get_activation(name):
        def hook(module, input, output):
            if isinstance(module, nn.Conv2d):
                batch_size = output.size(0)
                output_flat = output.view(batch_size, -1)
                forget_activations[name].append(output_flat.detach().cpu())
            elif isinstance(module, nn.Linear):
                forget_activations[name].append(output.detach().cpu())
        return hook
    
    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        formatted_name = name
        for i in range(10):
            formatted_name = formatted_name.replace(f'.{i}.', f'[{i}].')
        
        if formatted_name in full_svd.keys():
            hook = module.register_forward_hook(get_activation(formatted_name))
            hooks.append(hook)
    
    # Collect forget activations
    with torch.no_grad():
        for images, _ in tqdm(forget_loader, desc='Forget activations'):
            images = images.cuda()
            _ = model(images)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Update SVD for each layer
    retain_svd = {}
    
    for layer_name in full_svd.keys():
        if len(forget_activations[layer_name]) == 0:
            # No forget data, keep original SVD
            retain_svd[layer_name] = full_svd[layer_name].copy()
            continue
        
        # Get forget activation matrix
        A_forget = torch.cat(forget_activations[layer_name], dim=0).t()  # (features, n_forget)
        
        printer(f'{layer_name}: Removing {A_forget.shape[1]} samples from SVD')
        
        # Simple approach: recompute SVD excluding forget contributions
        # For more efficient incremental SVD, see: https://arxiv.org/abs/2006.10086
        
        U_full = full_svd[layer_name]['U']
        S_full = full_svd[layer_name]['S']
        
        # Project forget activations onto current space
        A_forget_proj = torch.mm(U_full.t(), A_forget)  # (features, n_forget)
        
        # Update singular values (approximate)
        # This is a simplified version - for exact incremental SVD see paper
        S_update = torch.norm(A_forget_proj, dim=1)
        S_retain = torch.clamp(S_full - S_update[:len(S_full)], min=0)
        
        # Normalize
        S_retain = S_retain / torch.sum(S_retain) * torch.sum(S_full)
        
        retain_svd[layer_name] = {
            'U': U_full,
            'S': S_retain,
            'V': full_svd[layer_name]['V']
        }
        
        printer(f'{layer_name}: Updated singular values range: [{S_retain.min():.4f}, {S_retain.max():.4f}]')
    
    return retain_svd


def get_projection_matrices(svd_dict, retained_variance=0.95, device='cuda'):
    """
    Compute projection matrices from SVD for each layer
    
    Args:
        svd_dict: Dictionary with SVD decomposition
        retained_variance: Variance threshold (0-1)
        device: Device to place matrices on
    
    Returns:
        Dictionary of projection matrices
    """
    P = {}
    
    for layer_name, svd in svd_dict.items():
        S = svd['S']
        U = svd['U']
        
        # Skip if no meaningful components
        if len(S) == 0 or S.sum() == 0:
            print(f'{layer_name}: Skipping (no variance)')
            continue
        
        # Find number of components to retain
        cumsum_var = torch.cumsum(S ** 2, dim=0) / torch.sum(S ** 2)
        k = torch.sum(cumsum_var < retained_variance).item()
        
        # Ensure at least 1 component
        k = max(1, min(k, len(S)))
        
        # For very small layers (like FC with 1 output), use all components
        if len(S) <= 10:
            k = len(S)
            print(f'{layer_name}: Small layer, using all {k} components')
        
        # Create projection matrix: P = U_k @ U_k^T
        U_k = U[:, :k].to(device).float()
        
        # Check memory requirement before allocation
        memory_required = U_k.shape[0] * U_k.shape[0] * 4 / (1024**3)  # GB
        if memory_required > 10:  # If more than 10GB
            print(f'{layer_name}: Projection matrix would require {memory_required:.2f}GB, skipping')
            continue
        
        P[layer_name] = torch.mm(U_k, U_k.t()).float()
        
        print(f'{layer_name}: Using {k}/{len(S)} components ({100*k/len(S):.2f}%) for {retained_variance*100:.1f}% variance')
        print(f'  Projection matrix shape: {P[layer_name].shape} ({P[layer_name].numel()*4/(1024**2):.2f}MB)')
    
    return P


def get_entropy(model, data_loader):
    """
    Compute prediction entropy for samples
    
    Args:
        model: PyTorch model
        data_loader: DataLoader
    
    Returns:
        List of entropy values
    """
    model.eval()
    entropies = []
    
    with torch.no_grad():
        for images, _ in data_loader:
            images = images.cuda()
            outputs = model(images)
            
            # For binary classification
            probs = torch.sigmoid(outputs)
            # Entropy: -p*log(p) - (1-p)*log(1-p)
            entropy = -(probs * torch.log(probs + 1e-10) + 
                       (1 - probs) * torch.log(1 - probs + 1e-10))
            
            entropies.extend(entropy.cpu().numpy().flatten())
    
    return np.array(entropies)