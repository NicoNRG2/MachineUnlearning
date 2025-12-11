"""
Unlearning script for TrueFace ResNet50 Real/Fake classifier
Based on: Projected Gradient Unlearning (https://arxiv.org/pdf/2211.00680)

Usage:
    python unlearn_trueface.py --name EXPERIMENT_NAME --poisoned_model PATH --poison_rate 0.20
"""

import os
import sys
import argparse
import json
import pickle as pkl
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import from existing project
import dataset as ImportDataset
from networks import get_network
from parser import get_parser as get_base_parser
from unlearn_utils import (
    AverageMeter, freeze_norm_stats, evaluate_binary,
    compute_svd_resnet, compute_retain_svd, get_projection_matrices,
    get_entropy
)


def get_unlearn_parser():
    """Extended parser for unlearning"""
    parser = argparse.ArgumentParser(description='Unlearning for TrueFace')
    
    # Paths and experiment settings
    parser.add_argument('--name', type=str, required=True, help='Experiment name')
    parser.add_argument('--poisoned_model', type=str, required=True, 
                       help='Path to poisoned model checkpoint (best.pt)')
    parser.add_argument('--baseline_model', type=str, default=None,
                       help='Path to baseline (clean) model for comparison')
    parser.add_argument('--poison_rate', type=float, required=True,
                       help='Poison rate used during training (e.g., 0.20)')
    
    # Dataset settings
    parser.add_argument('--data_root', type=str, required=True, help='Path to dataset')
    parser.add_argument('--split_path', type=str, required=True, help='Path to split files')
    parser.add_argument('--data', type=str, default='gan2:pre&gan3:pre&sdXL:pre&real:pre',
                       help='Dataset specification')
    
    # Model settings
    parser.add_argument('--model', type=str, default='nodown', help='Model architecture')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--freeze', action='store_true', 
                       help='Model was trained with frozen layers (only FC trained)')
    
    # Unlearning hyperparameters
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of unlearning epochs')
    parser.add_argument('--start_lr', type=float, default=0.01, help='Starting learning rate')
    parser.add_argument('--end_lr', type=float, default=0.001, help='Ending learning rate')
    parser.add_argument('--retained_var', type=float, default=0.95, 
                       help='Variance retention threshold for SVD (0-1)')
    parser.add_argument('--offset', type=float, default=0.1, help='Offset for unlearning loss')
    parser.add_argument('--loss1_w', type=float, default=1.0, help='Weight for loss1 (prediction)')
    parser.add_argument('--loss2_w', type=float, default=1.0, help='Weight for loss2 (entropy)')
    parser.add_argument('--wd', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--early_stop_thres', type=float, default=5.0,
                       help='Early stopping if clean accuracy drops more than this')
    
    # Computational settings
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_threads', type=int, default=4, help='Number of data loading threads')
    
    # Augmentation settings (reuse from training)
    parser.add_argument('--resize_prob', type=float, default=0.2)
    parser.add_argument('--resize_scale', type=float, nargs='+', default=[0.2, 1.0])
    parser.add_argument('--resize_ratio', type=float, nargs='+', default=[0.75, 1.33])
    parser.add_argument('--resize_size', type=int, default=512)
    parser.add_argument('--jpeg_prob', type=float, default=0.2)
    parser.add_argument('--jpeg_qual', type=int, nargs='+', default=[30, 100])
    parser.add_argument('--blur_prob', type=float, default=0.2)
    parser.add_argument('--blur_sigma', type=float, nargs='+', default=[1e-6, 3])
    parser.add_argument('--patch_size', type=int, default=96)
    
    # Output settings
    parser.add_argument('--save_model', action='store_true', help='Save model checkpoints')
    parser.add_argument('--output_dir', type=str, default='runs/unlearning', 
                       help='Output directory')
    
    return parser


def load_poisoned_data_info(poison_rate):
    """Load information about poisoned samples"""
    poison_file = f'runs/poison_info/poison_{poison_rate:.2f}.pkl'
    
    if os.path.exists(poison_file):
        with open(poison_file, 'rb') as f:
            poison_info = pkl.load(f)
        return poison_info
    else:
        print(f'Warning: Poison info file not found: {poison_file}')
        return None


def create_unlearn_datasets(args):
    """
    Create datasets for unlearning:
    - Clean (retain): samples that were correctly labeled
    - Poison (forget): samples that were mislabeled
    """
    # Create a modified settings object for dataset creation
    class Settings:
        pass
    
    settings = Settings()
    settings.data_root = args.data_root
    settings.split_path = args.split_path
    settings.data = args.data
    settings.task = 'train'  # Use train transforms
    settings.poison_rate = 0.0  # No poisoning when loading for unlearning
    
    # Copy augmentation settings
    for key in ['resize_prob', 'resize_size', 'resize_scale', 'resize_ratio',
                'jpeg_prob', 'jpeg_qual', 'blur_prob', 'blur_sigma', 'patch_size']:
        setattr(settings, key, getattr(args, key))
    
    # Load full training set (without poisoning)
    train_set_aug = ImportDataset.LoaderDatasetSplit(
        settings, 
        os.path.join(args.split_path, 'train.json')
    )
    
    # For evaluation, use no augmentation
    settings.task = 'test'
    train_set_noaug = ImportDataset.LoaderDatasetSplit(
        settings,
        os.path.join(args.split_path, 'train.json')
    )
    
    # Test set (always clean)
    test_set = ImportDataset.LoaderDatasetSplit(
        settings,
        os.path.join(args.split_path, 'test.json')
    )
    
    # Validation set
    val_set = ImportDataset.LoaderDatasetSplit(
        settings,
        os.path.join(args.split_path, 'val.json')
    )
    
    # Load poison info to separate clean and poisoned samples
    poison_info = load_poisoned_data_info(args.poison_rate)
    
    if poison_info is not None:
        poisoned_indices = set(poison_info['poisoned_indices'])
        
        # Create subsets
        clean_indices = [i for i in range(len(train_set_aug)) if i not in poisoned_indices]
        poison_indices = list(poisoned_indices)
        
        print(f'Dataset split:')
        print(f'  Total training samples: {len(train_set_aug)}')
        print(f'  Clean (retain) samples: {len(clean_indices)}')
        print(f'  Poisoned (forget) samples: {len(poison_indices)}')
    else:
        # If no poison info, assume we want to forget a random subset
        print('No poison info available, using random split for demonstration')
        n_total = len(train_set_aug)
        n_forget = int(n_total * args.poison_rate)
        
        indices = list(range(n_total))
        np.random.shuffle(indices)
        
        poison_indices = indices[:n_forget]
        clean_indices = indices[n_forget:]
        poisoned_indices = set(poison_indices)
    
    # Create data loaders
    from torch.utils.data import Subset
    
    clean_train_aug = Subset(train_set_aug, clean_indices)
    poison_train_aug = Subset(train_set_aug, poison_indices)
    
    clean_train_noaug = Subset(train_set_noaug, clean_indices)
    poison_train_noaug = Subset(train_set_noaug, poison_indices)
    
    loaders = {
        'clean_train_aug': DataLoader(clean_train_aug, batch_size=args.batch_size, 
                                       shuffle=True, num_workers=args.num_threads),
        'poison_train_aug': DataLoader(poison_train_aug, batch_size=args.batch_size,
                                        shuffle=True, num_workers=args.num_threads),
        'clean_train_noaug': DataLoader(clean_train_noaug, batch_size=args.batch_size,
                                         shuffle=False, num_workers=args.num_threads),
        'poison_train_noaug': DataLoader(poison_train_noaug, batch_size=args.batch_size,
                                          shuffle=False, num_workers=args.num_threads),
        'test': DataLoader(test_set, batch_size=args.batch_size,
                          shuffle=False, num_workers=args.num_threads),
        'val': DataLoader(val_set, batch_size=args.batch_size,
                         shuffle=False, num_workers=args.num_threads),
    }
    
    return loaders, poisoned_indices


def main():
    parser = get_unlearn_parser()
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create output directory
    exp_dir = os.path.join(args.output_dir, args.name)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(f'{exp_dir}/checkpoints', exist_ok=True)
    os.makedirs(f'{exp_dir}/plots', exist_ok=True)
    
    # Save configuration
    with open(f'{exp_dir}/config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print(f'\n{"="*80}')
    print(f'UNLEARNING EXPERIMENT: {args.name}')
    print(f'{"="*80}')
    print(f'Poisoned model: {args.poisoned_model}')
    print(f'Poison rate: {args.poison_rate*100:.1f}%')
    print(f'Output directory: {exp_dir}')
    print(f'{"="*80}\n')
    
    # Create datasets
    print('Loading datasets...')
    loaders, poisoned_indices = create_unlearn_datasets(args)
    
    # Load poisoned model
    print('\nLoading poisoned model...')
    class ModelSettings:
        pass
    model_settings = ModelSettings()
    model_settings.model = args.model
    model_settings.dropout = args.dropout
    model_settings.task = 'test'
    
    poisoned_model = get_network(model_settings)
    
    checkpoint = torch.load(args.poisoned_model, map_location=device)
    poisoned_model.load_state_dict(checkpoint['model_state_dict'])
    poisoned_model.to(device)
    poisoned_model.eval()
    
    print('Poisoned model architecture:')
    print(poisoned_model)
    
    # Evaluate poisoned model
    print('\n' + '-'*80)
    print('POISONED MODEL PERFORMANCE')
    print('-'*80)
    evaluate_binary(poisoned_model, loaders['clean_train_noaug'], 
                   'Clean train (retain)', print)
    evaluate_binary(poisoned_model, loaders['poison_train_noaug'],
                   'Poison train (forget)', print)
    evaluate_binary(poisoned_model, loaders['test'], 'Test', print)
    
    # Load baseline model if provided
    if args.baseline_model is not None:
        print('\n' + '-'*80)
        print('BASELINE (CLEAN) MODEL PERFORMANCE')
        print('-'*80)
        baseline_model = get_network(model_settings)
        checkpoint = torch.load(args.baseline_model, map_location=device)
        baseline_model.load_state_dict(checkpoint['model_state_dict'])
        baseline_model.to(device)
        baseline_model.eval()
        
        evaluate_binary(baseline_model, loaders['clean_train_noaug'],
                       'Clean train', print)
        evaluate_binary(baseline_model, loaders['test'], 'Test', print)
    
    # Compute SVD on clean data
    print('\n' + '-'*80)
    print('COMPUTING SVD ON CLEAN (RETAIN) DATA')
    print('-'*80)
    
    svd_file = f'{exp_dir}/clean_svd.pt'
    if os.path.exists(svd_file):
        print(f'Loading existing SVD from {svd_file}')
        clean_svd = torch.load(svd_file)
    else:
        print('Computing SVD (this may take a while)...')
        
        # Use optimized approach for frozen models
        if args.freeze:
            print('Using optimized SVD for frozen model (feature-based)')
            from unlearn_utils import compute_svd_frozen_model
            clean_svd = compute_svd_frozen_model(
                poisoned_model,
                loaders['clean_train_aug'],
                printer=print
            )
        else:
            clean_svd = compute_svd_resnet(
                poisoned_model,
                loaders['clean_train_aug'],
                frozen=False,
                printer=print
            )
        
        torch.save(clean_svd, svd_file)
        print(f'SVD saved to {svd_file}')
    
    # Compute projection matrices
    print('\n' + '-'*80)
    print(f'COMPUTING PROJECTION MATRICES (retained variance: {args.retained_var*100:.1f}%)')
    print('-'*80)
    P = get_projection_matrices(clean_svd, retained_variance=args.retained_var, device=device)
    
    # Initialize unlearning model
    unlearn_model = get_network(model_settings)
    unlearn_model.load_state_dict(poisoned_model.state_dict())
    unlearn_model.to(device)
    
    # Optimizer (no momentum for projected gradient)
    optimizer = optim.SGD(unlearn_model.parameters(), lr=args.start_lr, momentum=0, weight_decay=0)
    
    # Learning rate schedule
    alpha = np.exp(np.log(args.end_lr / args.start_lr) / args.num_epochs)
    current_lr = args.start_lr
    
    # Tracking
    results = {
        'clean_train': [],
        'poison_train': [],
        'test': [],
        'val': []
    }
    
    losses_tracker = AverageMeter()
    loss1_tracker = AverageMeter()
    loss2_tracker = AverageMeter()
    
    # Name mapping for projection (cache for efficiency)
    name_mapping = {}
    
    print('\n' + '='*80)
    print('STARTING UNLEARNING')
    print('='*80)
    
    for epoch in range(args.num_epochs):
        # Unlearning phase
        unlearn_model.train()
        unlearn_model.apply(freeze_norm_stats)  # Keep BN stats frozen
        
        losses_tracker.reset()
        loss1_tracker.reset()
        loss2_tracker.reset()
        
        with tqdm(loaders['poison_train_aug'], desc=f'Epoch {epoch+1}/{args.num_epochs}') as pbar:
            for batch_idx, (images, labels) in enumerate(pbar):
                if epoch == 0:
                    # Skip first epoch (evaluate initial state)
                    continue  # ← FIX: salta solo questa iterazione
                
                images = images.to(device)
                labels = labels.to(device)
                
                # Ensure FC layer requires grad (important with frozen BN)
                for param in unlearn_model.fc.parameters():
                    param.requires_grad = True
                
                optimizer.zero_grad()
                outputs = unlearn_model(images)
                
                # Verify outputs have gradients
                if not outputs.requires_grad:
                    continue
                
                # Unlearning loss (from paper)
                probs = torch.sigmoid(outputs)
                
                # Loss 1: Minimize confidence on correct label
                # For binary: minimize prob if label=1, maximize if label=0
                loss1 = -torch.log(1 - torch.abs(probs - (1 - labels)) - args.offset + 1e-10).mean()
                
                # Loss 2: Maximize entropy
                entropy = -(probs * torch.log(probs + 1e-10) + 
                           (1 - probs) * torch.log(1 - probs + 1e-10))
                loss2 = -entropy.mean()  # Negative because we want to maximize
                
                loss = args.loss1_w * loss1 + args.loss2_w * loss2
                
                loss1_tracker.update(loss1.item(), images.size(0))
                loss2_tracker.update(loss2.item(), images.size(0))
                losses_tracker.update(loss.item(), images.size(0))
                
                loss.backward()
                
                # Projected gradient update
                with torch.no_grad():
                    for name, param in unlearn_model.named_parameters():
                        if param.grad is None:
                            continue

                        # Mappiamo il nome al formato usato in SVD/proiezioni
                        if name not in name_mapping:
                            P_name = name
                            for i in range(10):
                                P_name = P_name.replace(f'.{i}.', f'[{i}].')
                            P_name = P_name.replace('.weight', '').replace('.bias', '')
                            name_mapping[name] = P_name
                        else:
                            P_name = name_mapping[name]

                        # Se questo layer non ha matrice di proiezione, facciamo un update "normale" (SGD)
                        if P_name not in P:
                            if args.wd > 0:
                                param.grad.data += args.wd * param.data
                            param.data -= current_lr * param.grad.data
                            continue

                        # Flatten del gradiente: (out_dim, fan_in_flat)
                        sz = param.grad.data.shape[0]
                        grad_flat = param.grad.data.view(sz, -1)

                        P_mat = P[P_name]  # matrice di proiezione per questo layer

                        # Se le dimensioni non matchano (es. bias 1x1 vs P 2048x2048),
                        # NON applichiamo la proiezione e usiamo un update normale.
                        if grad_flat.shape[1] != P_mat.shape[0]:
                            if args.wd > 0:
                                param.grad.data += args.wd * param.data
                            param.data -= current_lr * param.grad.data
                            continue

                        # Aggiungiamo weight decay nello spazio flatten
                        if args.wd > 0:
                            grad_flat = grad_flat + args.wd * param.data.view(sz, -1)

                        # Proiezione del gradiente: grad_proj = grad - P @ grad
                        grad_proj = grad_flat - torch.mm(grad_flat, P_mat)

                        # Torniamo alla shape originale dei parametri
                        grad_proj = grad_proj.view_as(param.data)

                        # Aggiornamento parametro con gradiente proiettato
                        param.data -= current_lr * grad_proj
                
                pbar.set_postfix({
                    'loss': f'{losses_tracker.avg:.4f}',
                    'loss1': f'{loss1_tracker.avg:.4f}',
                    'loss2': f'{loss2_tracker.avg:.4f}',
                    'lr': f'{current_lr:.6f}'
                })
        
        # Evaluation
        print(f'\nEpoch {epoch+1} Evaluation:')
        print('-' * 60)
        
        loss, acc = evaluate_binary(unlearn_model, loaders['clean_train_noaug'],
                                   '[Unlearned] Clean train', print)
        results['clean_train'].append((loss, acc))
        
        loss, acc = evaluate_binary(unlearn_model, loaders['poison_train_noaug'],
                                   '[Unlearned] Poison train', print)
        results['poison_train'].append((loss, acc))
        
        loss, acc = evaluate_binary(unlearn_model, loaders['test'],
                                   '[Unlearned] Test', print)
        results['test'].append((loss, acc))
        
        loss, acc = evaluate_binary(unlearn_model, loaders['val'],
                                   '[Unlearned] Val', print)
        results['val'].append((loss, acc))
        
        print(f'Loss: {losses_tracker.avg:.4f} ({loss1_tracker.avg:.4f} + {loss2_tracker.avg:.4f})')
        print(f'Learning rate: {current_lr:.6f}')
        print('-' * 60)
        
        # Save checkpoint
        if args.save_model:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': unlearn_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr': current_lr,
                'results': results
            }
            torch.save(checkpoint, f'{exp_dir}/checkpoints/ckp_{epoch:03d}.pt')
        
        # Save best model (highest clean train accuracy)
        if len(results['clean_train']) == 1 or \
           results['clean_train'][-1][1] >= max([x[1] for x in results['clean_train'][:-1]]):
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': unlearn_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr': current_lr,
                'results': results
            }
            torch.save(checkpoint, f'{exp_dir}/checkpoints/best.pt')
            print(f'✓ Best model saved (Clean acc: {results["clean_train"][-1][1]:.2f}%)')
        
        # Plot results
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot([x[1] for x in results['clean_train']], label='Clean Train', marker='o')
        plt.plot([x[1] for x in results['poison_train']], label='Poison Train', marker='s')
        plt.plot([x[1] for x in results['test']], label='Test', marker='^')
        plt.plot([x[1] for x in results['val']], label='Val', marker='d')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy during Unlearning')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot([x[0] for x in results['clean_train']], label='Clean Train', marker='o')
        plt.plot([x[0] for x in results['poison_train']], label='Poison Train', marker='s')
        plt.plot([x[0] for x in results['test']], label='Test', marker='^')
        plt.plot([x[0] for x in results['val']], label='Val', marker='d')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss during Unlearning')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{exp_dir}/plots/progress.png', dpi=150)
        plt.close()
        
        # Save results
        np.save(f'{exp_dir}/results.npy', results)
        
        # Early stopping
        if len(results['clean_train']) > 1:
            clean_acc_drop = results['clean_train'][0][1] - results['clean_train'][-1][1]
            if clean_acc_drop > args.early_stop_thres:
                print(f'\n⚠ Early stopping: Clean accuracy dropped by {clean_acc_drop:.2f}%')
                break
        
        # Update learning rate
        current_lr *= alpha
    
    print('\n' + '='*80)
    print('UNLEARNING COMPLETED')
    print('='*80)
    print(f'Results saved to: {exp_dir}')
    print(f'Best model: {exp_dir}/checkpoints/best.pt')
    print('='*80)
    
    # Final summary
    print('\nFinal Performance Summary:')
    print('-' * 60)
    print(f'Initial Clean Train Acc: {results["clean_train"][0][1]:.2f}%')
    print(f'Final Clean Train Acc:   {results["clean_train"][-1][1]:.2f}%')
    print(f'Clean Acc Drop:          {results["clean_train"][0][1] - results["clean_train"][-1][1]:.2f}%')
    print()
    print(f'Initial Poison Train Acc: {results["poison_train"][0][1]:.2f}%')
    print(f'Final Poison Train Acc:   {results["poison_train"][-1][1]:.2f}%')
    print(f'Poison Acc Drop:          {results["poison_train"][0][1] - results["poison_train"][-1][1]:.2f}%')
    print()
    print(f'Initial Test Acc: {results["test"][0][1]:.2f}%')
    print(f'Final Test Acc:   {results["test"][-1][1]:.2f}%')
    print(f'Test Acc Change:  {results["test"][-1][1] - results["test"][0][1]:+.2f}%')
    print('-' * 60)


if __name__ == '__main__':
    main()