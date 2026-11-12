"""
Minimal Baseline Training Script for Phase 5A Experiment 5A.1

This is a simplified training script to verify the baseline works.
Uses synthetic dataset (40 train, 5 val samples).

Goals:
1. Verify LLaVA-1.5-7B loads with 4-bit quantization
2. Verify MDP integrates correctly
3. Verify training runs without OOM (GPU ≤ 4GB)
4. Generate sample captions after training
"""

import os
import sys
import yaml
import torch
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from datasets.base_dataset import ImageDifferenceDataset
from models.llava_wrapper import LLaVAWithMDP


def collate_fn(batch):
    """Collate function for dataloader"""
    images_a = torch.stack([item['image_a'] for item in batch])
    images_b = torch.stack([item['image_b'] for item in batch])
    
    # Handle caption format (dict or string)
    captions = []
    for item in batch:
        if isinstance(item['caption'], dict):
            captions.append(item['caption']['text'])
        else:
            captions.append(item['caption'])
    
    return {
        'image_a': images_a,
        'image_b': images_b,
        'caption': captions,
        'image_id': [item.get('image_id', i) for i, item in enumerate(batch)]
    }


def create_dataloaders(config):
    """Create train and val dataloaders"""
    print("\n" + "=" * 60)
    print("Creating Dataloaders")
    print("=" * 60)
    
    # Train dataset
    train_dataset = ImageDifferenceDataset(
        root_dir=config['dataset']['data_dir'],
        split='train',
        annotation_file=config['dataset']['train_file']
    )
    
    # Val dataset
    val_dataset = ImageDifferenceDataset(
        root_dir=config['dataset']['data_dir'],
        split='val',
        annotation_file=config['dataset']['val_file']
    )
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory'],
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory'],
        collate_fn=collate_fn
    )
    
    print(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    
    return train_loader, val_loader


def create_model(config, device):
    """Create M3Diff model"""
    print("\n" + "=" * 60)
    print("Creating Model")
    print("=" * 60)
    print("\nThis will download LLaVA-1.5-7B (~14GB) on first run...")
    print("Please wait, this may take 10-30 minutes...\n")
    
    model_config = config['model']
    
    model = LLaVAWithMDP(
        llm_model_name=model_config['base_model'],
        load_in_4bit=model_config['use_quantization'],
        lora_r=model_config['lora']['r'],
        lora_alpha=model_config['lora']['alpha'],
        lora_dropout=model_config['lora']['dropout'],
        use_mdp=True,
        mdp_dim=model_config['mdp']['hidden_dim'],
        mdp_num_heads=model_config['mdp']['num_heads'],
        mdp_num_scales=model_config['mdp']['num_scales'],
        device=device
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel created successfully!")
    print(f"  Total parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
    print(f"  Trainable ratio: {trainable_params/total_params*100:.2f}%")
    
    # Check GPU memory
    if device == 'cuda':
        torch.cuda.empty_cache()
        mem_allocated = torch.cuda.memory_allocated() / 1024**3
        mem_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"\nGPU Memory:")
        print(f"  Allocated: {mem_allocated:.2f} GB")
        print(f"  Reserved: {mem_reserved:.2f} GB")
        
        if mem_allocated > 3.8:
            print(f"\n  WARNING: Memory usage high ({mem_allocated:.2f} GB > 3.8 GB)")
            print(f"           May OOM during training on 4GB GPU!")
    
    return model


def train_epoch(model, train_loader, optimizer, device, epoch, config):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    grad_accum_steps = config['training']['gradient_accumulation_steps']
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    optimizer.zero_grad()
    
    for step, batch in enumerate(pbar):
        # Move to device
        images_a = batch['image_a'].to(device)
        images_b = batch['image_b'].to(device)
        captions = batch['caption']
        
        # Forward pass
        try:
            outputs = model(
                images_a=images_a,
                images_b=images_b,
                captions=captions
            )
            
            loss = outputs['loss'] / grad_accum_steps
            
            # Backward
            loss.backward()
            
            # Update weights every grad_accum_steps
            if (step + 1) % grad_accum_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config['training']['max_grad_norm']
                )
                
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * grad_accum_steps
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item() * grad_accum_steps:.4f}",
                'avg_loss': f"{total_loss / (step + 1):.4f}"
            })
            
            # Check GPU memory periodically
            if step % 10 == 0 and device == 'cuda':
                mem_allocated = torch.cuda.memory_allocated() / 1024**3
                if mem_allocated > 3.9:
                    print(f"\n  WARNING: GPU memory {mem_allocated:.2f} GB > 3.9 GB!")
        
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n\nOUT OF MEMORY ERROR at step {step}!")
                print(f"Last GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                raise
            else:
                raise
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def validate(model, val_loader, device):
    """Validation"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            images_a = batch['image_a'].to(device)
            images_b = batch['image_b'].to(device)
            captions = batch['caption']
            
            outputs = model(
                images_a=images_a,
                images_b=images_b,
                captions=captions
            )
            
            loss = outputs['loss']
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    return avg_loss


def generate_samples(model, val_loader, device, num_samples=5):
    """Generate sample captions"""
    print("\n" + "=" * 60)
    print("Generating Sample Captions")
    print("=" * 60)
    
    model.eval()
    samples = []
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= num_samples:
                break
            
            images_a = batch['image_a'].to(device)
            images_b = batch['image_b'].to(device)
            ground_truth = batch['caption'][0]  # First in batch
            
            # Generate
            generated = model.generate(
                images_a=images_a,
                images_b=images_b,
                max_length=100,
                num_beams=3
            )
            
            samples.append({
                'ground_truth': ground_truth,
                'generated': generated[0]
            })
            
            print(f"\nSample {i+1}:")
            print(f"  GT:  {ground_truth}")
            print(f"  Gen: {generated[0]}")
    
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/base_config.yaml')
    parser.add_argument('--epochs', type=int, default=None, help='Override epochs from config')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = Path(config['training']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("M3Diff Baseline Training - Phase 5A Experiment 5A.1")
    print("=" * 60)
    print(f"\nConfig: {args.config}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print(f"Epochs: {config['training']['num_epochs']}")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config)
    
    # Create model
    model = create_model(config, device)
    
    # Optimizer
    print("\n" + "=" * 60)
    print("Setting up Optimizer")
    print("=" * 60)
    
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    print(f"Optimizer: AdamW")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"Weight decay: {config['training']['weight_decay']}")
    
    # Training loop
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'epochs': []
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(1, config['training']['num_epochs'] + 1):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch}/{config['training']['num_epochs']}")
        print(f"{'=' * 60}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch, config)
        
        # Validate
        val_loss = validate(model, val_loader, device)
        
        # Log
        print(f"\nEpoch {epoch} Results:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['epochs'].append(epoch)
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = output_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"  [SAVED] Best model: {checkpoint_path}")
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nBest Val Loss: {best_val_loss:.4f}")
    
    # Generate samples
    samples = generate_samples(model, val_loader, device, num_samples=5)
    
    # Save history
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nHistory saved: {history_path}")
    
    # Save samples
    samples_path = output_dir / 'generated_samples.json'
    with open(samples_path, 'w') as f:
        json.dump(samples, f, indent=2)
    print(f"Samples saved: {samples_path}")
    
    # Final memory check
    if device == 'cuda':
        print(f"\nFinal GPU Memory:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    print("\n" + "=" * 60)
    print("Experiment 5A.1 Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Review training history and generated samples")
    print("  2. Proceed to Experiment 5A.2 (Implement Improvement 2)")


if __name__ == '__main__':
    main()
