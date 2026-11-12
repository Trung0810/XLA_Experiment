"""
Quick test before full training - verify everything works
"""
import os
import sys
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("=" * 60)
print("Quick Pre-Training Test")
print("=" * 60)
print()

# Test 1: CUDA
print("Test 1: CUDA")
if torch.cuda.is_available():
    print(f"  [PASS] Device: {torch.cuda.get_device_name(0)}")
    print(f"  [PASS] Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("  [FAIL] CUDA not available")
    sys.exit(1)

# Test 2: Dataset
print("\nTest 2: Dataset")
try:
    from datasets.base_dataset import ImageDifferenceDataset
    
    dataset = ImageDifferenceDataset(
        root_dir="./data/synthetic_idc",
        split='train',
        annotation_file='train_annotations.json'
    )
    
    sample = dataset[0]
    print(f"  [PASS] Dataset loaded: {len(dataset)} samples")
    print(f"  [PASS] Sample shapes: {sample['image_a'].shape}, {sample['image_b'].shape}")
    print(f"  [PASS] Caption type: {type(sample['caption'])}")
    
except Exception as e:
    print(f"  [FAIL] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: MDP Module
print("\nTest 3: MDP Module")
try:
    from models.mdp_module import MDPModule
    
    mdp = MDPModule(dim=1024, num_scales=3, num_heads=8)
    
    feat_a = torch.randn(1, 1024, 24, 24)
    feat_b = torch.randn(1, 1024, 24, 24)
    
    with torch.no_grad():
        output_a, output_b = mdp(feat_a, feat_b)
    
    print(f"  [PASS] MDP created")
    print(f"  [PASS] Forward pass: {feat_a.shape} -> ({output_a.shape}, {output_b.shape})")
    
    total_params = sum(p.numel() for p in mdp.parameters())
    print(f"  [PASS] Parameters: {total_params/1e6:.1f}M")
    
except Exception as e:
    print(f"  [FAIL] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Check config
print("\nTest 4: Config File")
try:
    import yaml
    
    with open("configs/base_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"  [PASS] Config loaded")
    print(f"  [INFO] Base model: {config['model']['base_model']}")
    print(f"  [INFO] Batch size: {config['training']['batch_size']}")
    print(f"  [INFO] Epochs: {config['training']['num_epochs']}")
    print(f"  [INFO] LoRA r: {config['model']['lora']['r']}")
    
except Exception as e:
    print(f"  [FAIL] {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("All pre-checks PASSED!")
print("=" * 60)
print("\nReady to train. This will:")
print("  1. Download LLaVA-1.5-7B (~14GB) on first run")
print("  2. Train for 1-3 epochs (~1-3 hours)")
print("  3. Save checkpoints to ./experiments/")
print()
print("To start training, run:")
print("  python train_m3diff.py --config configs/base_config.yaml")
