"""
Test Baseline Setup - Verify Model Loading and Dataset

This script tests if we can:
1. Load the synthetic dataset
2. Load LLaVA-1.5-7B with 4-bit quantization (will download ~14GB on first run)
3. Create MDP module
4. Integrate MDP with LLaVA
5. Run a single forward pass

This is a pre-training verification step.
"""

import os
import sys
import torch
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from datasets.base_dataset import ImageDifferenceDataset
from models.mdp_module import MDPModule
from models.llava_wrapper import LLaVAWithMDP


def test_dataset():
    """Test dataset loading"""
    print("=" * 60)
    print("TEST 1: Dataset Loading")
    print("=" * 60)
    
    data_dir = "./data/synthetic_idc"
    
    if not os.path.exists(data_dir):
        print(f"✗ ERROR: Dataset directory not found: {data_dir}")
        return False
    
    try:
        dataset = ImageDifferenceDataset(
            root_dir=data_dir,
            split='train',
            annotation_file='train_annotations.json'
        )
        
        print(f"✓ Dataset loaded: {len(dataset)} samples")
        
        # Test loading a sample
        sample = dataset[0]
        print(f"✓ Sample 0 loaded:")
        print(f"    image_a shape: {sample['image_a'].shape}")
        print(f"    image_b shape: {sample['image_b'].shape}")
        print(f"    caption: {sample['caption'][:50]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ ERROR loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mdp_module():
    """Test MDP module creation"""
    print("\n" + "=" * 60)
    print("TEST 2: MDP Module")
    print("=" * 60)
    
    try:
        mdp = MDPModule(
            hidden_dim=1024,
            num_scales=3,
            num_heads=8
        )
        
        print(f"✓ MDP Module created")
        
        # Count parameters
        total_params = sum(p.numel() for p in mdp.parameters())
        print(f"  Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
        
        # Test forward pass
        batch_size = 2
        feat_a = torch.randn(batch_size, 1024, 24, 24)
        feat_b = torch.randn(batch_size, 1024, 24, 24)
        
        with torch.no_grad():
            output = mdp(feat_a, feat_b)
        
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {feat_a.shape}")
        print(f"  Output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ ERROR with MDP module: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_loading():
    """Test LLaVA + MDP model loading (requires ~14GB download on first run)"""
    print("\n" + "=" * 60)
    print("TEST 3: LLaVA + MDP Model Loading")
    print("=" * 60)
    
    print("\n⚠️  WARNING: This will download LLaVA-1.5-7B weights (~14GB) on first run!")
    print("This may take 10-30 minutes depending on your internet speed.")
    
    response = input("\nContinue? (y/n): ")
    if response.lower() != 'y':
        print("Skipping model loading test.")
        return None
    
    try:
        print("\nLoading LLaVA-1.5-7B with 4-bit quantization...")
        print("(This may take several minutes...)")
        
        model = LLaVAWithMDP(
            llm_model_name="liuhaotian/llava-v1.5-7b",
            load_in_4bit=True,
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            use_mdp=True,
            mdp_hidden_dim=1024,
            mdp_num_heads=8,
            mdp_num_scales=3,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        print(f"✓ Model loaded successfully!")
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_ratio = trainable_params / total_params * 100
        
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,} ({trainable_ratio:.2f}%)")
        
        # Test GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            mem_allocated = torch.cuda.memory_allocated() / 1024**3
            mem_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"  GPU memory allocated: {mem_allocated:.2f} GB")
            print(f"  GPU memory reserved: {mem_reserved:.2f} GB")
            
            if mem_allocated > 3.5:
                print(f"\n⚠️  WARNING: Model using {mem_allocated:.2f}GB > 3.5GB")
                print("    May not fit with 4GB VRAM during training!")
        
        return True
        
    except Exception as e:
        print(f"✗ ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("M³Diff Baseline Setup Test")
    print("Phase 5 - Experimental Validation")
    print("=" * 60)
    print()
    
    # Check CUDA
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        device_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✓ CUDA available")
        print(f"  Device: {device_name}")
        print(f"  Memory: {device_memory:.2f} GB")
    else:
        print("✗ CUDA not available - will use CPU (very slow)")
    
    print()
    
    # Run tests
    results = {}
    
    results['dataset'] = test_dataset()
    results['mdp'] = test_mdp_module()
    results['model'] = test_model_loading()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, result in results.items():
        if result is True:
            status = "✓ PASSED"
        elif result is False:
            status = "✗ FAILED"
        else:
            status = "⊘ SKIPPED"
        print(f"  {test_name.upper():15s}: {status}")
    
    # Check if ready
    if results['dataset'] and results['mdp']:
        if results['model'] is True:
            print("\n" + "=" * 60)
            print("✓✓✓ ALL TESTS PASSED ✓✓✓")
            print("=" * 60)
            print("\n✓ Ready to train M³Diff baseline!")
            print("\nNext steps:")
            print("  1. Run: python train_m3diff.py --config configs/base_config.yaml")
            print("  2. Monitor training progress")
            print("  3. Evaluate on test set")
        elif results['model'] is None:
            print("\n" + "=" * 60)
            print("⊘ Model test skipped")
            print("=" * 60)
            print("\nDataset and MDP module are working.")
            print("Model loading will happen during training.")
        else:
            print("\n" + "=" * 60)
            print("✗ Model loading failed")
            print("=" * 60)
            print("\nPlease fix model loading issues before training.")
    else:
        print("\n" + "=" * 60)
        print("✗ SOME TESTS FAILED")
        print("=" * 60)
        print("\nPlease fix the errors above before proceeding.")


if __name__ == '__main__':
    main()
