"""
Base dataset class for Image Difference Captioning.

This module provides a flexible dataset loader for various IDC datasets including:
- CLEVR-Change
- Spot-the-Diff  
- IEdit
- Custom datasets

Author: OmniDiff Reproduction Project
Date: 2026-11-11
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
import json
import pandas as pd


class ImageDifferenceDataset(Dataset):
    """
    Generic dataset for image difference captioning tasks.
    
    Expected directory structure:
        dataset_root/
            images/
                image_a_001.jpg
                image_b_001.jpg
                ...
            annotations.json  # or annotations.csv
    
    Annotation format (JSON):
        [
            {
                "image_a": "images/image_a_001.jpg",
                "image_b": "images/image_b_001.jpg", 
                "caption": "The red cube changed to a blue sphere.",
                "change_type": "object_replacement"  # optional
            },
            ...
        ]
    
    Annotation format (CSV):
        image_a,image_b,caption,change_type
        images/image_a_001.jpg,images/image_b_001.jpg,"The red cube...",object_replacement
        ...
    
    Args:
        root_dir: Root directory of the dataset
        split: Dataset split ("train", "val", "test")
        annotation_file: Name of annotation file (default: "annotations.json")
        image_processor: Optional image preprocessing function
        tokenizer: Optional tokenizer for text
        max_length: Maximum caption length (default: 128)
        return_change_type: Whether to return change type labels (default: False)
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        annotation_file: str = "annotations.json",
        image_processor: Optional[Callable] = None,
        tokenizer: Optional[Callable] = None,
        max_length: int = 128,
        return_change_type: bool = False,
    ):
        super().__init__()
        
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.return_change_type = return_change_type
        
        # Load annotations
        annotation_path = self.root_dir / split / annotation_file
        if not annotation_path.exists():
            # Try without split subdirectory
            annotation_path = self.root_dir / annotation_file
        
        if not annotation_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
        
        self.annotations = self._load_annotations(annotation_path)
        
        print(f"Loaded {len(self.annotations)} samples from {split} split")
    
    def _load_annotations(self, annotation_path: Path) -> List[Dict]:
        """Load annotations from JSON or CSV file."""
        if annotation_path.suffix == '.json':
            with open(annotation_path, 'r', encoding='utf-8') as f:
                annotations = json.load(f)
        elif annotation_path.suffix == '.csv':
            df = pd.read_csv(annotation_path)
            annotations = df.to_dict('records')
        else:
            raise ValueError(f"Unsupported annotation format: {annotation_path.suffix}")
        
        return annotations
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            sample: Dictionary containing:
                - image_a: Processed first image [3, H, W]
                - image_b: Processed second image [3, H, W]
                - caption: Caption text (str) or tokenized (dict)
                - change_type: Optional change type label
        """
        ann = self.annotations[idx]
        
        # Load images
        image_a_path = self.root_dir / self.split / ann['image_a']
        image_b_path = self.root_dir / self.split / ann['image_b']
        
        # Fallback: try without split subdirectory
        if not image_a_path.exists():
            image_a_path = self.root_dir / ann['image_a']
            image_b_path = self.root_dir / ann['image_b']
        
        image_a = Image.open(image_a_path).convert('RGB')
        image_b = Image.open(image_b_path).convert('RGB')
        
        # Process images
        if self.image_processor is not None:
            image_a = self.image_processor(image_a, return_tensors='pt')['pixel_values'][0]
            image_b = self.image_processor(image_b, return_tensors='pt')['pixel_values'][0]
        else:
            # Default: convert to tensor and normalize
            import torchvision.transforms as T
            transform = T.Compose([
                T.Resize((336, 336)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image_a = transform(image_a)
            image_b = transform(image_b)
        
        # Process caption
        caption = ann['caption']
        if self.tokenizer is not None:
            caption_tokens = self.tokenizer(
                caption,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            caption_data = {
                'input_ids': caption_tokens['input_ids'][0],
                'attention_mask': caption_tokens['attention_mask'][0],
                'text': caption
            }
        else:
            caption_data = {'text': caption}
        
        # Build sample
        sample = {
            'image_a': image_a,
            'image_b': image_b,
            'caption': caption_data,
            'idx': idx
        }
        
        # Optional: change type
        if self.return_change_type and 'change_type' in ann:
            sample['change_type'] = ann['change_type']
        
        return sample


class CLEVRChangeDataset(ImageDifferenceDataset):
    """
    Dataset loader for CLEVR-Change dataset.
    
    CLEVR-Change format:
        clevr_change/
            images/
                train/
                    CLEVR_train_000000_A.png
                    CLEVR_train_000000_B.png
                    ...
                val/
                    CLEVR_val_000000_A.png
                    CLEVR_val_000000_B.png
                    ...
            change_captions.json
    
    JSON format:
        {
            "train": [
                {
                    "image_id": "CLEVR_train_000000",
                    "caption": "A red cube was added to the scene.",
                    "change_type": "addition"
                },
                ...
            ],
            "val": [...]
        }
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        image_processor: Optional[Callable] = None,
        tokenizer: Optional[Callable] = None,
        max_length: int = 128,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load CLEVR-Change specific annotations
        annotation_path = self.root_dir / "change_captions.json"
        if not annotation_path.exists():
            raise FileNotFoundError(f"CLEVR-Change annotations not found: {annotation_path}")
        
        with open(annotation_path, 'r') as f:
            all_annotations = json.load(f)
        
        if split not in all_annotations:
            raise ValueError(f"Split '{split}' not found in annotations. Available: {list(all_annotations.keys())}")
        
        # Convert to standard format
        self.annotations = []
        for ann in all_annotations[split]:
            image_id = ann['image_id']
            self.annotations.append({
                'image_a': f"images/{split}/{image_id}_A.png",
                'image_b': f"images/{split}/{image_id}_B.png",
                'caption': ann['caption'],
                'change_type': ann.get('change_type', 'unknown')
            })
        
        print(f"Loaded {len(self.annotations)} CLEVR-Change samples from {split} split")


class SpotTheDiffDataset(ImageDifferenceDataset):
    """
    Dataset loader for Spot-the-Diff dataset.
    
    Spot-the-Diff format:
        spot_the_diff/
            images/
                img_000001_a.jpg
                img_000001_b.jpg
                ...
            annotations.json
    """
    pass  # Uses base ImageDifferenceDataset with default format


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching samples.
    
    Args:
        batch: List of samples from __getitem__
        
    Returns:
        batched: Dictionary with batched tensors
    """
    batched = {}
    
    # Stack images
    batched['image_a'] = torch.stack([sample['image_a'] for sample in batch])
    batched['image_b'] = torch.stack([sample['image_b'] for sample in batch])
    
    # Handle captions (check if tokenized or raw text)
    first_caption = batch[0]['caption']
    if isinstance(first_caption, dict) and 'input_ids' in first_caption:
        # Tokenized captions
        batched['caption'] = {
            'input_ids': torch.stack([sample['caption']['input_ids'] for sample in batch]),
            'attention_mask': torch.stack([sample['caption']['attention_mask'] for sample in batch]),
            'text': [sample['caption']['text'] for sample in batch]
        }
    elif isinstance(first_caption, dict) and 'text' in first_caption:
        # Dict with only text
        batched['caption'] = [sample['caption']['text'] for sample in batch]
    else:
        # Raw text
        batched['caption'] = [sample['caption'] for sample in batch]
    
    # Optional fields
    if 'change_type' in batch[0]:
        batched['change_type'] = [sample['change_type'] for sample in batch]
    
    batched['idx'] = torch.tensor([sample['idx'] for sample in batch])
    
    return batched


def test_dataset():
    """Test dataset loading with dummy data."""
    print("Testing ImageDifferenceDataset...")
    print("=" * 60)
    
    # Create dummy dataset structure
    import tempfile
    import os
    from PIL import Image
    import numpy as np
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create directory structure
        train_dir = tmpdir / "train" / "images"
        train_dir.mkdir(parents=True)
        
        # Create dummy images (100x100 RGB)
        for i in range(3):
            img_a = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
            img_b = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
            
            img_a.save(train_dir / f"image_a_{i:03d}.jpg")
            img_b.save(train_dir / f"image_b_{i:03d}.jpg")
        
        # Create dummy annotations
        annotations = [
            {
                "image_a": f"images/image_a_{i:03d}.jpg",
                "image_b": f"images/image_b_{i:03d}.jpg",
                "caption": f"This is difference description {i}.",
                "change_type": "test_change"
            }
            for i in range(3)
        ]
        
        with open(tmpdir / "train" / "annotations.json", 'w') as f:
            json.dump(annotations, f)
        
        # Test dataset loading
        print(f"Created test dataset at: {tmpdir}")
        dataset = ImageDifferenceDataset(
            root_dir=str(tmpdir),
            split="train",
            return_change_type=True
        )
        
        print(f"Dataset size: {len(dataset)}")
        
        # Test __getitem__
        sample = dataset[0]
        print(f"\nSample 0:")
        print(f"  image_a shape: {sample['image_a'].shape}")
        print(f"  image_b shape: {sample['image_b'].shape}")
        print(f"  caption: {sample['caption']['text']}")
        print(f"  change_type: {sample['change_type']}")
        
        # Test collate_fn
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
        batch = next(iter(dataloader))
        
        print(f"\nBatch:")
        print(f"  image_a shape: {batch['image_a'].shape}")
        print(f"  image_b shape: {batch['image_b'].shape}")
        print(f"  captions: {batch['caption']}")
        
    print("\n" + "=" * 60)
    print("[PASS] Dataset test completed!")


if __name__ == "__main__":
    # Run test
    test_dataset()
