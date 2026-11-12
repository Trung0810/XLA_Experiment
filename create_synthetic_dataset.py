"""
Create a small synthetic dataset for testing M3Diff training pipeline
Generates simple image pairs with known changes and captions
"""

import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import random

def create_simple_scene(size=(336, 336), num_objects=3, seed=None):
    """
    Create a simple scene with geometric objects
    
    Args:
        size: Image size (width, height)
        num_objects: Number of objects in scene
        seed: Random seed for reproducibility
        
    Returns:
        PIL Image
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Create white background
    img = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(img)
    
    # Object types
    shapes = ['circle', 'rectangle', 'triangle']
    colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple']
    
    objects = []
    
    for i in range(num_objects):
        # Random shape and color
        shape = random.choice(shapes)
        color = random.choice(colors)
        
        # Random position (avoid edges)
        x = random.randint(50, size[0] - 100)
        y = random.randint(50, size[1] - 100)
        obj_size = random.randint(30, 60)
        
        # Draw shape
        if shape == 'circle':
            bbox = [x - obj_size//2, y - obj_size//2, x + obj_size//2, y + obj_size//2]
            draw.ellipse(bbox, fill=color, outline='black', width=2)
        
        elif shape == 'rectangle':
            bbox = [x - obj_size//2, y - obj_size//2, x + obj_size//2, y + obj_size//2]
            draw.rectangle(bbox, fill=color, outline='black', width=2)
        
        elif shape == 'triangle':
            points = [
                (x, y - obj_size//2),
                (x - obj_size//2, y + obj_size//2),
                (x + obj_size//2, y + obj_size//2)
            ]
            draw.polygon(points, fill=color, outline='black')
        
        objects.append({
            'shape': shape,
            'color': color,
            'position': (x, y),
            'size': obj_size
        })
    
    return img, objects


def create_changed_scene(original_img, objects, change_type='add', size=(336, 336), seed=None):
    """
    Create a changed version of the scene
    
    Args:
        original_img: Original PIL Image
        objects: List of objects in original scene
        change_type: Type of change ('add', 'remove', 'color', 'position')
        size: Image size
        seed: Random seed
        
    Returns:
        Changed PIL Image and change description
    """
    if seed is not None:
        random.seed(seed + 1000)
    
    # Copy original image
    img = original_img.copy()
    draw = ImageDraw.Draw(img)
    
    shapes = ['circle', 'rectangle', 'triangle']
    colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple']
    
    if change_type == 'add':
        # Add new object
        shape = random.choice(shapes)
        color = random.choice(colors)
        x = random.randint(50, size[0] - 100)
        y = random.randint(50, size[1] - 100)
        obj_size = random.randint(30, 60)
        
        if shape == 'circle':
            bbox = [x - obj_size//2, y - obj_size//2, x + obj_size//2, y + obj_size//2]
            draw.ellipse(bbox, fill=color, outline='black', width=2)
        elif shape == 'rectangle':
            bbox = [x - obj_size//2, y - obj_size//2, x + obj_size//2, y + obj_size//2]
            draw.rectangle(bbox, fill=color, outline='black', width=2)
        elif shape == 'triangle':
            points = [
                (x, y - obj_size//2),
                (x - obj_size//2, y + obj_size//2),
                (x + obj_size//2, y + obj_size//2)
            ]
            draw.polygon(points, fill=color, outline='black')
        
        position = "left" if x < size[0]//2 else "right"
        description = f"A {color} {shape} was added on the {position} side of the scene"
    
    elif change_type == 'remove':
        # Describe removal
        if objects:
            obj = random.choice(objects)
            position = "left" if obj['position'][0] < size[0]//2 else "right"
            description = f"A {obj['color']} {obj['shape']} on the {position} side was removed from the scene"
        else:
            description = "An object was removed from the scene"
    
    elif change_type == 'color':
        # Describe color change
        if objects:
            obj = random.choice(objects)
            new_color = random.choice([c for c in colors if c != obj['color']])
            position = "left" if obj['position'][0] < size[0]//2 else "right"
            description = f"The {obj['color']} {obj['shape']} on the {position} side changed color to {new_color}"
        else:
            description = "An object changed color"
    
    elif change_type == 'position':
        # Describe position change
        if objects:
            obj = random.choice(objects)
            old_pos = "left" if obj['position'][0] < size[0]//2 else "right"
            new_pos = "right" if old_pos == "left" else "left"
            description = f"The {obj['color']} {obj['shape']} moved from the {old_pos} to the {new_pos} side"
        else:
            description = "An object moved position"
    
    else:
        description = "The scene has changed"
    
    return img, description


def create_synthetic_dataset(
    num_samples=50,
    output_dir='./data/synthetic_idc',
    train_ratio=0.8,
    val_ratio=0.1
):
    """
    Create a small synthetic dataset for IDC task
    
    Args:
        num_samples: Number of image pairs to generate
        output_dir: Output directory
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (output_path / 'images_a').mkdir(exist_ok=True)
    (output_path / 'images_b').mkdir(exist_ok=True)
    
    change_types = ['add', 'remove', 'color', 'position']
    
    annotations = []
    
    print(f"[INFO] Generating {num_samples} synthetic image pairs...")
    
    for i in range(num_samples):
        # Create original scene
        img_a, objects = create_simple_scene(seed=i)
        
        # Create changed scene
        change_type = random.choice(change_types)
        img_b, description = create_changed_scene(img_a, objects, change_type, seed=i)
        
        # Save images
        img_a_path = f"images_a/{i:04d}.png"
        img_b_path = f"images_b/{i:04d}.png"
        
        img_a.save(output_path / img_a_path)
        img_b.save(output_path / img_b_path)
        
        # Add annotation
        annotations.append({
            'image_id': i,
            'image_a': img_a_path,
            'image_b': img_b_path,
            'caption': description,
            'change_type': change_type
        })
        
        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{num_samples} pairs")
    
    # Split dataset
    num_train = int(num_samples * train_ratio)
    num_val = int(num_samples * val_ratio)
    num_test = num_samples - num_train - num_val
    
    train_anns = annotations[:num_train]
    val_anns = annotations[num_train:num_train+num_val]
    test_anns = annotations[num_train+num_val:]
    
    # Save annotations
    splits = {
        'train': train_anns,
        'val': val_anns,
        'test': test_anns
    }
    
    for split_name, split_data in splits.items():
        ann_path = output_path / f'{split_name}_annotations.json'
        with open(ann_path, 'w') as f:
            json.dump(split_data, f, indent=2)
        print(f"[SAVED] {split_name}: {len(split_data)} samples -> {ann_path}")
    
    # Save dataset info
    info = {
        'dataset': 'Synthetic IDC',
        'num_samples': num_samples,
        'splits': {
            'train': len(train_anns),
            'val': len(val_anns),
            'test': len(test_anns)
        },
        'change_types': change_types,
        'image_size': [336, 336]
    }
    
    info_path = output_path / 'dataset_info.json'
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"[SAVED] Dataset info -> {info_path}")
    
    print(f"\n[SUCCESS] Synthetic dataset created at {output_path}")
    print(f"  Total samples: {num_samples}")
    print(f"  Train: {len(train_anns)}, Val: {len(val_anns)}, Test: {len(test_anns)}")
    
    return output_path


if __name__ == "__main__":
    # Create synthetic dataset
    create_synthetic_dataset(
        num_samples=50,
        output_dir='./data/synthetic_idc',
        train_ratio=0.8,
        val_ratio=0.1
    )
