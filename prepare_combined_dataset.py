"""
Prepare COMBINED dataset from all downloaded sources
Creates unified training/val/test splits for MobileNetV3-Small
"""
import os
import shutil
import random
from pathlib import Path
from collections import defaultdict
import csv

# Paths
IMP_DIR = Path('/home/ml/CAS/imp')
OUTPUT_DIR = Path('/home/ml/CAS/combined_dataset')

# Define unified class mapping
# We'll create a multi-domain defect classifier
UNIFIED_CLASSES = [
    'clean',           # No defect
    'bridge',          # Carinthia: bridge defects
    'line_collapse',   # Carinthia: line collapse
    'line_break',      # Carinthia: line break / opens
    'scratch',         # WM-811K, MixedWM38
    'center',          # WM-811K, MixedWM38
    'edge',            # WM-811K edge defects
    'donut',           # WM-811K, MixedWM38
    'random',          # WM-811K random patterns
    'surface_defect',  # SD-Saliency, general surface
    'pcb_defect',      # DeepPCB
    'other'            # Catch-all
]

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
MAX_PER_CLASS = 10000  # Cap each class to balance

def load_carinthia():
    """Load Carinthia SEM dataset with 6 defect classes"""
    images = []
    csv_path = IMP_DIR / 'carinthia_sem/data/carinthia.csv'
    img_dir = IMP_DIR / 'carinthia_sem/data/images'
    
    # Class mapping: 1-6 to our classes
    carinthia_map = {
        '1': 'bridge',        # Single Bridge
        '2': 'bridge',        # Thin Bridge
        '3': 'line_collapse', # Line Collapse (most common)
        '4': 'line_break',    # Line Break
        '5': 'bridge',        # Multi-Bridge Horizontal
        '6': 'bridge',        # Multi-Bridge Non-Horizontal
    }
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            label = row['label']
            if label in carinthia_map:
                img_path = img_dir / row['file_name']
                if img_path.exists():
                    images.append((img_path, carinthia_map[label]))
    
    print(f"Carinthia: {len(images)} images")
    return images

def load_wm811k():
    """Load WM-811K wafer map dataset"""
    images = []
    base_dir = IMP_DIR / 'wm811k/images'
    
    wm_map = {
        'none': 'clean',
        'Scratch': 'scratch',
        'Edge-Ring': 'edge',
        'Edge-Loc': 'edge',
        'Center': 'center',
        'Loc': 'center',  # Similar to center
        'Donut': 'donut',
        'Random': 'random',
        'Near-full': 'other',
    }
    
    for class_dir in base_dir.iterdir():
        if class_dir.is_dir():
            target = wm_map.get(class_dir.name, 'other')
            for img in class_dir.glob('*.png'):
                images.append((img, target))
    
    print(f"WM-811K: {len(images)} images")
    return images

def load_mixedwm38():
    """Load MixedWM38 dataset"""
    images = []
    base_dir = IMP_DIR / 'mixedwm38/images'
    
    mixed_map = {
        'None': 'clean',
        'Scratch': 'scratch',
        'Edge-Ring': 'edge',
        'Edge-Loc': 'edge',
        'Center': 'center',
        'Loc': 'center',
        'Donut': 'donut',
        'Random': 'random',
        'Mixed': 'other',
        'Near-full': 'other',
    }
    
    for class_dir in base_dir.iterdir():
        if class_dir.is_dir():
            target = mixed_map.get(class_dir.name, 'other')
            for img in class_dir.glob('*.png'):
                images.append((img, target))
    
    print(f"MixedWM38: {len(images)} images")
    return images

def load_sd_saliency():
    """Load SD-Saliency-900 surface defects"""
    images = []
    base_dir = IMP_DIR / 'sd_saliency/SD-saliency-900/Source Images'
    
    if base_dir.exists():
        for img in base_dir.glob('*.*'):
            if img.suffix.lower() in ['.jpg', '.png', '.bmp']:
                images.append((img, 'surface_defect'))
    
    print(f"SD-Saliency: {len(images)} images")
    return images

def load_deeppcb():
    """Load DeepPCB dataset"""
    images = []
    base_dir = IMP_DIR / 'deeppcb/PCBData'
    
    if base_dir.exists():
        for img in base_dir.rglob('*_test.jpg'):
            images.append((img, 'pcb_defect'))
    
    print(f"DeepPCB: {len(images)} images")
    return images

def balance_and_split(all_images):
    """Balance classes and create train/val/test splits"""
    
    # Group by class
    by_class = defaultdict(list)
    for img_path, label in all_images:
        by_class[label].append(img_path)
    
    print("\nClass distribution:")
    for cls, imgs in sorted(by_class.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  {cls}: {len(imgs)}")
    
    # Balance and split
    splits = {'train': [], 'val': [], 'test': []}
    
    for cls, imgs in by_class.items():
        random.shuffle(imgs)
        sampled = imgs[:MAX_PER_CLASS]
        
        n = len(sampled)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        
        for img in sampled[:n_train]:
            splits['train'].append((img, cls))
        for img in sampled[n_train:n_train+n_val]:
            splits['val'].append((img, cls))
        for img in sampled[n_train+n_val:]:
            splits['test'].append((img, cls))
    
    return splits

def copy_images(splits):
    """Copy images to output directory structure"""
    
    # Get all unique classes
    all_classes = set()
    for split_data in splits.values():
        for _, cls in split_data:
            all_classes.add(cls)
    
    # Create directories
    for split in ['train', 'val', 'test']:
        for cls in all_classes:
            (OUTPUT_DIR / split / cls).mkdir(parents=True, exist_ok=True)
    
    # Copy images
    for split, data in splits.items():
        print(f"\n{split.upper()}: {len(data)} images")
        class_counts = defaultdict(int)
        
        for i, (src, cls) in enumerate(data):
            ext = src.suffix
            dst = OUTPUT_DIR / split / cls / f"{cls}_{i:06d}{ext}"
            shutil.copy2(src, dst)
            class_counts[cls] += 1
        
        for cls, count in sorted(class_counts.items()):
            print(f"  {cls}: {count}")

def main():
    print("=" * 60)
    print("Creating COMBINED Multi-Domain Defect Dataset")
    print("=" * 60)
    
    random.seed(42)
    
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)
    
    # Load all datasets
    all_images = []
    all_images.extend(load_carinthia())
    all_images.extend(load_wm811k())
    all_images.extend(load_mixedwm38())
    all_images.extend(load_sd_saliency())
    all_images.extend(load_deeppcb())
    
    print(f"\nTotal: {len(all_images)} images")
    
    # Balance and split
    splits = balance_and_split(all_images)
    
    # Copy
    copy_images(splits)
    
    print(f"\n{'='*60}")
    print(f"DONE! Dataset saved to: {OUTPUT_DIR}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
