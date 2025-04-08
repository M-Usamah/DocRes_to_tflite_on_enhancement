import os
import json
import shutil
import random
from pathlib import Path
import numpy as np

def setup_appearance_dataset():
    # Source directories
    realdae_dir = Path("MannualDataset/RealDAE")
    docunet_dir = Path("MannualDataset/DocUNet_from_DocAligner")
    
    # Target directories
    train_dir = Path("data/train/appearance/realdae")
    val_dir = Path("data/val/appearance/realdae")
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all image pairs
    image_pairs = []
    
    # Process RealDAE images
    if realdae_dir.exists():
        print(f"Processing images from {realdae_dir}")
        for img in realdae_dir.glob("*"):
            if img.name.endswith("_docaligner.png"):
                input_img = img
                target_img = img.parent / img.name.replace("_docaligner.png", "_target.png")
                if target_img.exists():
                    image_pairs.append((input_img, target_img))
                    print(f"Found pair: {input_img.name} -> {target_img.name}")
    
    # Process DocUNET images
    if docunet_dir.exists():
        print(f"Processing images from {docunet_dir}")
        for img in docunet_dir.glob("*"):
            if img.name.endswith("_docaligner.png"):
                input_img = img
                target_img = img.parent / img.name.replace("_docaligner.png", "_target.png")
                if target_img.exists():
                    image_pairs.append((input_img, target_img))
                    print(f"Found pair: {input_img.name} -> {target_img.name}")
    
    if not image_pairs:
        print("No image pairs found! Please check the directory structure and file names.")
        return
        
    # Shuffle and split dataset
    random.shuffle(image_pairs)
    split_idx = int(len(image_pairs) * 0.7)  # 70% for training
    train_pairs = image_pairs[:split_idx]
    val_pairs = image_pairs[split_idx:]
    
    # Create JSON files for training and validation
    train_json = []
    val_json = []
    
    # Process training pairs
    for idx, (in_img, gt_img) in enumerate(train_pairs):
        new_in_name = f"{idx+1}_in.png"
        new_gt_name = f"{idx+1}_gt.png"
        
        # Copy files
        shutil.copy2(in_img, train_dir / new_in_name)
        shutil.copy2(gt_img, train_dir / new_gt_name)
        
        # Add to JSON
        train_json.append({
            "in_path": f"realdae/{new_in_name}",
            "gt_path": f"realdae/{new_gt_name}"
        })
    
    # Process validation pairs
    for idx, (in_img, gt_img) in enumerate(val_pairs):
        new_in_name = f"{idx+1}_in.png"
        new_gt_name = f"{idx+1}_gt.png"
        
        # Copy files
        shutil.copy2(in_img, val_dir / new_in_name)
        shutil.copy2(gt_img, val_dir / new_gt_name)
        
        # Add to JSON
        val_json.append({
            "in_path": f"realdae/{new_in_name}",
            "gt_path": f"realdae/{new_gt_name}"
        })
    
    # Save JSON files
    os.makedirs("data", exist_ok=True)
    with open("data/train_appearance.json", "w") as f:
        json.dump(train_json, f, indent=4)
    
    with open("data/val_appearance.json", "w") as f:
        json.dump(val_json, f, indent=4)
    
    print(f"\nDataset preparation completed:")
    print(f"Training samples: {len(train_pairs)}")
    print(f"Validation samples: {len(val_pairs)}")
    print(f"\nData saved to:")
    print(f"- Training directory: {train_dir}")
    print(f"- Validation directory: {val_dir}")
    print(f"- Training JSON: data/train_appearance.json")
    print(f"- Validation JSON: data/val_appearance.json")

if __name__ == "__main__":
    setup_appearance_dataset() 