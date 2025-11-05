import os
import shutil
from sklearn.model_selection import train_test_split

def split_and_merge_dataset(new_dataset_path, existing_split_path, test_size=0.15, val_size=0.05, random_state=42):
    """
    Split new dataset into train/val/test and merge with existing split dataset
    
    Args:
        new_dataset_path: Path to the new dataset (dataset3)
        existing_split_path: Path to the existing split dataset
        test_size: Proportion for test set (default 60%)
        val_size: Proportion for validation set (default 5%)
        random_state: Random seed for reproducibility
    """
    # Calculate actual validation ratio (since we split train_val -> train + val)
    val_ratio = val_size / (1 - test_size)
    
    for class_name in os.listdir(new_dataset_path):
        class_path = os.path.join(new_dataset_path, class_name)
        if not os.path.isdir(class_path):
            continue
            
        # Get all image files in the class
        images = [f for f in os.listdir(class_path) if f.lower().endswith('.jpg')]
        print(f"Processing {class_name} with {len(images)} images")
        
        # Split into train_val and test first
        train_val, test = train_test_split(
            images, 
            test_size=test_size, 
            random_state=random_state
        )
        
        # Split train_val into train and val
        train, val = train_test_split(
            train_val, 
            test_size=val_ratio, 
            random_state=random_state
        )
        
        # Copy files to existing split directories
        for split, files in [('train', train), ('val', val), ('test', test)]:
            dest_dir = os.path.join(existing_split_path, split, class_name)
            os.makedirs(dest_dir, exist_ok=True)
            
            for file in files:
                src = os.path.join(class_path, file)
                dst = os.path.join(dest_dir, file)
                
                # Check if file already exists in destination
                if os.path.exists(dst):
                    base, ext = os.path.splitext(file)
                    new_name = f"{base}_from_dataset3{ext}"
                    dst = os.path.join(dest_dir, new_name)
                    print(f"Renaming duplicate {file} to {new_name} in {split}/{class_name}")
                
                shutil.copy2(src, dst)
                
            print(f"Copied {len(files)} images to {split}/{class_name}")

# Example usage
new_dataset_path = 'original_1800_fotos/dataset3'
existing_split_path = 'original_1800_fotos/split_dataset'
split_and_merge_dataset(new_dataset_path, existing_split_path)