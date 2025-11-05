import os
from collections import defaultdict

def find_duplicates(root_dir):
    # Dictionary to store file names per class
    class_files = defaultdict(list)
    
    # Walk through the directory structure
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(root_dir, split)
        if not os.path.exists(split_path):
            continue
            
        for class_name in os.listdir(split_path):
            class_path = os.path.join(split_path, class_name)
            if not os.path.isdir(class_path):
                continue
                
            for filename in os.listdir(class_path):
                # Store the filename and which split it's in
                class_files[(class_name, filename)].append(split)
    
    # Find duplicates (files that appear in more than one split)
    duplicates = {k: v for k, v in class_files.items() if len(v) > 1}
    
    return duplicates

# Example usage
root_directory = 'original_1800_fotos/split_dataset'
duplicates = find_duplicates(root_directory)

if duplicates:
    print("Found duplicates across splits:")
    for (class_name, filename), splits in duplicates.items():
        print(f"Class: {class_name}, File: {filename} appears in splits: {', '.join(splits)}")
else:
    print("No duplicates found across train, val, and test splits!")