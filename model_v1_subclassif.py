import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
import json
import shutil
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras import layers, models, utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from itertools import product
import sys

# Constants
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 64
CLASS_NAMES = ['class_1', 'class_2', 'class_3', 'class_4', 'class_5', 'class_6']
BASE_RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")
TRAIN_DIR = '/home/simul4/Documentos/Silvia_FP/original_1800_fotos/split_dataset/train/'
VAL_DIR = '/home/simul4/Documentos/Silvia_FP/original_1800_fotos/split_dataset/val/'
TEST_DIR = '/home/simul4/Documentos/Silvia_FP/original_1800_fotos/split_dataset/test/'
NEW_DATASET_DIR = '/home/simul4/Documentos/Silvia_FP/original_1800_fotos/dataset3/'
EVAL_RESULTS_DIR = BASE_RESULTS_DIR / "evaluation_results"

def create_hierarchical_dataset_structure():
    """Create dataset structure for hierarchical models"""
    HIERARCHICAL_DIR = Path("hierarchical_datasets")
    HIERARCHICAL_DIR.mkdir(exist_ok=True)
    
    for split in ['train', 'val', 'test']:
        # Create top-level dataset (4 classes)
        top_dir = HIERARCHICAL_DIR / f"top_{split}"
        (top_dir / "class_1").mkdir(parents=True, exist_ok=True)
        (top_dir / "class_4").mkdir(parents=True, exist_ok=True)
        (top_dir / "class_2_3").mkdir(parents=True, exist_ok=True)
        (top_dir / "class_5_6").mkdir(parents=True, exist_ok=True)
        
        # Create mid-level datasets (2 classes each)
        for group in ['2_3', '5_6']:
            mid_dir = HIERARCHICAL_DIR / f"mid{group}_{split}"
            (mid_dir / f"class_{group.split('_')[0]}").mkdir(parents=True, exist_ok=True)
            (mid_dir / f"class_{group.split('_')[1]}").mkdir(parents=True, exist_ok=True)
        
        # Populate with symbolic links
        split_dir = Path(TRAIN_DIR).parent / split
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
                
            class_name = class_dir.name
            if class_name in ['class_1', 'class_4']:
                shutil.copytree(class_dir, top_dir / class_name, dirs_exist_ok=True)
            elif class_name in ['class_2', 'class_3']:
                shutil.copytree(class_dir, top_dir / "class_2_3", dirs_exist_ok=True)
                shutil.copytree(class_dir, HIERARCHICAL_DIR / f"mid2_3_{split}" / class_name, dirs_exist_ok=True)
            elif class_name in ['class_5', 'class_6']:
                shutil.copytree(class_dir, top_dir / "class_5_6", dirs_exist_ok=True)
                shutil.copytree(class_dir, HIERARCHICAL_DIR / f"mid5_6_{split}" / class_name, dirs_exist_ok=True)

def load_datasets(model_type):
    """Load datasets for specific model type"""
    if model_type == 'top':
        base_dir = "hierarchical_datasets/top_"
        class_names = ['class_1', 'class_4', 'class_2_3', 'class_5_6']
    elif model_type == 'mid2_3':
        base_dir = "hierarchical_datasets/mid2_3_"
        class_names = ['class_2', 'class_3']
    elif model_type == 'mid5_6':
        base_dir = "hierarchical_datasets/mid5_6_"
        class_names = ['class_5', 'class_6']
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    train_ds = utils.image_dataset_from_directory(
        str(base_dir + "train"),
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        shuffle=True,
        seed=123,
        class_names=class_names
    ).map(lambda x, y: (x / 255.0, y))

    val_ds = utils.image_dataset_from_directory(
        str(base_dir + "val"),
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        shuffle=True,
        seed=123,
        class_names=class_names
    ).map(lambda x, y: (x / 255.0, y))

    test_ds = utils.image_dataset_from_directory(
        str(base_dir + "test"),
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        shuffle=False,
        class_names=class_names
    ).map(lambda x, y: (x / 255.0, y))

    return train_ds, val_ds, test_ds, class_names

def build_model(config, num_classes):
    """Build model with specified architecture"""
    model = models.Sequential()
    model.add(layers.Input(shape=(*IMAGE_SIZE, 1)))
    
    for layer in config['conv_layers']:
        filters, kernel_size = layer
        model.add(layers.Conv2D(filters, (kernel_size, kernel_size), activation='relu'))
        #dropout
        model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Flatten())
    
    for units in config['dense_units']:
        model.add(layers.Dense(units, activation='relu'))
    #dropout
    
    model.add(layers.Dropout(config['dropout_rate']))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_and_evaluate(config, model_type, train_ds, val_ds, test_ds, output_dir, model_save_path, class_names):
    """Train and evaluate a single model"""
    model = build_model(config, len(class_names))
    
    # Save model summary
    with open(output_dir / "model_summary.txt", "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))
    
    # Save configuration
    with open(output_dir / "config.json", "w") as f:
        json.dump({
            "model_type": model_type,
            "config": config,
            "class_names": class_names
        }, f)
    
    # Train model
    history = model.fit(
        train_ds,
        epochs=config['epochs'],
        validation_data=val_ds
    )
    
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    # Save training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.legend()
    
    plt.savefig(output_dir / "training_history.png")
    plt.close()
    
    # Save class names
    with open(model_save_path / "class_names.json", "w") as f:
        json.dump(class_names, f)

def evaluate_combination(combo_name, top_model_path, mid23_model_path, mid56_model_path, output_dir):
    """Evaluate a specific combination of models"""
    # Load models
    top_model = tf.keras.models.load_model(top_model_path)
    mid23_model = tf.keras.models.load_model(mid23_model_path)
    mid56_model = tf.keras.models.load_model(mid56_model_path)
    
    # Load original test dataset
    test_ds = utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        shuffle=False
    ).map(lambda x, y: (x / 255.0, y))
    
    y_true = []
    y_pred = []
    
    for images, labels in test_ds:
        # Top model prediction
        top_preds = top_model.predict(images, verbose=0)
        top_classes = np.argmax(top_preds, axis=1)
        
        # Process each image in batch
        batch_preds = []
        for i, top_class in enumerate(top_classes):
            if top_class == 0:  # class_1
                batch_preds.append(0)
            elif top_class == 1:  # class_4
                batch_preds.append(3)
            elif top_class == 2:  # class_2_3
                mid_pred = mid23_model.predict(images[i:i+1], verbose=0)
                batch_preds.append(1 + np.argmax(mid_pred))
            elif top_class == 3:  # class_5_6
                mid_pred = mid56_model.predict(images[i:i+1], verbose=0)
                batch_preds.append(4 + np.argmax(mid_pred))
        
        y_true.extend(labels.numpy())
        y_pred.extend(batch_preds)
    
    # Generate reports
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f'Confusion Matrix: {combo_name}')
    plt.savefig(output_dir / "confusion_matrix.png")
    plt.close()
    
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    with open(output_dir / "classification_report.txt", "w") as f:
        f.write(report)
    
    # Calculate accuracy
    accuracy = np.mean(np.array(y_true) == np.array(y_pred))
    print(f"Combination {combo_name} accuracy: {accuracy:.4f}")
    
    return accuracy

def main():
    """Main training pipeline"""
    # Create dataset structure
    # create_hierarchical_dataset_structure()
    
    # Load configurations
    with open('./configurations/models_v1.json', 'r') as f:
        configs = json.load(f)['configurations']
    
    # Generate all combinations
    all_combinations = list(product(configs, repeat=3))
    print(f"Total combinations to test: {len(all_combinations)}")
    
    print(f"All combinations: \n")
    for comb in all_combinations:
        print(f"Top: {comb[0]['name']}, Mid23: {comb[1]['name']}, Mid56: {comb[2]['name']}\n")
    
    all_combinations = all_combinations[-4:]
    
    # Results storage
    combo_results = {}
    
    for i, (top_cfg, mid23_cfg, mid56_cfg) in enumerate(all_combinations):
        combo_name = (f"top_{top_cfg['name']}_"
                      f"mid23_{mid23_cfg['name']}_"
                      f"mid56_{mid56_cfg['name']}")
        
        print(f"\n{'='*80}")
        print(f"Testing combination {i+1}/{len(all_combinations)}: {combo_name}")
        print(f"{'='*80}")
        
        # Create output directories
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        combo_dir = BASE_RESULTS_DIR / "combinations" / f"{combo_name}_{timestamp}"
        combo_dir.mkdir(parents=True)
        model_save_base = MODELS_DIR / "combinations" / combo_name
        model_save_base.mkdir(parents=True, exist_ok=True)
        
        # Train top model
        print(f"\nTraining top model: {top_cfg['name']}")
        top_type = 'top'
        train_ds, val_ds, test_ds, class_names = load_datasets(top_type)
        top_output_dir = combo_dir / "top_model"
        top_output_dir.mkdir()
        top_model_path = model_save_base / "top_model"
        train_and_evaluate(
            top_cfg, top_type, 
            train_ds, val_ds, test_ds, 
            top_output_dir, top_model_path, 
            class_names
        )
        
        # Train mid23 model
        print(f"\nTraining mid23 model: {mid23_cfg['name']}")
        mid23_type = 'mid2_3'
        train_ds, val_ds, test_ds, class_names = load_datasets(mid23_type)
        mid23_output_dir = combo_dir / "mid23_model"
        mid23_output_dir.mkdir()
        mid23_model_path = model_save_base / "mid23_model"
        train_and_evaluate(
            mid23_cfg, mid23_type, 
            train_ds, val_ds, test_ds, 
            mid23_output_dir, mid23_model_path, 
            class_names
        )
        
        # Train mid56 model
        print(f"\nTraining mid56 model: {mid56_cfg['name']}")
        mid56_type = 'mid5_6'
        train_ds, val_ds, test_ds, class_names = load_datasets(mid56_type)
        mid56_output_dir = combo_dir / "mid56_model"
        mid56_output_dir.mkdir()
        mid56_model_path = model_save_base / "mid56_model"
        train_and_evaluate(
            mid56_cfg, mid56_type, 
            train_ds, val_ds, test_ds, 
            mid56_output_dir, mid56_model_path, 
            class_names
        )
        
        # Evaluate combination
        print("\nEvaluating combination...")
        combo_eval_dir = combo_dir / "evaluation"
        combo_eval_dir.mkdir()
        accuracy = evaluate_combination(
            combo_name,
            top_model_path,
            mid23_model_path,
            mid56_model_path,
            combo_eval_dir
        )
        
        # Store results
        combo_results[combo_name] = {
            "top_model": top_cfg['name'],
            "mid23_model": mid23_cfg['name'],
            "mid56_model": mid56_cfg['name'],
            "accuracy": accuracy,
            "directory": str(combo_dir)
        }
        
        print(f"Completed combination: {combo_name}")
    
    # Save all results summary
    results_file = BASE_RESULTS_DIR / "combinations" / "all_results.json"
    with open(results_file, "w") as f:
        json.dump(combo_results, f, indent=2)
    
    print("\nAll combinations tested!")
    print(f"Results summary saved to {results_file}")
    
    # Find best combination
    best_combo = max(combo_results.items(), key=lambda x: x[1]['accuracy'])
    print("\nBest combination:")
    print(f"Name: {best_combo[0]}")
    print(f"Accuracy: {best_combo[1]['accuracy']:.4f}")
    print(f"Top model: {best_combo[1]['top_model']}")
    print(f"Mid23 model: {best_combo[1]['mid23_model']}")
    print(f"Mid56 model: {best_combo[1]['mid56_model']}")
    print(f"Directory: {best_combo[1]['directory']}")

if __name__ == "__main__":
    main()