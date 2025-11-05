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
from joblib import Parallel, delayed
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

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

# Create directories if they don't exist
BASE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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
        model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Flatten())
    
    for units in config['dense_units']:
        model.add(layers.Dense(units, activation='relu'))
    
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
    logger.info(f"Training {model_type} model with config: {config['name']}")
    
    
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
    logger.info(f"Model saved to {model_save_path}")

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
    
    return model

def evaluate_combination(combo_name, top_model_path, mid23_model_path, mid56_model_path, output_dir):
    """Evaluate a specific combination of models"""
    logger.info(f"Evaluating combination: {combo_name}")
    
   
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
    logger.info(f"Combination {combo_name} accuracy: {accuracy:.4f}")
    
    return accuracy

def process_combination(i, total, top_cfg, mid23_cfg, mid56_cfg):
    """Process a single combination in parallel"""
    combo_name = (f"top_{top_cfg['name']}_"
                  f"mid23_{mid23_cfg['name']}_"
                  f"mid56_{mid56_cfg['name']}")
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing combination {i+1}/{total}: {combo_name}")
    logger.info(f"{'='*80}")
    
    try:
        # Create output directories
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        combo_dir = BASE_RESULTS_DIR / "combinations" / f"{combo_name}_{timestamp}"
        combo_dir.mkdir(parents=True, exist_ok=True)
        model_save_base = MODELS_DIR / "combinations" / combo_name
        model_save_base.mkdir(parents=True, exist_ok=True)
        
        # Train top model
        logger.info(f"Training top model: {top_cfg['name']}")
        top_type = 'top'
        train_ds, val_ds, test_ds, class_names = load_datasets(top_type)
        top_output_dir = combo_dir / "top_model"
        top_output_dir.mkdir(exist_ok=True)
        top_model_path = model_save_base / "top_model"
        train_and_evaluate(
            top_cfg, top_type, 
            train_ds, val_ds, test_ds, 
            top_output_dir, top_model_path, 
            class_names
        )
        
        # Train mid23 model
        logger.info(f"Training mid23 model: {mid23_cfg['name']}")
        mid23_type = 'mid2_3'
        train_ds, val_ds, test_ds, class_names = load_datasets(mid23_type)
        mid23_output_dir = combo_dir / "mid23_model"
        mid23_output_dir.mkdir(exist_ok=True)
        mid23_model_path = model_save_base / "mid23_model"
        train_and_evaluate(
            mid23_cfg, mid23_type, 
            train_ds, val_ds, test_ds, 
            mid23_output_dir, mid23_model_path, 
            class_names
        )
        
        # Train mid56 model
        logger.info(f"Training mid56 model: {mid56_cfg['name']}")
        mid56_type = 'mid5_6'
        train_ds, val_ds, test_ds, class_names = load_datasets(mid56_type)
        mid56_output_dir = combo_dir / "mid56_model"
        mid56_output_dir.mkdir(exist_ok=True)
        mid56_model_path = model_save_base / "mid56_model"
        train_and_evaluate(
            mid56_cfg, mid56_type, 
            train_ds, val_ds, test_ds, 
            mid56_output_dir, mid56_model_path, 
            class_names
        )
        
        # Evaluate combination
        logger.info("Evaluating combination...")
        combo_eval_dir = combo_dir / "evaluation"
        combo_eval_dir.mkdir(exist_ok=True)
        accuracy = evaluate_combination(
            combo_name,
            top_model_path,
            mid23_model_path,
            mid56_model_path,
            combo_eval_dir
        )
        
        logger.info(f"Completed combination: {combo_name}")
        
        return {
            "combo_name": combo_name,
            "top_model": top_cfg['name'],
            "mid23_model": mid23_cfg['name'],
            "mid56_model": mid56_cfg['name'],
            "accuracy": accuracy,
            "directory": str(combo_dir)
        }
        
    except Exception as e:
        logger.error(f"Error processing combination {combo_name}: {str(e)}")
        return {
            "combo_name": combo_name,
            "error": str(e)
        }

def main():
    """Main training pipeline with parallel execution"""
    logger.info("Starting hierarchical model training pipeline")
    
    # Create dataset structure once
    if not Path("hierarchical_datasets").exists():
        logger.info("Creating hierarchical dataset structure...")
        create_hierarchical_dataset_structure()
    else:
        logger.info("Hierarchical datasets already exist")
    
    # Load configurations
    with open('./configurations/models_v1.json', 'r') as f:
        configs = json.load(f)['configurations']
    
    # Generate all combinations
    all_combinations = list(product(configs, repeat=3))
    all_combinations = all_combinations[-3:-2]
    # Print / Log all combinations
    # for i in range(len(all_combinations)):
    #     top_cfg, mid23_cfg, mid56_cfg = all_combinations[i]
    #     print(f"Combination {i}: {top_cfg['name']}, {mid23_cfg['name']}, {mid56_cfg['name']}")
    #     logger.info(f"Combination {i}: {top_cfg['name']}, {mid23_cfg['name']}, {mid56_cfg['name']}")
    # sys.exit()
    total_combinations = len(all_combinations)
    logger.info(f"Total combinations to test: {total_combinations}")
    
    # Run combinations in parallel
    results = Parallel(n_jobs=2, verbose=10)(
        delayed(process_combination)(i, total_combinations, top_cfg, mid23_cfg, mid56_cfg)
        for i, (top_cfg, mid23_cfg, mid56_cfg) in enumerate(all_combinations)
    )
    
    # Filter out failed runs
    combo_results = {}
    for result in results:
        if result and "accuracy" in result:
            combo_results[result["combo_name"]] = {
                "top_model": result["top_model"],
                "mid23_model": result["mid23_model"],
                "mid56_model": result["mid56_model"],
                "accuracy": result["accuracy"],
                "directory": result["directory"]
            }
        else:
            logger.warning(f"Failed combination: {result.get('combo_name', 'unknown')}")
    
    # Save all results summary
    results_file = BASE_RESULTS_DIR / "combinations" / "all_results.json"
    with open(results_file, "w") as f:
        json.dump(combo_results, f, indent=2)
    
    logger.info(f"\nAll combinations tested! Results saved to {results_file}")
    
    # Find best combination
    if combo_results:
        best_combo = max(combo_results.items(), key=lambda x: x[1]['accuracy'])
        logger.info("\nBest combination:")
        logger.info(f"Name: {best_combo[0]}")
        logger.info(f"Accuracy: {best_combo[1]['accuracy']:.4f}")
        logger.info(f"Top model: {best_combo[1]['top_model']}")
        logger.info(f"Mid23 model: {best_combo[1]['mid23_model']}")
        logger.info(f"Mid56 model: {best_combo[1]['mid56_model']}")
        logger.info(f"Directory: {best_combo[1]['directory']}")
    else:
        logger.error("No successful combinations completed!")

if __name__ == "__main__":
    main()