import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras import layers, models, utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json

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

def load_datasets():
    train_ds = utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        shuffle=True,
        seed=123
    ).map(lambda x, y: (x / 255.0, y))  # Normalization
    
    val_ds = utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        shuffle=True,
        seed=123
    ).map(lambda x, y: (x / 255.0, y))  # Normalization
    
    test_ds = utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        shuffle=True,
        seed=123
    ).map(lambda x, y: (x / 255.0, y))  # Normalization
    
    return train_ds, val_ds, test_ds

def build_model(config):
    model = models.Sequential()
    model.add(layers.Input(shape=(*IMAGE_SIZE, 1)))
    
    # Add convolutional layers from config
    for layer in config['conv_layers']:
        filters, kernel_size = layer
        model.add(layers.Conv2D(filters, (kernel_size, kernel_size), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Flatten())
    
    # Add dense layers from config
    for units in config['dense_units']:
        model.add(layers.Dense(units, activation='relu'))
    
    model.add(layers.Dropout(config['dropout_rate']))
    model.add(layers.Dense(len(CLASS_NAMES), activation='softmax'))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_and_evaluate(config, train_ds, val_ds, test_ds, output_dir, model_save_path):
    # Create model
    model = build_model(config)
    
    # Save model summary
    with open(output_dir / "model_summary.txt", "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))
    
    # Train model
    history = model.fit(
        train_ds,
        epochs=config['epochs'],
        validation_data=val_ds
    )
    
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    # Evaluate
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test accuracy: {test_acc}")
    
    # Save training history plots
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
    
    # Generate and save confusion matrices
    def generate_and_save_cm(dataset, name):
        y_true = []
        y_pred = []
        for images, labels in dataset:
            y_true.extend(labels.numpy())
            y_pred.extend(np.argmax(model.predict(images), axis=1))
        
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        plt.title(f'Confusion Matrix ({name})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(output_dir / f"confusion_matrix_{name}.png")
        plt.close()
        
        # Save classification report
        report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
        with open(output_dir / f"classification_report_{name}.txt", "w") as f:
            f.write(report)
    
    generate_and_save_cm(train_ds, "train")
    generate_and_save_cm(test_ds, "test")

def main():
    # Load configurations from JSON file
    list_configurations = ["models_v1.json", "models_v0.json", "baseline.json"]
    for config in list_configurations:
        try:
            #with open('./configurations/models_v1.json', 'r') as f:
            with open(f'./configurations/{config}', 'r') as f:
                config_data = json.load(f)
                configurations = config_data['configurations']
        except FileNotFoundError:
            print("Error: configs.json file not found!")
            return
        except json.JSONDecodeError:
            print("Error: Invalid JSON format in configs.json!")
            return

        # Load datasets once
        train_ds, val_ds, test_ds = load_datasets()
        
        # Create base results directory
        BASE_RESULTS_DIR.mkdir(exist_ok=True)
        MODELS_DIR.mkdir(exist_ok=True)
        train_dir = BASE_RESULTS_DIR / "train_3datasets"
        train_dir.mkdir(exist_ok=True)
        
        # Run experiments
        for config in configurations:
            # Create unique output directory
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            run_name = f"{config['name']}_{timestamp}"
            output_dir = train_dir / run_name
            output_dir.mkdir()
            model_save_path = MODELS_DIR / "train_3datasets" / run_name
            
            print(f"Running configuration: {config['name']}")
            train_and_evaluate(config, train_ds, val_ds, test_ds, output_dir, model_save_path)
            print(f"Completed! Results saved to {output_dir}\n")

def generate_reports(model, dataset, dataset_name, output_dir):
    """Helper function to generate and save confusion matrix and classification report."""
    y_true = []
    y_pred = []
    for images, labels in dataset:
        y_true.extend(labels.numpy())
        y_pred_batch = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(y_pred_batch, axis=1))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f'Confusion Matrix ({dataset_name})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(output_dir / f"confusion_matrix_{dataset_name}.png")
    plt.close()
    
    # Classification Report
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    with open(output_dir / f"classification_report_{dataset_name}.txt", "w") as f:
        f.write(report)

def evaluate_saved_models():
    """Main function to evaluate all saved models on new dataset."""
    # Load new dataset
    new_ds = utils.image_dataset_from_directory(
        NEW_DATASET_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        shuffle=False,  # Important for correct label order
        seed=123
    ).map(lambda x, y: (x / 255.0, y))  # Normalization
    
    # Create evaluation results directory
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process each model
    for model_dir in MODELS_DIR.iterdir():
        if model_dir.is_dir():
            try:
                print(f"\nEvaluating model: {model_dir.name}")
                
                # Load model
                model = tf.keras.models.load_model(model_dir)
                
                # Create output directory
                eval_output_dir = EVAL_RESULTS_DIR / model_dir.name
                eval_output_dir.mkdir(exist_ok=True)
                
                # Evaluate metrics
                test_loss, test_acc = model.evaluate(new_ds)
                print(f"Test accuracy: {test_acc}")
                
                # Save metrics
                with open(eval_output_dir / "evaluation_metrics.txt", "w") as f:
                    f.write(f"Test Loss: {test_loss}\nTest Accuracy: {test_acc}\n")
                
                # Generate reports
                generate_reports(model, new_ds, "new_dataset", eval_output_dir)
                
            except Exception as e:
                print(f"Error processing {model_dir.name}: {str(e)}")


def rotate_image(image, label):
    """Rotate image 90 degrees clockwise"""
    image = tf.image.rot90(image, k=1)  # k=1 means 90 degrees clockwise
    return image, label

def evaluate_saved_models_rotated():
    """Evaluate all saved models on rotated (90° right) dataset."""
    # Create evaluation results directory
    rotated_results_dir = BASE_RESULTS_DIR / "evaluation_results_rotated90"
    rotated_results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load new dataset with rotation
    new_ds = utils.image_dataset_from_directory(
        NEW_DATASET_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        shuffle=False,
        seed=123
    ).map(lambda x, y: (x / 255.0, y))  # Normalization
    new_ds = new_ds.map(rotate_image)  # Apply rotation
        
        
    # Process each model
    for model_dir in MODELS_DIR.iterdir():
        if model_dir.is_dir():
            try:
                print(f"\nEvaluating model: {model_dir.name}")
                
                # Load model
                model = tf.keras.models.load_model(model_dir)
                
                # Create output directory
                eval_output_dir = rotated_results_dir / model_dir.name
                eval_output_dir.mkdir(exist_ok=True)
                
                # Evaluate metrics
                test_loss, test_acc = model.evaluate(new_ds)
                print(f"Test accuracy: {test_acc}")
                
                # Save metrics
                with open(eval_output_dir / "evaluation_metrics.txt", "w") as f:
                    f.write(f"Test Loss: {test_loss}\nTest Accuracy: {test_acc}\n")
                
                # Generate reports
                generate_reports(model, new_ds, "new_dataset", eval_output_dir)
                
            except Exception as e:
                print(f"Error processing {model_dir.name}: {str(e)}")

if __name__ == "__main__":
    main()
    # evaluate_saved_models()
    # evaluate_saved_models_rotated()