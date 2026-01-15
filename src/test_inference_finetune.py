import os
import torch
import pandas as pd
import numpy as np
import json
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,  # Regression metrics
    roc_auc_score, f1_score, accuracy_score, precision_score, recall_score,  # Binary classification metrics
    balanced_accuracy_score, classification_report  # Multiclass metrics
)
from datetime import datetime

# Import model and dataset classes
from model import ViTBackboneNet, Classifier, SingleScanModel, SingleScanModelBP, SingleScanModelQuad
from dataset import (BrainAgeDataset, MCIStrokeDataset, SequenceDataset, DualImageDataset, QuadImageDataset, 
                    get_validation_transform, get_validation_transform_dual, get_validation_transform_quad,
                    dual_image_collate_fn, quad_image_collate_fn)

# =============================================================================
# CONFIGURATION - MODIFY THESE PATHS AS NEEDED
# =============================================================================

# Model paths
SIMCLR_CKPT_PATH = "/cluster/work/bmdslab/brain-mri/foundation_models/BrainIAC/src/checkpoints/BrainIAC.ckpt"

# Dataset configurations - Add your datasets here
DATASETS = {
    "survival_task": {
        "checkpoint_path": "/media/data/divyanshu/BrainIAC_V2/checkpoints/BrainIAC_OS.ckpt",
        "test_csv_path": "/media/data/divyanshu/foundation_model/brainiac_data/os/csv/brats_clinical.csv",
        "root_dir": "/media/data/divyanshu/foundation_model/brainiac_data/os/data",
        "output_csv_path": "/media/data/divyanshu/BrainIAC_V2/inference/os_100_test_predictions_brats_clinical.csv",
        "task_type": "classification",
        "image_type": "quad",
        "num_classes": 1
    },
    "idh_task": {
        "checkpoint_path": "/media/data/divyanshu/foundation_model/Brainiac_revision/checkpoints/simclrvitfinetune_checkpoints/100percent_idh_simclrvitcls_0001_best-model-epoch=99-val_auroc=0.87.ckpt",
        "test_csv_path": "/media/data/divyanshu/foundation_model/Brainiac_revision/simclrvit_finetuning/idh_csvs/test.csv",
        "root_dir": "/media/data/divyanshu/foundation_model/brainiac_data",
        "output_csv_path": "/media/data/divyanshu/foundation_model/Brainiac_revision/simclrvit_finetuning/perturbation_results/idh_100_test_predictions.csv",
        "task_type": "classification",
        "image_type": "dual",
        "num_classes": 1
    },
    "mci_task": {
        "checkpoint_path": "/media/data/divyanshu/foundation_model/Brainiac_revision/checkpoints/simclrvitfinetune_checkpoints/100percent_mci_tejas_cls_simclrvitcls_0005_best-model-epoch=09-val_auc=0.78.ckpt",
        "test_csv_path": "/media/data/divyanshu/foundation_model/Brainiac_revision/mae_finetuning/mci_oasis1_csvs/test.csv",
        "root_dir": "/media/data/divyanshu/foundation_model/brainiac_data",
        "output_csv_path": "/media/data/divyanshu/foundation_model/Brainiac_revision/simclrvit_finetuning/perturbation_mci/dummy_100_mcitestverificationresults.csv",
        "task_type": "classification",
        "image_type": "single",
        "num_classes": 1
    },
    "sequence_task": {
        "checkpoint_path": "/media/data/divyanshu/foundation_model/Brainiac_revision/simclrvit_finetuning/frozen_downstream_checkpoints/100percent_seq_tejas_cls_swinvit_0008_best-model-epoch=76-val_auc=0.95.ckpt",
        "test_csv_path": "/media/data/divyanshu/foundation_model/brainiac_data/BRATS23_dataset/csv/testing_dataset_info.csv",
        "root_dir": "/media/data/divyanshu/foundation_model/brainiac_data",
        "output_csv_path": "/media/data/divyanshu/foundation_model/Brainiac_revision/simclrvit_finetuning/frozen_finetune_medicalnet/multiclass_100_test_predictions.csv",
        "task_type": "multiclass",
        "image_type": "single",
        "num_classes": 4
    },
    "brainage_task": {
        "checkpoint_path": "/media/data/divyanshu/foundation_model/Brainiac_revision/checkpoints/simclrvitfinetune_checkpoints/100percent_brainage_tejas_cls_simclrvitcls_001_best-model-epoch=49-val_mae=65.54.ckpt",
        "test_csv_path": "/media/data/divyanshu/foundation_model/Brainiac_revision/mae_finetuning/brainage_csvs/ALL_TEST.csv",
        "root_dir": "/media/data/divyanshu/foundation_model/brainiac_data",
        "output_csv_path": "/media/data/divyanshu/foundation_model/Brainiac_revision/simclrvit_finetuning/brainage_inference_output/brain_age_10_test_predictions_ext.csv",
        "task_type": "regression",
        "image_type": "single",
        "num_classes": 1
    },
    "timetostroke_task": {
        "checkpoint_path": "/media/data/divyanshu/foundation_model/Brainiac_revision/simclrvit_finetuning/frozen_downstream_checkpoints/100percent_stroke_tejas_cls_swinvit_0008_best-model-epoch=19-val_mae=55.83.ckpt",
        "test_csv_path": "/media/data/divyanshu/foundation_model/Brainiac_revision/few_shot_benchmarking/few_shot_csvs/stroke/V2_cropped_test.csv",
        "root_dir": "/media/data/divyanshu/foundation_model/brainiac_data",
        "output_csv_path": "/media/data/divyanshu/foundation_model/Brainiac_revision/simclrvit_finetuning/perturbation_results/stroke_100_test_predictions.csv",
        "task_type": "regression",
        "image_type": "single",
        "num_classes": 1
    },
    "radiotherapy_response": {
        "checkpoint_path": "/cluster/work/bmdslab/brain-mri/foundation_models/BrainIAC/results/rt_config_15_train_best-model-epoch=57-val_auc=0.00.ckpt",
        "test_csv_path": "/cluster/work/bmdslab/brain-mri/foundation_models/BrainIAC/src/data/csvs/sample_test_rt_pred_binary.csv",
        "root_dir": "/cluster/work/bmdslab/borgw/GLIMMRT",
        "output_csv_path": "/cluster/work/bmdslab/brain-mri/foundation_models/BrainIAC/results/rt_test_predictions.csv",
        "task_type": "classification",
        "image_type": "quad",
        "num_classes": 1
    }
}

# Select which datasets to run inference on (use dataset keys from above)
DATASETS_TO_RUN = [
   "radiotherapy_response"
]

# Data configuration
IMAGE_SIZE = (96, 96, 96)
BATCH_SIZE = 1  # Keep small for inference
NUM_WORKERS = 1

# GPU configuration
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# =============================================================================

def load_model(checkpoint_path, simclr_ckpt_path, task_type="classification", image_type="quad", num_classes=1):
    """
    Load the trained model from checkpoint
    """
    print(f"Loading model from {checkpoint_path}")
    
    # Create backbone
    backbone = ViTBackboneNet(simclr_ckpt_path=simclr_ckpt_path)
    
    # Create classifier
    classifier = Classifier(d_model=768, num_classes=num_classes)
    
    # Create full model based on image type
    if image_type == "single":
        model = SingleScanModel(backbone, classifier)
    elif image_type == "dual":
        model = SingleScanModelBP(backbone, classifier)
    elif image_type == "quad":
        model = SingleScanModelQuad(backbone, classifier)
    else:
        raise ValueError(f"Unknown image_type: {image_type}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Extract state dict - handle Lightning module format
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        # Remove Lightning module prefixes
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("model."):
                new_key = key[6:]  # Remove "model." prefix
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict
    else:
        state_dict = checkpoint
    
    # Load weights
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(DEVICE)
    
    print(f"Model loaded successfully on {DEVICE}")
    return model

def create_test_dataset(csv_path, root_dir, image_type="quad", image_size=(96, 96, 96), dataset_name=""):
    """
    Create test dataset based on image type and task
    """
    print(f"Creating {image_type} image test dataset from {csv_path}")
    
    if image_type == "single":
        transform = get_validation_transform(image_size=image_size)
        
        # Select appropriate dataset class based on task
        if "brain_age" in dataset_name:
            dataset = BrainAgeDataset(csv_path=csv_path, root_dir=root_dir, transform=transform)
        elif "mci" in dataset_name or "stroke" in dataset_name:
            dataset = MCIStrokeDataset(csv_path=csv_path, root_dir=root_dir, transform=transform)
        elif "sequence" in dataset_name or "multiclass" in dataset_name:
            dataset = SequenceDataset(csv_path=csv_path, root_dir=root_dir, transform=transform)
        else:
            # Default to BrainAgeDataset for backwards compatibility
            dataset = BrainAgeDataset(csv_path=csv_path, root_dir=root_dir, transform=transform)
            print(f"Warning: Unknown single image task '{dataset_name}', using BrainAgeDataset")
        
        collate_fn = None
        
    elif image_type == "dual":
        transform = get_validation_transform_dual(image_size=image_size)
        dataset = DualImageDataset(csv_path=csv_path, root_dir=root_dir, transform=transform)
        collate_fn = dual_image_collate_fn
        
    elif image_type == "quad":
        transform = get_validation_transform_quad(image_size=image_size)
        dataset = QuadImageDataset(csv_path=csv_path, root_dir=root_dir, transform=transform)
        collate_fn = quad_image_collate_fn
        
    else:
        raise ValueError(f"Unknown image_type: {image_type}")
    
    print(f"Test dataset created with {len(dataset)} samples using {dataset.__class__.__name__}")
    return dataset, collate_fn

def run_inference(model, dataset, collate_fn, batch_size=1, task_type="classification"):
    """
    Run inference on the test dataset
    """
    print("Running inference...")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )
    
    raw_outputs = []  # Store raw model outputs
    predictions = []  # Store processed predictions (probabilities for classification)
    class_predictions = []  # Store class predictions for multiclass
    labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            if collate_fn is not None:
                # For dual/quad datasets with custom collate_fn
                images, batch_labels = batch
                images = images.to(DEVICE)
                batch_labels = batch_labels.to(DEVICE)
            else:
                # For single image dataset
                images = batch['image'].to(DEVICE)
                batch_labels = batch['label'].to(DEVICE)
            
            # Forward pass
            outputs = model(images)
            
            # Store raw outputs
            raw_outputs.extend(outputs.cpu().numpy())
            
            # Process outputs based on task type
            if task_type == "classification":
                # Apply sigmoid for binary classification
                probs = torch.sigmoid(outputs)
                predictions.extend(probs.cpu().numpy().flatten())
                # Binary class predictions (0 or 1)
                class_preds = (probs > 0.5).int()
                class_predictions.extend(class_preds.cpu().numpy().flatten())
            elif task_type == "multiclass":
                # Apply softmax for multiclass classification
                probs = torch.softmax(outputs, dim=1)
                predictions.extend(probs.cpu().numpy())  # Keep all class probabilities
                # Class predictions using argmax
                class_preds = torch.argmax(outputs, dim=1)
                class_predictions.extend(class_preds.cpu().numpy().flatten())
            else:
                # For regression, predictions are same as raw outputs
                predictions.extend(outputs.cpu().numpy().flatten())
            
            # Store labels
            labels.extend(batch_labels.cpu().numpy().flatten())
    
    print(f"Inference completed. Generated {len(predictions)} predictions.")
    
    if task_type == "multiclass":
        return np.array(raw_outputs), np.array(predictions), np.array(class_predictions), np.array(labels)
    else:
        return np.array(raw_outputs), np.array(predictions), np.array(class_predictions), np.array(labels)

def save_predictions(csv_path, raw_outputs, predictions, class_predictions, output_path, task_type="classification"):
    """
    Save predictions to CSV file
    """
    print(f"Saving predictions to {output_path}")
    
    # Load original CSV
    df = pd.read_csv(csv_path)
    
    # Add predictions column based on task type
    if task_type == "classification":
        df['model_output'] = raw_outputs.flatten()  # Raw logits
        df['predicted_probability'] = predictions  # Sigmoid probabilities
        df['predicted_class'] = class_predictions  # Binary predictions
    elif task_type == "multiclass":
        # Raw logits for each class
        for i in range(raw_outputs.shape[1]):
            df[f'model_output_class_{i}'] = raw_outputs[:, i]
        
        # Probabilities for each class
        for i in range(predictions.shape[1]):
            df[f'predicted_probability_class_{i}'] = predictions[:, i]
        
        # Predicted class (argmax)
        df['predicted_class'] = class_predictions
    else:
        # For regression, only save model output
        df['predicted_value'] = raw_outputs.flatten()  # Raw model output
    
    # Save to output path
    df.to_csv(output_path, index=False)
    print(f"Results saved successfully!")
    
    # Print some statistics
    print("\n=== PREDICTION STATISTICS ===")
    if task_type == "classification":
        print(f"Raw model outputs (logits):")
        print(f"  Mean: {raw_outputs.flatten().mean():.4f}")
        print(f"  Std: {raw_outputs.flatten().std():.4f}")
        print(f"  Min: {raw_outputs.flatten().min():.4f}")
        print(f"  Max: {raw_outputs.flatten().max():.4f}")
        print(f"Predicted probabilities:")
        print(f"  Mean: {predictions.mean():.4f}")
        print(f"  Std: {predictions.std():.4f}")
        print(f"  Min: {predictions.min():.4f}")
        print(f"  Max: {predictions.max():.4f}")
        print(f"Predicted positive cases: {(class_predictions == 1).sum()}/{len(class_predictions)}")
    elif task_type == "multiclass":
        print(f"Raw model outputs (logits):")
        print(f"  Shape: {raw_outputs.shape}")
        print(f"  Mean: {raw_outputs.mean():.4f}")
        print(f"  Std: {raw_outputs.std():.4f}")
        print(f"Predicted probabilities:")
        print(f"  Shape: {predictions.shape}")
        print(f"  Mean: {predictions.mean():.4f}")
        print(f"  Std: {predictions.std():.4f}")
        print(f"Class distribution:")
        unique, counts = np.unique(class_predictions, return_counts=True)
        for cls, count in zip(unique, counts):
            print(f"  Class {cls}: {count}/{len(class_predictions)} ({100*count/len(class_predictions):.1f}%)")
    else:
        print(f"Predicted values:")
        print(f"  Mean: {raw_outputs.flatten().mean():.4f}")
        print(f"  Std: {raw_outputs.flatten().std():.4f}")
        print(f"  Min: {raw_outputs.flatten().min():.4f}")
        print(f"  Max: {raw_outputs.flatten().max():.4f}")

def calculate_metrics(y_true, raw_outputs, predictions, class_predictions, task_type, dataset_name):
    """
    Calculate appropriate metrics based on task type
    """
    metrics = {
        "dataset": dataset_name,
        "task_type": task_type,
        "n_samples": len(y_true),
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        if task_type == "regression":
            # Regression metrics
            y_pred = raw_outputs.flatten()
            y_true_flat = y_true.flatten()
            
            metrics.update({
                "mae": float(mean_absolute_error(y_true_flat, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_true_flat, y_pred))),
                "r2_score": float(r2_score(y_true_flat, y_pred)),
                "mean_prediction": float(y_pred.mean()),
                "std_prediction": float(y_pred.std()),
                "mean_true": float(y_true_flat.mean()),
                "std_true": float(y_true_flat.std())
            })
            
        elif task_type == "classification":
            # Binary classification metrics
            y_true_flat = y_true.flatten().astype(int)
            y_pred_probs = predictions.flatten()
            y_pred_class = class_predictions.flatten().astype(int)
            
            # Basic metrics
            metrics.update({
                "accuracy": float(accuracy_score(y_true_flat, y_pred_class)),
                "precision": float(precision_score(y_true_flat, y_pred_class, average='binary', zero_division=0)),
                "recall": float(recall_score(y_true_flat, y_pred_class, average='binary', zero_division=0)),
                "f1_score": float(f1_score(y_true_flat, y_pred_class, average='binary', zero_division=0))
            })
            
            # AUC (only if both classes present)
            if len(np.unique(y_true_flat)) > 1:
                metrics["auc"] = float(roc_auc_score(y_true_flat, y_pred_probs))
            else:
                metrics["auc"] = None
                
            # Class distribution
            unique, counts = np.unique(y_true_flat, return_counts=True)
            metrics["true_class_distribution"] = {f"class_{int(cls)}": int(count) for cls, count in zip(unique, counts)}
            
            unique_pred, counts_pred = np.unique(y_pred_class, return_counts=True)
            metrics["predicted_class_distribution"] = {f"class_{int(cls)}": int(count) for cls, count in zip(unique_pred, counts_pred)}
            
        elif task_type == "multiclass":
            # Multiclass classification metrics
            y_true_flat = y_true.flatten().astype(int)
            y_pred_class = class_predictions.flatten().astype(int)
            y_pred_probs = predictions  # Shape: (n_samples, n_classes)
            
            # Basic metrics
            metrics.update({
                "accuracy": float(accuracy_score(y_true_flat, y_pred_class)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true_flat, y_pred_class)),
                "macro_f1": float(f1_score(y_true_flat, y_pred_class, average='macro', zero_division=0)),
                "weighted_f1": float(f1_score(y_true_flat, y_pred_class, average='weighted', zero_division=0))
            })
            
            # Macro AUC (only if all classes present and probabilities available)
            try:
                if len(np.unique(y_true_flat)) > 1 and y_pred_probs.shape[1] > 1:
                    metrics["macro_auc"] = float(roc_auc_score(y_true_flat, y_pred_probs, multi_class='ovr', average='macro'))
                else:
                    metrics["macro_auc"] = None
            except:
                metrics["macro_auc"] = None
            
            # Per-class metrics
            try:
                report = classification_report(y_true_flat, y_pred_class, output_dict=True, zero_division=0)
                per_class_metrics = {}
                for class_label in report:
                    if class_label.isdigit():
                        per_class_metrics[f"class_{class_label}"] = {
                            "precision": float(report[class_label]["precision"]),
                            "recall": float(report[class_label]["recall"]),
                            "f1_score": float(report[class_label]["f1-score"]),
                            "support": int(report[class_label]["support"])
                        }
                metrics["per_class_metrics"] = per_class_metrics
            except:
                metrics["per_class_metrics"] = {}
            
            # Class distribution
            unique, counts = np.unique(y_true_flat, return_counts=True)
            metrics["true_class_distribution"] = {f"class_{int(cls)}": int(count) for cls, count in zip(unique, counts)}
            
            unique_pred, counts_pred = np.unique(y_pred_class, return_counts=True)
            metrics["predicted_class_distribution"] = {f"class_{int(cls)}": int(count) for cls, count in zip(unique_pred, counts_pred)}
            
    except Exception as e:
        metrics["error"] = f"Error calculating metrics: {str(e)}"
        print(f"Warning: Error calculating metrics for {dataset_name}: {str(e)}")
    
    return metrics

def main():
    """
    Main inference pipeline
    """
    print("=== MODEL INFERENCE PIPELINE ===")
    print(f"Device: {DEVICE}")
    print(f"Datasets to run: {DATASETS_TO_RUN}")
    print("="*50)
    
    # Store all metrics
    all_metrics = {
        "run_info": {
            "timestamp": datetime.now().isoformat(),
            "device": DEVICE,
            "datasets_processed": [],
            "total_datasets": len(DATASETS_TO_RUN),
            "simclr_checkpoint": SIMCLR_CKPT_PATH
        },
        "dataset_metrics": {}
    }
    
    for dataset_key in DATASETS_TO_RUN:
        if dataset_key not in DATASETS:
            print(f"Error: Dataset '{dataset_key}' not found in DATASETS configuration!")
            continue
            
        print(f"Running inference for {dataset_key}")
        config = DATASETS[dataset_key]
        
        print(f"  Task type: {config['task_type']}")
        print(f"  Image type: {config['image_type']}")
        print(f"  Test CSV: {config['test_csv_path']}")
        print(f"  Output CSV: {config['output_csv_path']}")
        print("-" * 40)
        
        try:
            # Load model
            model = load_model(
                checkpoint_path=config["checkpoint_path"],
                simclr_ckpt_path=SIMCLR_CKPT_PATH,
                task_type=config["task_type"],
                image_type=config["image_type"],
                num_classes=config["num_classes"]
            )
            
            # Create test dataset
            test_dataset, collate_fn = create_test_dataset(
                csv_path=config["test_csv_path"],
                root_dir=config["root_dir"],
                image_type=config["image_type"],
                image_size=IMAGE_SIZE,
                dataset_name=dataset_key
            )
            
            # Run inference
            raw_outputs, predictions, class_predictions, labels = run_inference(
                model=model,
                dataset=test_dataset,
                collate_fn=collate_fn,
                batch_size=BATCH_SIZE,
                task_type=config["task_type"]
            )
            
            # Save results
            save_predictions(
                csv_path=config["test_csv_path"],
                raw_outputs=raw_outputs,
                predictions=predictions,
                class_predictions=class_predictions,
                output_path=config["output_csv_path"],
                task_type=config["task_type"]
            )
            
            # Calculate metrics
            metrics = calculate_metrics(
                y_true=labels,
                raw_outputs=raw_outputs,
                predictions=predictions,
                class_predictions=class_predictions,
                task_type=config["task_type"],
                dataset_name=dataset_key
            )
            
            # Add config info to metrics
            metrics["config"] = {
                "checkpoint_path": config["checkpoint_path"],
                "test_csv_path": config["test_csv_path"],
                "output_csv_path": config["output_csv_path"],
                "image_type": config["image_type"],
                "num_classes": config["num_classes"]
            }
            
            # Store metrics
            all_metrics["dataset_metrics"][dataset_key] = metrics
            all_metrics["run_info"]["datasets_processed"].append(dataset_key)
            
            print(f" {dataset_key} completed successfully!")
            
            # Print key metrics
            if config["task_type"] == "regression":
                print(f"    MAE: {metrics.get('mae', 'N/A'):.4f}, RMSE: {metrics.get('rmse', 'N/A'):.4f}, R²: {metrics.get('r2_score', 'N/A'):.4f}")
            elif config["task_type"] == "classification":
                print(f"    AUC: {metrics.get('auc', 'N/A'):.4f}, F1: {metrics.get('f1_score', 'N/A'):.4f}, Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
            elif config["task_type"] == "multiclass":
                print(f"    Balanced Acc: {metrics.get('balanced_accuracy', 'N/A'):.4f}, Macro F1: {metrics.get('macro_f1', 'N/A'):.4f}, Macro AUC: {metrics.get('macro_auc', 'N/A'):.4f}")
            
        except Exception as e:
            print(f"Error processing {dataset_key}: {str(e)}")
            all_metrics["dataset_metrics"][dataset_key] = {
                "dataset": dataset_key,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            continue
    
    # Save metrics to JSON
    metrics_output_path = "inference/eval_results.json"
    try:
        with open(metrics_output_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        print(f" Metrics saved to: {metrics_output_path}")
    except Exception as e:
        print(f" Error saving metrics: {str(e)}")
    
    # Print summary
    print("\n=== INFERENCE PIPELINE COMPLETED ===")
    print("Processed datasets:")
    for dataset_key in DATASETS_TO_RUN:
        if dataset_key in DATASETS and dataset_key in all_metrics["dataset_metrics"]:
            if "error" not in all_metrics["dataset_metrics"][dataset_key]:
                print(f"  {dataset_key}: {DATASETS[dataset_key]['output_csv_path']}")
            else:
                print(f"   {dataset_key}: Failed")
    print(f" Comprehensive metrics saved to: {metrics_output_path}")
    print("="*50)

if __name__ == "__main__":
    main() 