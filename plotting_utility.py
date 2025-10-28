"""
Performance plotting utilities for DDPM and Gradient Boosting models
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_curve, auc, precision_recall_curve,
    accuracy_score, f1_score
)
from sklearn.preprocessing import label_binarize
import tensorflow as tf


class PerformancePlotter:
    """Comprehensive performance visualization for hybrid models"""
    
    def __init__(self, save_dir='plots/performance'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        sns.set_style('whitegrid')
        
    def plot_training_metrics(self, train_losses, val_losses=None, 
                             train_acc=None, val_acc=None, 
                             model_name='Model', show=True):
        """Plot training and validation loss/accuracy curves"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        epochs = range(1, len(train_losses) + 1)
        axes[0].plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=4)
        if val_losses is not None:
            axes[0].plot(epochs, val_losses, 'r-s', label='Val Loss', linewidth=2, markersize=4)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title(f'{model_name} - Training Loss', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy plot
        if train_acc is not None:
            axes[1].plot(epochs, train_acc, 'b-o', label='Train Acc', linewidth=2, markersize=4)
            if val_acc is not None:
                axes[1].plot(epochs, val_acc, 'r-s', label='Val Acc', linewidth=2, markersize=4)
            axes[1].set_xlabel('Epoch', fontsize=12)
            axes[1].set_ylabel('Accuracy', fontsize=12)
            axes[1].set_title(f'{model_name} - Training Accuracy', fontsize=14, fontweight='bold')
            axes[1].legend(fontsize=10)
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, 'Accuracy data not available', 
                        ha='center', va='center', fontsize=12)
            axes[1].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'{model_name.lower()}_training_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved training curves: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names=None, 
                             model_name='Model', normalize=True, show=True):
        """Plot confusion matrix with optional normalization"""
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = f'{model_name} - Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = f'{model_name} - Confusion Matrix'
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Proportion' if normalize else 'Count'})
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, f'{model_name.lower()}_confusion_matrix.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved confusion matrix: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_classification_report(self, y_true, y_pred, class_names=None,
                                   model_name='Model', show=True):
        """Generate and plot classification metrics"""
        report = classification_report(y_true, y_pred, target_names=class_names, 
                                      output_dict=True, zero_division=0)
        
        # Extract metrics
        metrics_df = pd.DataFrame(report).transpose()
        metrics_df = metrics_df[metrics_df.index.isin(class_names or [])]
        
        fig, ax = plt.subplots(figsize=(10, max(6, len(metrics_df) * 0.5)))
        
        x = np.arange(len(metrics_df))
        width = 0.25
        
        ax.bar(x - width, metrics_df['precision'], width, label='Precision', alpha=0.8)
        ax.bar(x, metrics_df['recall'], width, label='Recall', alpha=0.8)
        ax.bar(x + width, metrics_df['f1-score'], width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Classes', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'{model_name} - Classification Metrics', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_df.index, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'{model_name.lower()}_classification_report.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved classification report: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return report
    
    def plot_roc_curves(self, y_true, y_pred_proba, class_names=None,
                       model_name='Model', show=True):
        """Plot ROC curves for multi-class classification"""
        n_classes = y_pred_proba.shape[1]
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_proba.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # Plot
        plt.figure(figsize=(10, 8))
        
        # Plot micro-average
        plt.plot(fpr["micro"], tpr["micro"],
                label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})',
                color='deeppink', linestyle=':', linewidth=3)
        
        # Plot per-class curves
        colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
        for i, color in zip(range(n_classes), colors):
            label = class_names[i] if class_names else f'Class {i}'
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'{label} (AUC = {roc_auc[i]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'{model_name} - ROC Curves', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=9)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'{model_name.lower()}_roc_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved ROC curves: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return roc_auc
    
    def plot_feature_importance(self, feature_names, importances, 
                               model_name='Model', top_n=20, show=True):
        """Plot feature importance scores"""
        # Sort features by importance
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(10, max(6, top_n * 0.3)))
        plt.barh(range(len(indices)), importances[indices], align='center', alpha=0.8)
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Importance Score', fontsize=12)
        plt.title(f'{model_name} - Top {top_n} Feature Importances', 
                 fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'{model_name.lower()}_feature_importance.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved feature importance: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_ddpm_generated_samples(self, generated_images, real_images=None,
                                   class_labels=None, n_samples=16, show=True):
        """Plot generated DDPM samples in a grid"""
        n_samples = min(n_samples, len(generated_images))
        grid_size = int(np.ceil(np.sqrt(n_samples)))
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
        axes = axes.flatten()
        
        for i in range(n_samples):
            axes[i].imshow(generated_images[i], cmap='viridis')
            if class_labels is not None:
                axes[i].set_title(f'Class: {class_labels[i]}', fontsize=9)
            axes[i].axis('off')
        
        # Hide extra subplots
        for i in range(n_samples, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('DDPM Generated Samples', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, 'ddpm_generated_samples.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved generated samples: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_comparison_real_vs_generated(self, real_images, generated_images,
                                         n_pairs=5, show=True):
        """Side-by-side comparison of real and generated images"""
        fig, axes = plt.subplots(2, n_pairs, figsize=(15, 6))
        
        for i in range(n_pairs):
            # Real images
            axes[0, i].imshow(real_images[i], cmap='viridis')
            axes[0, i].set_title('Real', fontsize=10)
            axes[0, i].axis('off')
            
            # Generated images
            axes[1, i].imshow(generated_images[i], cmap='viridis')
            axes[1, i].set_title('Generated', fontsize=10)
            axes[1, i].axis('off')
        
        plt.suptitle('Real vs Generated Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, 'ddpm_real_vs_generated.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved comparison: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_model_comparison(self, results_dict, metric='accuracy', show=True):
        """Compare multiple models on a given metric"""
        models = list(results_dict.keys())
        values = [results_dict[m][metric] for m in models]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, values, alpha=0.8, color=plt.cm.Set3(np.linspace(0, 1, len(models))))
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.4f}',
                    ha='center', va='bottom', fontsize=10)
        
        plt.xlabel('Model', fontsize=12)
        plt.ylabel(metric.capitalize(), fontsize=12)
        plt.title(f'Model Comparison - {metric.capitalize()}', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.ylim([0, max(values) * 1.15])
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'model_comparison_{metric}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved model comparison: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_denoising_progression(self, hybrid_model,
        timesteps=None, n_samples=1, class_id=0,
        aux_val=0.0, show=True
    ):
        """
        Visualize the DDPM denoising process step by step.
        Displays images at selected timesteps from pure noise to denoised output.
        """
        
        if timesteps is None:
            timesteps = [0, 50, 100, 200, 400, 600, 800, hybrid_model.ddpm_timesteps - 1]

        n = n_samples
        class_ids = [class_id] * n
        aux_scalars = np.full((n, 1), aux_val, dtype=np.float32)
        
        # initialize random noise
        shape = (64, 64, 1)  # adjust if your histograms differ
        x_t = tf.random.normal((n, *shape), dtype=tf.float32)
        
        betas = tf.cast(hybrid_model.betas, tf.float32)
        alphas = tf.cast(hybrid_model.alphas, tf.float32)
        alphas_cumprod = tf.cast(hybrid_model.alphas_cumprod, tf.float32)
        alphas_cumprod_prev = tf.cast(hybrid_model.alphas_cumprod_prev, tf.float32)
        
        cls_ids = tf.convert_to_tensor(class_ids, dtype=tf.int32)
        aux_scalars = tf.convert_to_tensor(aux_scalars, dtype=tf.float32)
        
        denoised_images = []
        
        for t_idx in reversed(range(hybrid_model.ddpm_timesteps)):
            t_steps = tf.fill((n,), tf.cast(t_idx, tf.int32))
            pred_noise = hybrid_model.diffusion_model([x_t, t_steps, cls_ids, aux_scalars], training=False)
        
            alpha_t = alphas[t_idx]
            alpha_bar_t = alphas_cumprod[t_idx]
            beta_t = betas[t_idx]
        
            mean = (1.0 / tf.sqrt(alpha_t)) * (x_t - (beta_t / tf.sqrt(1.0 - alpha_bar_t)) * pred_noise)
        
            if t_idx > 0:
                noise = tf.random.normal(tf.shape(x_t))
                var = beta_t * (1.0 - alphas_cumprod_prev[t_idx]) / (1.0 - alpha_bar_t)
                x_t = mean + tf.sqrt(var) * noise
            else:
                x_t = mean
        
            if t_idx in timesteps:
                denoised_images.append(np.squeeze(x_t.numpy()))
        
        # Plot the progression
        fig, axes = plt.subplots(1, len(denoised_images), figsize=(2.5 * len(denoised_images), 3))
        for i, img in enumerate(denoised_images):
            axes[i].imshow(img, cmap="viridis")
            axes[i].axis("off")
            axes[i].set_title(f"t={timesteps[i]}")
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, 'ddpm_denoising_progression.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    def plot_denoised_sample(
        self, real_img, hybrid_model,
        timesteps=None, class_id=0, aux_val=0.0, show=True
    ):
        """
        Plot the denoising of a single real test sample using the trained DDPM.
        Shows the original, noised, and denoised versions across selected timesteps.
        """
        
        if timesteps is None:
            timesteps = [900, 700, 500, 300, 100, 0]
        
        #Prepare inputs
        real_img = np.expand_dims(np.squeeze(real_img), axis=0)  # shape (1, H, W)
        real_img = tf.convert_to_tensor(real_img, dtype=tf.float32)
        
        cls_ids = tf.convert_to_tensor([class_id], dtype=tf.int32)
        aux_scalars = tf.convert_to_tensor([[aux_val]], dtype=tf.float32)
        
        betas = tf.cast(hybrid_model.betas, tf.float32)
        alphas_cumprod = tf.cast(hybrid_model.alphas_cumprod, tf.float32)
        
        #Plot setup
        fig, axes = plt.subplots(1, len(timesteps) + 2, figsize=(3 * (len(timesteps) + 2), 3))
        
        #Original image
        axes[0].imshow(np.squeeze(real_img.numpy()), cmap='viridis')
        axes[0].set_title("Original")
        axes[0].axis("off")
        
        #Loop through timesteps (noised + denoised)
        for i, t_idx in enumerate(timesteps):
            t_steps = tf.fill((1,), tf.cast(t_idx, tf.int32))
            alpha_bar_t = alphas_cumprod[t_idx]
            
            noise = tf.random.normal(tf.shape(real_img))
            x_t = tf.sqrt(alpha_bar_t) * real_img + tf.sqrt(1 - alpha_bar_t) * noise
            
            pred_noise = hybrid_model.diffusion_model([x_t, t_steps, cls_ids, aux_scalars], training=False)
            x0_pred = (x_t - tf.sqrt(1 - alpha_bar_t) * pred_noise) / tf.sqrt(alpha_bar_t)
            
            #Handle multi-channel output
            x0_np = x0_pred.numpy()

            #Remove batch dimension
            if x0_np.ndim == 4:
                x0_np = x0_np[0]
            
            #Collapse channels if needed
            if x0_np.ndim == 3 and x0_np.shape[-1] > 1:
                img_to_plot = np.mean(x0_np, axis=-1)
            else:
                img_to_plot = np.squeeze(x0_np)
            
            #Ensure final shape is 2D for matplotlib
            if img_to_plot.ndim != 2:
                img_to_plot = np.mean(img_to_plot, axis=0)
 
            axes[i + 1].imshow(img_to_plot, cmap='viridis')
            axes[i + 1].set_title(f"Denoised t={t_idx}")
            axes[i + 1].axis("off")

        
        #Show the fully denoised last prediction
        x0_np = x0_pred.numpy()
        # Remove batch dimension if present
        if x0_np.ndim == 4:
            x0_np = x0_np[0]
        # Collapse multi-channel outputs to a single 2D image
        if x0_np.ndim == 3 and x0_np.shape[-1] > 1:
            img_to_plot = np.mean(x0_np, axis=-1)
        else:
            img_to_plot = np.squeeze(x0_np)
        # Ensure a valid 2D shape for Matplotlib
        if img_to_plot.ndim != 2:
            img_to_plot = np.mean(img_to_plot, axis=0)

        axes[-1].imshow(np.squeeze(x0_pred.numpy()), cmap='viridis')
        axes[-1].set_title("Final Denoised")
        axes[-1].axis("off")
        
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, "ddpm_denoised_sample.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    def generate_summary_report(self, y_true, y_pred, model_name='Model'):
        """Generate and save a text summary report"""
        report = classification_report(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        summary = f"""
{'='*60}
{model_name} Performance Summary
{'='*60}

Overall Metrics:
  - Accuracy:  {accuracy:.4f}
  - F1-Score:  {f1:.4f}

Classification Report:
{report}

{'='*60}
"""
        
        save_path = os.path.join(self.save_dir, f'{model_name.lower()}_summary.txt')
        with open(save_path, 'w') as f:
            f.write(summary)
        
        print(summary)
        print(f"✓ Saved summary report: {save_path}")
        
        return summary


# Example usage function
def evaluate_and_plot_all(model, X_test, y_test, y_pred_proba=None, 
                          class_names=None, model_name='Model'):
    """
    Comprehensive evaluation and plotting for a trained model
    
    Args:
        model: Trained model object
        X_test: Test features
        y_test: True labels
        y_pred_proba: Predicted probabilities (optional, for ROC curves)
        class_names: List of class names
        model_name: Name of the model for plot titles
    """
    plotter = PerformancePlotter()
    
    # Get predictions
    if hasattr(model, 'predict'):
        y_pred = model.predict(X_test)
    else:
        y_pred = y_test  # Placeholder if model doesn't have predict
    
    # Generate all plots
    print(f"\n{'='*60}")
    print(f"Generating performance plots for {model_name}")
    print(f"{'='*60}\n")
    
    # 1. Confusion Matrix
    plotter.plot_confusion_matrix(y_test, y_pred, class_names, model_name, show=False)
    
    # 2. Classification Report
    plotter.plot_classification_report(y_test, y_pred, class_names, model_name, show=False)
    
    # 3. ROC Curves (if probabilities available)
    if y_pred_proba is not None:
        plotter.plot_roc_curves(y_test, y_pred_proba, class_names, model_name, show=False)
    
    # 4. Feature Importance (if available)
    if hasattr(model, 'feature_importances_'):
        feature_names = [f'Feature {i}' for i in range(len(model.feature_importances_))]
        plotter.plot_feature_importance(feature_names, model.feature_importances_, 
                                       model_name, show=False)
    
    # 5. Summary Report
    plotter.generate_summary_report(y_test, y_pred, model_name)
    
    print(f"\n{'='*60}")
    print(f"All plots saved to: {plotter.save_dir}")
    print(f"{'='*60}\n")
