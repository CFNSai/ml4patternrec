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
from typing import List, Tuple, Union, Optional
from tqdm import tqdm


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
        
        if show:
            plt.show()
        else:
            plt.close()
        print(f"✓ Saved training curves: {save_path}")
    
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
        
        if show:
            plt.show()
        else:
            plt.close()
        print(f"✓ Saved confusion matrix: {save_path}")
    
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
        
        if show:
            plt.show()
        else:
            plt.close()
        print(f"✓ Saved classification report: {save_path}")
        
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
        
        if show:
            plt.show()
        else:
            plt.close()
        print(f"✓ Saved feature importance: {save_path}")
    
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
        
        if show:
            plt.show()
        else:
            plt.close()
        print(f"✓ Saved generated samples: {save_path}")
    
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
        
        if show:
            plt.show()
        else:
            plt.close()
        print(f"✓ Saved comparison: {save_path}")
    
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
        
        if show:
            plt.show()
        else:
            plt.close()
        plt.close()
        print(f"✓ Saved model comparison: {save_path}")
    
    def plot_denoising_progression(self, hybrid_model, timesteps=1000, n_samples=1,
                               class_id=None, aux_val=0.0, show=True):
        """
        Visualize progressive denoising using the trained DDPM model.
        Shows how the model reconstructs structure from noise.
 
        Args:
            hybrid_model: trained GradientBoostHybrid instance
            timesteps: number of diffusion steps (default: 1000)
            n_samples: number of samples to visualize (default: 1)
            class_id: optional integer class label for conditional DDPM
            aux_val: optional scalar conditioning value
            show: display plots interactively
        """
 
        print("\n[DDPM] Starting denoising progression visualization...")
 
        # === Load or construct schedule ===
        betas = hybrid_model.ddpm_betas if hasattr(hybrid_model, "ddpm_betas") else tf.linspace(1e-4, 0.02, timesteps)
        betas = tf.cast(betas, tf.float32)
        alphas = 1.0 - betas
        alphas_cumprod = tf.math.cumprod(alphas, axis=0)
 
        # === Initialize latent noise ===
        hist_shape = hybrid_model.hist_shape
        x_t = tf.random.normal((n_samples, *hist_shape, 1), dtype=tf.float32)
        print(f"[Info] Starting from random noise: shape={x_t.shape}")
 
        # === Class & auxiliary conditioning ===
        cls_ids = tf.zeros((n_samples,), dtype=tf.int32) if class_id is None else tf.fill((n_samples,), int(class_id))
        aux_scalars = tf.ones((n_samples, 1), dtype=tf.float32) * float(aux_val)
 
        # === Set up visualization ===
        fig, axes = plt.subplots(1, 10, figsize=(20, 2))
        vis_steps = np.linspace(0, timesteps - 1, 10, dtype=int)
        img_buffer = []
 
        # === Reverse (denoising) process ===
        for t_idx in tqdm(reversed(range(timesteps)), desc="Denoising", total=timesteps):
            t_steps = tf.fill((n_samples,), tf.cast(t_idx, tf.int32))
 
            # Predict noise ε_θ(x_t, t)
            eps_pred = hybrid_model.diffusion_model(
                x_t, t_steps, cls_ids, aux_scalars, training=False
            )
            eps_pred = tf.cast(eps_pred, tf.float32)
 
            alpha_t = alphas[t_idx]
            alpha_bar_t = alphas_cumprod[t_idx]
            sqrt_alpha_t = tf.sqrt(alpha_t)
            sqrt_one_minus_alpha_bar_t = tf.sqrt(1.0 - alpha_bar_t)
 
            # Predicted clean image x₀
            x0_pred = (x_t - sqrt_one_minus_alpha_bar_t * eps_pred) / tf.sqrt(alpha_bar_t)
            x0_pred = tf.clip_by_value(x0_pred, 0.0, 1.0)
 
            # Deterministic DDIM step (σ_t = 0)
            if t_idx > 0:
                alpha_bar_prev = alphas_cumprod[t_idx - 1]
                c = tf.sqrt(1.0 - alpha_bar_prev)
                x_t = tf.sqrt(alpha_bar_prev) * x0_pred + c * eps_pred
            else:
                x_t = x0_pred
 
            # Save visualization frame every ~T/10 steps
            if t_idx in vis_steps:
                img = np.squeeze(x_t[0].numpy())
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                img_buffer.append(img)
 
        # === Plot denoising frames ===
        for i, img in enumerate(img_buffer):
            axes[i].imshow(img, cmap='viridis')
            axes[i].set_title(f"t={vis_steps[i]}")
            axes[i].axis("off")
 
        plt.suptitle("Denoising Progression (Reverse Diffusion)")
        plt.tight_layout()
 
        # === Save ===
        save_path = os.path.join(self.save_dir, "ddpm_denoising_progression.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
        print(f"✓ Saved denoising progression → {save_path}")
 
        return img_buffer[-1]  # return final reconstruction


    def plot_denoised_sample(
        self,
        real_img,
        hybrid_model,
        timesteps: int = None,
        class_id: int = None,
        aux_val: float = 0.0,
        noise_level_step: int = None,
        show: bool = True,
        save_frames: bool = True,
        frames_dir: str = None,
        make_gif: bool = True,
        gif_name: str = "ddpm_denoising.gif",
        n_intermediate: int = 10,
        ddim_eta: float = 0.0,
        return_as_gif_bytes: bool = False,
    ):
        """
        Visualize DDIM-style deterministic/stochastic denoising with optional frame saving and GIF.

        Args:
            real_img: np.ndarray or tf.Tensor; (H,W) or (H,W,1)
            hybrid_model: trained GradientBoostHybrid instance
            timesteps: total diffusion timesteps (defaults to hybrid_model.ddpm_timesteps)
            class_id: optional conditioning class id
            aux_val: auxiliary scalar conditioning (pT, energy)
            noise_level_step: timestep to start denoising from (defaults to T//2)
            show: whether to show plots
            save_frames: whether to save each intermediate frame
            frames_dir: directory to save intermediate frames
            make_gif: whether to create GIF from saved frames
            gif_name: filename for GIF
            n_intermediate: number of intermediate frames (set 0 for all)
            ddim_eta: DDIM eta parameter (0 = deterministic, >0 adds stochasticity)
        Returns:
            dict: {'reconstructed', 'frames', 'frame_paths', 'gif_path'}
        """
        # --- setup and diffusion parameters ---
        T = timesteps or getattr(hybrid_model, "ddpm_timesteps", None)
        if T is None:
            raise RuntimeError("timesteps must be provided or hybrid_model.ddpm_timesteps must exist.")

        if hasattr(hybrid_model, "alphas_cumprod") and hybrid_model.alphas_cumprod is not None:
            alphas_cumprod_np = np.asarray(hybrid_model.alphas_cumprod, dtype=np.float32)
        else:
            betas = np.linspace(1e-4, 2e-2, T, dtype=np.float32)
            alphas = 1.0 - betas
            alphas_cumprod_np = np.cumprod(alphas, axis=0).astype(np.float32)

        # --- input prep ---
        if isinstance(real_img, tf.Tensor):
            real_img = real_img.numpy()
        x0 = np.asarray(real_img, dtype=np.float32)

        if x0.ndim == 2:
            x0 = x0[..., None]
        if x0.ndim == 3 and x0.shape[-1] != 1:
            x0 = np.mean(x0, axis=-1, keepdims=True)
        x0 = np.expand_dims(x0, axis=0) if x0.ndim == 3 else x0[:1]
        x0 = (x0 - x0.min()) / (x0.max() - x0.min() + 1e-8)
        x0 = x0.astype(np.float32)
        n = x0.shape[0]

        # --- conditioning ---
        if class_id is None:
            cls_ids = tf.zeros((n,), dtype=tf.int32)
        else:
            if getattr(hybrid_model, "label_encoder", None):
                le = hybrid_model.label_encoder
                try:
                    cls_ids = tf.convert_to_tensor(le.transform([class_id]), dtype=tf.int32)
                except Exception:
                    cls_ids = tf.fill((n,), int(class_id))
            else:
                cls_ids = tf.fill((n,), int(class_id))
        aux_scalars = tf.fill((n, 1), float(aux_val))

        # --- forward diffuse to noisy sample ---
        t_T = noise_level_step if noise_level_step is not None else T // 2
        alpha_bar_T = tf.cast(alphas_cumprod_np[t_T], tf.float32)
        noise = tf.random.normal(shape=tf.shape(x0), dtype=tf.float32)
        x_t = tf.sqrt(alpha_bar_T) * x0 + tf.sqrt(1.0 - alpha_bar_T) * noise

        # --- frame bookkeeping ---
        if frames_dir is None:
            frames_dir = os.path.join(self.save_dir, "ddpm_frames")
        if save_frames:
            os.makedirs(frames_dir, exist_ok=True)

        frames, frame_paths = [], []
        keep_ts = (
            sorted(
                list(
                    {int(round(t_T * i / max(1, n_intermediate - 1))) for i in range(max(1, n_intermediate))}
                )
            )
            if n_intermediate and n_intermediate > 0
            else list(range(t_T, -1, -1))
        )

        # --- reverse denoising (DDIM update) ---
        x_current = tf.identity(x_t)
        for t_idx in reversed(range(0, t_T + 1)):
            t_steps = tf.fill((n,), tf.cast(t_idx, tf.int32))
            pred_noise = hybrid_model.diffusion_model(x_current, t_steps, cls_ids, aux_scalars, training=False)
            pred_noise = tf.cast(pred_noise, tf.float32)

            alpha_bar_t = tf.cast(alphas_cumprod_np[t_idx], tf.float32)
            sqrt_alpha_bar_t = tf.sqrt(alpha_bar_t)
            sqrt_one_minus_alpha_bar_t = tf.sqrt(1.0 - alpha_bar_t)
            x0_pred = (x_current - sqrt_one_minus_alpha_bar_t * pred_noise) / (sqrt_alpha_bar_t + 1e-12)
            x0_pred = tf.clip_by_value(x0_pred, 0.0, 1.0)

            if t_idx > 0:
                alpha_bar_prev = tf.cast(alphas_cumprod_np[t_idx - 1], tf.float32)
                sqrt_alpha_bar_prev = tf.sqrt(alpha_bar_prev)
                ratio = (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t + 1e-12)
                sigma_t = ddim_eta * tf.sqrt(tf.clip_by_value(ratio, 0.0, 1e6)) * tf.sqrt(
                    tf.clip_by_value(1.0 - alpha_bar_t / (alpha_bar_prev + 1e-12), 0.0, 1.0)
                )
                coeff_x0 = sqrt_alpha_bar_prev
                coeff_eps = tf.sqrt(tf.clip_by_value(1.0 - alpha_bar_prev - sigma_t ** 2, 0.0, 1.0))
                noise_term = tf.random.normal(shape=tf.shape(x_current), dtype=tf.float32) if float(sigma_t) > 0 else tf.zeros_like(x_current)
                x_current = coeff_x0 * x0_pred + coeff_eps * pred_noise + sigma_t * noise_term
            else:
                x_current = x0_pred

            if (t_idx in keep_ts) or (t_idx == 0 and 0 not in keep_ts):
                img_np = x_current.numpy().squeeze()
                if img_np.ndim == 3 and img_np.shape[-1] > 1:
                    img_np = np.mean(img_np, axis=-1)
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
                frames.append(img_np)
                if save_frames:
                    fpath = os.path.join(frames_dir, f"frame_t{t_idx:04d}.png")
                    plt.imsave(fpath, img_np, cmap="viridis")
                    frame_paths.append(fpath)

        # --- final reconstruction and visualization ---
        x_rec = x_current.numpy().squeeze()
        if x_rec.ndim == 3 and x_rec.shape[-1] > 1:
            x_rec = np.mean(x_rec, axis=-1)
        x_rec = (x_rec - x_rec.min()) / (
            x_rec.max() - x_rec.min() + 1e-8
        )

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(x0[0, ..., 0], cmap="viridis")
        axes[0].set_title("Original")
        axes[1].imshow(x_t.numpy()[0, ..., 0], cmap="viridis")
        axes[1].set_title(f"Noisy (t={t_T})")
        axes[2].imshow(x_rec, cmap="viridis")
        axes[2].set_title("Reconstructed")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()

        summary_path = os.path.join(self.save_dir, "ddpm_reconstruction_comparison.png")
        plt.savefig(summary_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()

        gif_path = None
        gif_bytes = None
 
        if make_gif and frame_paths:
            try:
                import imageio
                gif_path = os.path.join(frames_dir, gif_name)
                imgs = [imageio.imread(f) for f in frame_paths]
 
                if return_as_gif_bytes:
                    buf = io.BytesIO()
                    imageio.mimsave(buf, imgs, format="GIF", duration=0.08)
                    gif_bytes = buf.getvalue()
                else:
                    imageio.mimsave(gif_path, imgs, duration=0.08)
 
            except Exception as e:
                print(f"[GIF ERROR] {e}")
 
        return {
            "reconstructed": x_rec,
            "frames": frames,
            "frame_paths": frame_paths,
            "gif_path": gif_path,
            "gif_bytes": gif_bytes,
        }

    ################################################
    #         Performance Summary
    ################################################
    def plot_joint_performance(self, gbhm_metrics, xgb_metrics, ddpm_metrics, save=True):
        """
        Combine GBHM/XGB + DDPM metrics into one figure.
        gbhm_metrics: dict with keys ['accuracy', 'precision', 'recall', 'f1']
        xgb_metrics: same structure
        ddpm_metrics: dict with keys ['loss_curve', 'final_loss', 'recon_mse']
        """
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))

        # GBHM vs XGB Accuracy
        axs[0,0].bar(["GBHM","XGB"],
                     [gbhm_metrics["accuracy"], xgb_metrics["accuracy"]])
        axs[0,0].set_title("Classifier Accuracy Comparison")
        axs[0,0].set_ylim(0, 1)

        # DDPM loss curve
        axs[0,1].plot(ddpm_metrics["loss_curve"])
        axs[0,1].set_title(f"DDPM Noise-Prediction Loss (final={ddpm_metrics['final_loss']:.4f})")

        # Reconstruction MSE
        axs[1,0].bar(["Reconstruction MSE"], [ddpm_metrics["recon_mse"]])
        axs[1,0].set_title("DDPM Reconstruction Error")
        axs[1,0].set_ylim(0, max(ddpm_metrics["recon_mse"],0.05))

        # Classifier consistency after DDPM reconstruction
        axs[1,1].bar(["GBHM Consistency (%)"], [ddpm_metrics["consistency"]*100.0])
        axs[1,1].set_ylim(0, 100)
        axs[1,1].set_title("Do DDPM Reconstructions Preserve Classifier Predictions?")

        fig.tight_layout()

        if save:
            fig.savefig(os.path.join(self.save_dir, "joint_performance.png"), dpi=200)

        return fig

    def generate_combined_report(self, gbhm_metrics, xgb_metrics, ddpm_metrics, label_encoder):
        """
        Returns a multi-section summary text report.
        """

        report = []
        report.append("==== Combined Model Report ====\n")

        # Label encoder details
        report.append("LABEL ENCODER")
        report.append(f"  Classes: {list(label_encoder.classes_)}\n")

        # GBHM summary
        report.append("GBHM PERFORMANCE")
        for k,v in gbhm_metrics.items():
            report.append(f"  {k}: {v:.4f}")
        report.append("")

        # XGB summary
        report.append("XGBOOST PERFORMANCE")
        for k,v in xgb_metrics.items():
            report.append(f"  {k}: {v:.4f}")
        report.append("")

        # DDPM summary
        report.append("DDPM PERFORMANCE")
        report.append(f"  Final loss: {ddpm_metrics['final_loss']:.4f}")
        report.append(f"  Reconstruction MSE: {ddpm_metrics['recon_mse']:.4f}")
        report.append(f"  Classifier consistency: {ddpm_metrics['consistency']*100:.2f}%")
        report.append("")

        return "\n".join(report)

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
