'''
This is utility scheduler for DDPM
Class also handles loading and rebinning TH2
'''
import os
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec, style
import matplotlib.colors as mcolors
import seaborn as sns
import uproot as ur
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.preprocessing import LabelEncoder
from typing import List, Tuple, Union, Optional


class DDPM_utils:
    #####################
    # Schedule Utilities
    #####################
    def make_beta_schedule(timesteps: int, beta_start=1e-4, beta_end=2e-2, schedule='cosine'):
        """Return a beta schedule for diffusion."""
        if schedule == 'linear':
            betas = np.linspace(beta_start, beta_end, timesteps, dtype=np.float32)

        elif schedule == 'quadratic':
            betas = (np.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2).astype(np.float32)

        elif schedule == 'cosine':
            steps = np.linspace(0, timesteps, timesteps + 1, dtype=np.float32)
            alphas = np.cos(((steps / timesteps) + 0.008) / 1.008 * np.pi / 2) ** 2
            alphas = alphas / alphas[0]
            betas = 1 - (alphas[1:] / alphas[:-1])
            betas = np.clip(betas, 1e-5, 0.999).astype(np.float32)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        return betas

    def _ensure_dir(path):
        """Utility to ensure a directory exists."""
        os.makedirs(path, exist_ok=True)
        return path

    def extract(a, t, x_shapes):
        '''
        Extract coeff. for batch of timesteps t (shapte [B]) from arry (shape [T])
        and reshape to [B,1,1,1] for broadcasting to images
        '''
        batch_size = tf.shape(t)[0]
        out = tf.gather(a, t)
        #reshape to (B,1,1,1)
        return tf.reshape(out, (batch_size, 1, 1, 1))

    #Time embedding (sinusoidal + dense)
    @staticmethod
    def sinusoidal_embedding(timesteps, dim):
        #timesteps: (...,)int32
        timesteps = tf.cast(timesteps, tf.float32)
        half = dim//2
        freqs = tf.exp(-np.log(10000)*(tf.range(0,half,dtype=tf.float32)/float(half)))
        args = tf.cast(tf.expand_dims(timesteps, -1),tf.float32)*tf.reshape(freqs,(1,-1))
        emb = tf.concat([tf.sin(args),tf.cos(args)],axis=-1)
        if dim % 2 ==1:
            emb =tf.pad(emb, [[0,0],[0,1]])
        #shape (B,dim)
        return emb

    '''
    Small UNet-like model (conditioned on timestep, class and aux scalar)
    '''
    def make_unet2d(
        input_shape=(64,64,1),
        base_channels=64,
        time_embed_dim=128,
        class_embed_dim=32,
        aux_embed_dim=32
    ):
        #Inputs
        img_in = keras.Input(shape=input_shape, name='image')
        t_in = keras.Input(shape=(),dtype=tf.int32, name='timestep')
        cls_in = keras.Input(shape=(),dtype=tf.int32, name='class_id')
        aux_in = keras.Input(shape=(1,), dtype=tf.float32, name='aux_scaler')

        #Produce dense time embedding
        '''
        t_emb = layers.Lambda(lambda : sinusoidal_embedding(x, time_embed_dim))(t_in)
        t_emb = layers.Dense(time_embed_dim, activation='swish')(t_emb)
        t_emb = layers.Dense(time_embed_dim, activation='swish')(t_emb)

        #Class embedding
        n_classes = 1_000
        cls_emb_layer = layers.Embedding(input_dim=64, output_dim=class_embed_dim)

        #aus scalar embedding
        aux_emb = layers.Dense(aux_embed_dim, activation='swish')(aux_in)

        #We'll build a small encoder-decoder with FiLM conditioning (scale+shift) using time+class+aux embeddings
        #Encoder
        x = img_in
        #conv block 1
        x = layers.Conv2D(base_channels, 3, padding='same', activation='relu')(x)
        x = layers.Conv2D(base_channels, 3, padding='same', activation='relu')(x)
        skip1 = x
        #Downsample
        x = layers.AveragePooling2D()(x)
        #conv block 2
        x = layers.Conv2D(base_channels*2, 3, padding='same', activation='relu')(x)
        x = layers.Conv2D(base_channels*2, 3, padding='same', activation='relu')(x)
        skip2 = x
        #Downsample
        x = layers.AveragePooling2D()(x)
        #Bottleneck
        x = layers.Conv2D(base_channels*4, 3, padding='same', activation='relu')(x)
        x = layers.Conv2D(base_channels*4, 3, padding='same', activation='relu')(x)

        #Combine embeddings: concatenate t_emb, aux_emb, class embedding
        #Create conditioning vector via dense
        # shape (B, time+aux+class)
        cond = layers.Concatenate()([t_emb, aux_emb])
        cond = layers.Dense(base_channels*4, activation='swish')(cond)
        cond = layers.Dense(base_channels*4, activation='swish')(cond)
        #Broadcast and add
        cond_b = layers.Reshape((1,1,base_channels*4))(cond)
        x = layers.Add()([x,cond_b])

        #Decoder
        x = layers.UpSampling2D()(x)
        x = layers.Concatenate()([x, skip2])
        x = layers.Conv2D(base_channels*2, 3, padding='same', activation='relu')(x)
        x = layers.Conv2D(base_channels*2, 3, padding='same', activation='relu')(x)

        x = layers.UpSampling2D()(x)
        x = layers.Concatenate()([x, skip1])
        x = layers.Conv2D(base_channels, 3, padding='same', activation='relu')(x)
        x = layers.Conv2D(base_channels, 3, padding='same', activation='relu')(x)

        #predict noise, no activation
        out = layers.Conv2D(1,1,padding='same')(x)
        '''
        #specify output shape explicitly
        t_emb = layers.Lambda(
            lambda x: DDPM_utils.sinusoidal_embedding(x, time_embed_dim),
            output_shape=(time_embed_dim,)
        )(t_in)
        t_emb = layers.Dense(time_embed_dim, activation="swish")(t_emb)
        t_emb = layers.Dense(time_embed_dim, activation="swish")(t_emb)
        aux_emb = layers.Dense(32, activation="swish")(aux_in)

        x = layers.Conv2D(base_channels, 3, padding="same", activation="relu")(img_in)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(base_channels * 2, 3, padding="same", activation="relu")(x)
        x = layers.GlobalAveragePooling2D()(x)

        cls_emb = layers.Embedding(input_dim=64, output_dim=class_embed_dim)(cls_in)
        cls_emb = layers.Flatten()(cls_emb)
        cond = layers.Concatenate()([t_emb, aux_emb, cls_emb])
        cond = layers.Dense(base_channels * 2, activation="relu")(cond)

        x = layers.Concatenate()([x, cond])
        x = layers.Dense(np.prod(input_shape), activation='tanh')(x)
        out = layers.Reshape(input_shape)(x)

        model = keras.Model([img_in, t_in, cls_in, aux_in],out, name='unet2d')
        return model

    '''
    TH2 laoding and rebinning to his_shape
    '''
    def  _load_th2_images(
        inputfiles,
        histogram_name = 'hist2D',
        tree_name = 'Events',
        branch_name = ('posX','posY'),
        aux_scalar_branch = 'pT',
        target = 'pdgID',
        hist_shape = (64,64),
        x_edges = None,
        y_edges = None
    ):
        """
        Return arrays:
            imgs: shape (N, H, W)  -- rebinned & normalized histograms
            labels: shape (N,)      -- integer encoded pdgIDs
            aux: shape (N,1)        -- scalar conditioning (e.g., pT or energy) (float)
        Flow:
            - For each input file, try to read histogram self.histogram_name (TH2).
            - Also read per-event auxiliary scalar + pdgID from TTree (first event-level mapping).
        NOTE: This function expects that histograms in files correspond to events/classes in some meaningful way.
        If your setup is different (e.g., one TH2 per file summarizing many events), adapt accordingly.
        """
        imgs = []
        labels = []
        auxs = []

        # We'll attempt two ways:
        # 1) If a TH2 named self.histogram_name exists, take its 2D bin content as a single image for that file.
        # 2) Otherwise, we fall back to producing histogram images from the tree per-event (expensive).
        for fname in inputfiles:
            with ur.open(fname) as f:
                #Handle multiple possible histogram names
                if isinstance(histogram_name, list):
                    for hname in histogram_name:
                        if hname in f:
                            hist = f[hname]
                            # uproot .to_numpy() returns (values, xedges, yedges)
                            values, xedges, yedges = hist.to_numpy()
                            img = np.array(values, dtype=np.float32)
                            # rebin/rescale to target hist_shape
                            img = DDPM_utils._rebin_image(img, hist_shape)
                            imgs.append(img)
                            
                            # For labels/aux, try reading small sample from TTree: take the mode/first pdgID and mean aux
                            try:
                                tree = f[tree_name]
                                # read small subset
                                arrs = tree.arrays([aux_scalar_branch, target], library="np", entry_stop=100)
                                aux_arr = arrs[aux_scalar_branch]
                                pdg_arr = arrs[target]
                                # fallbacks
                                aux_val = float(np.mean(aux_arr)) if len(aux_arr) > 0 else 0.0
                                pdg_val = int(pdg_arr[0]) if len(pdg_arr) > 0 else 0
                            except Exception:
                                aux_val = 0.0
                                pdg_val = 0
                            auxs.append([aux_val])
                            labels.append(pdg_val)
                else:
                    if histogram_name in f:
                        hist = f[histogram_name]
                        # uproot .to_numpy() returns (values, xedges, yedges)
                        values, xedges, yedges = hist.to_numpy()
                        img = np.array(values, dtype=np.float32)
                        # rebin/rescale to target hist_shape
                        img = DDPM_utils._rebin_image(img, hist_shape)
                        imgs.append(img)
                    
                        # For labels/aux, try reading small sample from TTree: take the mode/first pdgID and mean aux
                        try:
                            tree = f[tree_name]
                            # read small subset
                            arrs = tree.arrays([aux_scalar_branch, target], library="np", entry_stop=100)
                            aux_arr = arrs[aux_scalar_branch]
                            pdg_arr = arrs[target]
                            # fallbacks
                            aux_val = float(np.mean(aux_arr)) if len(aux_arr) > 0 else 0.0
                            pdg_val = int(pdg_arr[0]) if len(pdg_arr) > 0 else 0
                        except Exception:
                            aux_val = 0.0
                            pdg_val = 0
                        auxs.append([aux_val])
                        labels.append(pdg_val)
                    else:
                        # Fall-back: build histogram from events in TTree (slower)
                        tree = f[tree_name]
                        arrays = tree.arrays([branch_name[0],branch_name[1],aux_scalar_branch,target],library="np")
                        x = arrays[branch_name[0]]
                        y = arrays[branch_name[1]]
                        aux_arr = arrays[aux_scalar_branch]
                        pdg_arr = arrays[target]
                        # make 2D histogram using the global x_edges, y_edges
                        H, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
                        img = DDPM_utils._rebin_image(np.array(H, dtype=np.float32), hist_shape)
                        imgs.append(img)
                        auxs.append([float(np.mean(aux_arr)) if len(aux_arr) else 0.0])
                        labels.append(int(pdg_arr[0]) if len(pdg_arr) else 0)

        if not imgs:
            raise RuntimeError("No TH2 images found across input files.")

        imgs = np.stack(imgs, axis=0)  # (N,H,W)
        labels = np.array(labels, dtype=np.int32)
        auxs = np.array(auxs, dtype=np.float32)

        # Normalize images to [0,1] by per-image max to stabilize training
        imgs = imgs / (imgs.max(axis=(1, 2), keepdims=True) + 1e-6)

        # Encode labels
        le = LabelEncoder()
        labels_enc = le.fit_transform(labels)

        return imgs, labels_enc, auxs, le

    def _rebin_image(img: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """
        Resize 2D array `img` to `target_shape` using bilinear interpolation (TensorFlow).
        """
        img_tf = tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(img, dtype=tf.float32), 0), -1)  # (1,H,W,1)
        resized = tf.image.resize(img_tf, target_shape, method="bilinear", antialias=True)
        resized = tf.squeeze(resized, axis=0)  # (H,W,1)
        resized = tf.squeeze(resized, axis=-1).numpy()  # (H,W)
        return resized.astype(np.float32)
    
    #####################################
    ####   Visualization Utilities   ####
    #####################################
    def plot_loss_curve(train_losses, val_losses=None, save_dir="plots/ddpm_training", show=True):
        DDPM_utils._ensure_dir(save_dir)
        plt.figure(figsize=(8,5))
        plt.plot(train_losses, label='Train Loss', lw=2)
        if val_losses is not None:
            plt.plot(val_losses, label='Validation Loss', lw=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('DDPM Training Loss Curve')
        plt.legend()
        plt.grid(alpha=0.3)
        out_path = os.path.join(save_dir, "ddpm_loss_curve.png")
        plt.savefig(out_path, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
        print(f"[Saved] Plot loss curve → {out_path}")

    def plot_beta_schedule(betas, save_dir="plots/ddpm_training", show=True):
        DDPM_utils._ensure_dir(save_dir)
        plt.plot(betas)
        plt.xlabel('Timestep')
        plt.ylabel('$\\beta$')
        plt.title('Diffusion Beta Schedule')
        plt.grid(True)
        out_path = os.path.join(save_dir, "ddpm_beta_schedule.png")
        plt.savefig(out_path, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
        print(f"[Saved] Plot beta scheduler → {out_path}")

        # plot cumulative alpha
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas)
        plt.figure(figsize=(6, 4))
        plt.plot(alphas_cumprod, label="α̅_t", color='darkorange')
        plt.title("Cumulative Alpha $(\\alpha_{t})$")
        plt.xlabel("Timestep")
        plt.ylabel("$\\alpha{_t}$")
        plt.legend()
        plt.grid(alpha=0.3)
        out_path = os.path.join(save_dir, "alpha_cumprod.png")
        plt.savefig(out_path, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
        print(f"[Saved] Cumulative alpha plot → {out_path}")

    @staticmethod
    def plot_noising_process(original_img, noise_fn, num_steps=5, save_dir="plots/ddpm_training", show=True):
        DDPM_utils._ensure_dir(save_dir)
        timesteps = np.linspace(0, 1, num_steps)
        fig, axes = plt.subplots(1, num_steps, figsize=(15,3))
        for i, t in enumerate(timesteps):
            noisy_img = noise_fn(original_img, t)
            axes[i].imshow(noisy_img, cmap='viridis')
            axes[i].set_title(f't={t:.2f}')
            axes[i].axis('off')
        plt.suptitle('Noising Process Visualization')
        out_path = os.path.join(save_dir, "ddpm_noise_loss.png")
        plt.savefig(out_path, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
        print(f"[Saved] Noise loss per step → {out_path}")

    def plot_sampling_progress(num_timesteps,model, shape, num_steps=6,
                               device='cpu', save_dir="plots/ddpm_training", show=True):
        DDPM_utils._ensure_dir(save_dir)
        model.eval()
        imgs = []
        x = torch.randn(shape, device=device)
        timesteps = np.linspace(num_timesteps - 1, 0, num_steps, dtype=int)
        for t in timesteps:
            x = model.p_sample(x, t)
            imgs.append(x.detach().cpu().numpy()[0,0])
        fig, axes = plt.subplots(1, num_steps, figsize=(15,3))
        for i, img in enumerate(imgs):
            axes[i].imshow(img, cmap='viridis')
            axes[i].set_title(f'Step {timesteps[i]}')
            axes[i].axis('off')
        plt.suptitle('Reverse Diffusion Trajectory')
        out_path = os.path.join(save_dir, "ddpm_sampling_progress.png")
        plt.savefig(out_path, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
        print(f"[Saved] Sampling progess → {out_path}")

    def plot_ddpm_noising_example(self, sample_img, betas, save_dir):
        """Visualize the forward diffusion (noising) process."""
        alphas_cumprod = np.cumprod(1 - betas)

        def noise_fn(img, t_scalar):
            t_index = int(t_scalar * (len(betas) - 1))
            alpha_bar = alphas_cumprod[t_index]
            noise = np.random.randn(*img.shape)
            return np.sqrt(alpha_bar) * img + np.sqrt(1 - alpha_bar) * noise

        DDPM_utils.plot_noising_process(sample_img, noise_fn, num_steps=6, save_dir=save_dir, show=False)

    @staticmethod
    def compare_generated_vs_real(real_imgs, generated_imgs, n=5, save_dir="plots/ddpm_training", show=True):
        DDPM_utils._ensure_dir(save_dir)
        real_imgs = np.array(real_imgs)
        generated_imgs = np.array(generated_imgs)
        n = min(n, len(real_imgs), len(generated_imgs))
        fig, axes = plt.subplots(2, n, figsize=(3*n,6))

        #Handle different shapes of axes returned by matplotlib
        if n == 1:
            axes = np.expand_dims(axes, axis=1)  # make it 2D (2, 1)
            
        for i in range(n):
            axes[0,i].imshow(real_imgs[i], cmap='viridis')
            axes[0,i].set_title('Real')
            axes[0,i].axis('off')
            axes[1,i].imshow(generated_imgs[i], cmap='viridis')
            axes[1,i].set_title('Generated')
            axes[1,i].axis('off')
        plt.suptitle('Real vs Generated Histograms')
        out_path = os.path.join(save_dir, "ddpm_generated_vs_real.png")
        plt.savefig(out_path, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
        print(f"[Saved] Generated vs Real → {out_path}")

    @staticmethod
    def plot_latent_space(embeddings, labels, save_dir="plots/ddpm_training", show=True):
        DDPM_utils._ensure_dir(save_dir)
        pca = PCA(n_components=2).fit_transform(embeddings)
        plt.figure(figsize=(6,5))
        scatter = plt.scatter(pca[:,0], pca[:,1], c=labels, cmap='tab10', alpha=0.7)
        plt.title('Latent Space Projection')
        plt.colorbar(scatter)
        out_path = os.path.join(save_dir, "ddpm_latent_space.png")
        plt.savefig(out_path, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
        print(f"[Saved] Latent space → {out_path}")

    @staticmethod
    def plot_hist2D(X, Y, save_dir="plots", name="pfrich_ringsQA.png", show=True):
        DDPM_utils._ensure_dir(save_dir)
        plt.figure(figsize=(6,5))
        scatter = plt.hexbin(X, Y, gridsize=100, cmap='viridis', mincnt=1)
        plt.title('Hits Profile')
        plt.colorbar(scatter)
        plt.xlabel('X [mm]', fontsize=12)
        plt.ylabel('Y [mm]', fontsize=12)
        out_path = os.path.join(save_dir, name)
        plt.savefig(out_path, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
        print(f"[Saved] Hist 2D profile → {out_path}")
