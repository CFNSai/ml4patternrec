import  os
import math
import time
from datetime import datetime

from utilityddpm import DDPM_utils
from hybrid_unetdit_tf import HybridUNetDiT_TF, EMA
from plotting_utility import PerformancePlotter, evaluate_and_plot_all

import pickle
import uproot as ur
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from typing import List, Tuple, Union, Optional

# Try TF (diffusion). If not available, we gracefully disable DDPM functionality.
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False
    tf = None
    keras = None
    layers = None

class GradientBoostHybrid:
    """
    A hybrid AI class model using uproot data and gradient boosting.
    Class combines numerical variables from a TTree and binned data from a TH1 histogram.
    """

    def __init__(
        self,
        inputfile_name: Union[str, List[str]],
        tree_name: str,
        branch_name: Tuple[str,str],
        histogram_name: str,
        features: List[str],
        target: str,
        x_edges: np.array = None,
        y_edges: np.array = None,
        r_max: float = 650,
        chunck_size: int = 100_000,
        max_events: int = 250_000, #safety cap for WSL2
        hist_shape: Tuple[int,int] = (64,64),
        ddpm_timesteps: int = 1000,
        model_dir='saved_models'
    ):
        """
        Initializes the HybridModel.

        Args:
            inputfile_name (str): Path to the ROOT file.
            tree_name (str): Name of the TTree to read numerical variables from.
            histogram_name (str): Name of the TH2 histogram to use.
            features (list): A list of numerical branch names from the TTree.
            target (str): The name of the target variable branch from the TTree.
        """
        if isinstance(inputfile_name, str):
            inputfile_name = [inputfile_name]
        self._inputfiles = inputfile_name
        self.tree_name = tree_name
        self.branch_name = branch_name
        self.histogram_name = histogram_name
        self.features = features
        self.target = target
        self.aux_scalar_branch: Optional[str] = 'pT' #scalar to use for conditioning (energy or pT)
        self.chunck_size = chunck_size
        self.r_max = r_max
        self.max_events = max_events
        #This is the hist range
        self.x_edges = np.linspace(-650,650,50) if x_edges is None else np.asarray(x_edges)
        self.y_edges = np.linspace(-650,650,50) if y_edges is None else np.asarray(y_edges)
        self.scaler = None  # Add a scaler for potential feature scaling

        #classifier
        self.xgb_model = None
        self.gbhm_clf = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.xgb_label_encoder: Optional[LabelEncoder] = None
        self.gbhm_label_encoder: Optional[LabelEncoder] = None

        self.hist_shape = hist_shape
        self.model = None
        self.ddpm_timesteps = ddpm_timesteps
        self.betas = DDPM_utils.make_beta_schedule(self.ddpm_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas,axis=0).astype(np.float32)
        self.alphas_cumprod_prev = np.append(1.0,self.alphas_cumprod[:-1]).astype(np.float32)
        self.shape = hist_shape #target image size for diffusion
        self.diffusion_model = None
        self.diffusion_compiled = False
        self.ema_helper = None

        #Model dir
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

        #Create plotter and generate necessary plots
        self.plotter = PerformancePlotter(save_dir='plots/performance')

    def _get_th2_bin_index(self,row: pd.Series) -> int:
        x_var,y_var = self.branch_name
        x_val = row[x_var]
        y_val = row[y_var]

        if self.r_max and (x_val**2 + y_val**2 > self.r_max**2):
            return -1

        x_bin = np.digitize(x_val, self.x_edges) - 1
        y_bin = np.digitize(y_val, self.y_edges) - 1

        if not (0 <= x_bin < len(self.x_edges) - 1):
            return -1
        if not (0 <= y_bin < len(self.y_edges) - 1):
            return -1

        return y_bin * (len(self.x_edges) -1) + x_bin

    def _compute_radius(self, row: pd.Series) -> float:
        r_xy = np.sqrt(row['posX']**2 + row['posY']**2)
        return r_xy

    def _load_data(self) -> pd.DataFrame:
        """
        Loads data from the ROOT file using uproot.
        Return a single pandas DataFrame with features, target,
        and derived columns ['bin_index', 'radius']
        """
        dfs = []
        processed = 0
        branches = self.features + list(self.branch_name) + [self.aux_scalar_branch,self.target]
        for fname in self._inputfiles:
            for arrays in ur.iterate(
                f"{fname}:{self.tree_name}",
                expressions=branches,
                step_size=self.chunck_size,
                library="np" # ensures dict of numpy arrays
            ):
                #Convert awkward to pandas
                df = pd.DataFrame({k: arrays[k] for k in arrays.keys()})
            
                #Verify that expected columns exist
                missing = [c for c in branches if c not in df.columns]
                if missing:
                    raise KeyError(f"Missing branches in TTree: {missing}")
                #Add derived columns
                df['bin_index'] = df.apply(self._get_th2_bin_index, axis=1)
                df['radius'] = df.apply(self._compute_radius, axis=1)
            
                #Keep only valid entries
                df = df[df['bin_index'] >= 0]
            
                dfs.append(df)
                processed += len(df)
            
                if processed >= self.max_events:
                    print(f"Reached max number of event: {self.max_events}. Stopping early...")
                    break
            if processed >= self.max_events:
                break

        if not dfs:
            raise RuntimeError("***No valid events found in ROOT file!***")

        df_tree = pd.concat(dfs, ignore_index=True)
        #Apply final cap (in case las chunk overshoots)
        if len(df_tree) > self.max_events:
            df_tree = df_tree.iloc[:self.max_events].reset_index(drop=True)
        return df_tree

    #########################
    # Feature Engineering 
    #########################
    def encode_histograms(self, imgs, batch_size=64):
        """
        Return embeddings for hist images using the diffusion model encoder if present,
        otherwise fallback to simple hist_to_features.
        imgs: np.ndarray (N,H,W) or (N,H,W,1)
        """
        imgs = np.asarray(imgs)
        # Normalize if values not in [0,1] (DDPM utils produces normalized images)
        if imgs.ndim == 4 and imgs.shape[-1] == 1:
            imgs_proc = imgs[..., 0]
        else:
            imgs_proc = imgs if imgs.ndim == 3 else np.squeeze(imgs)

        if TF_AVAILABLE and self.diffusion_model is not None and hasattr(self.diffusion_model, 'encoder'):
            encoder = self.diffusion_model.encoder
            out = []
            n = imgs_proc.shape[0]
            for i in range(0, n, batch_size):
                batch = imgs_proc[i:i+batch_size]
                try:
                    emb = encoder(batch, training=False).numpy()
                except Exception:
                    # ensure channel dim
                    if batch.ndim == 3:
                        emb = encoder(batch[..., None], training=False).numpy()
                    else:
                        raise
                out.append(emb)
            embeddings = np.vstack(out)
            self.hist_feature_names = [f'cnn_embed_{i}' for i in range(embeddings.shape[1])]
            return embeddings
        else:
            return self.hist_to_features(imgs_proc, n_flat=64)

    def hist_to_features(self, imgs, n_flat=64):
        """
        Convert histogram images to a compact feature vector.
        imgs: np.ndarray (N,H,W) or (N,H,W,1)
        Returns: np.ndarray shape (N, n_flat + 3)  -> flattened-lowdim + mean + std + max
        """
        imgs = np.asarray(imgs)
        if imgs.ndim == 4 and imgs.shape[-1] == 1:
            imgs = imgs[..., 0]
        N, H, W = imgs.shape

        # 1) Resize each image to small fixed grid (use TF resize if available)
        try:
            imgs_tf = tf.convert_to_tensor(imgs[..., None], dtype=tf.float32)  # (N,H,W,1)
            small = tf.image.resize(imgs_tf, (int(max(4, H//8)), int(max(4, W//8))), method='bilinear')
            small = tf.reshape(small, (N, -1)).numpy()
        except Exception:
            # fallback: simple downsample by slicing (fast)
            step_h = max(1, H // 8)
            step_w = max(1, W // 8)
            small = imgs[:, ::step_h, ::step_w].reshape(N, -1)

        # 2) If needed, pad/truncate to n_flat
        flat = small
        if flat.shape[1] >= n_flat:
            flat = flat[:, :n_flat]
        else:
            pad = np.zeros((N, n_flat - flat.shape[1]), dtype=flat.dtype)
            flat = np.concatenate([flat, pad], axis=1)

        # 3) add simple global stats
        means = imgs.mean(axis=(1, 2)).reshape(N, 1)
        stds = imgs.std(axis=(1, 2)).reshape(N, 1)
        maxs = imgs.max(axis=(1, 2)).reshape(N, 1)

        feats = np.concatenate([flat, means, stds, maxs], axis=1)
        # store feature names for reference
        self.hist_feature_names = [f"hist_feat_{i}" for i in range(feats.shape[1])]
        return feats

    #################################################################
    
    #         ------------ MODEL TRAINING ----------------------    #
    
    #################################################################
    def xgb_train(
            self,max_depth=6,loss='mlogloss',eta=0.1,subsample=0.8,tree_method='hist',
            num_boost_round=200,test_size=0.2, 
            random_state=42, use_histograms: bool = True
        ):
        '''
        Train XGBoost Classifier
        Hyperparameters:
        loss:
         - default: mgloss
        max_depth:
         - default: 6
        eta:
         - default: 0.1
        subsample:
         - default: 0.8
        tree_method:
         - default: hist
        num_boost_round:
         - default: 200
        '''
        df = self._load_data()
        X_tree = df[self.features + ['bin_index', 'radius']]
        y = df[self.target]

        #Map labels -> 0..N-1
        le = getattr(self, "label_encoder", None)
        if le is None and getattr(self, "xgb_label_encoder", None) is not None:
            le = self.xgb_label_encoder
        if le is None:
            le = LabelEncoder()
            le.fit(y)
        self.label_encoder = le  # keep canonical
        y_encoded = le.transform(y).astype(int)
        '''
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        self.label_encoder = le
        '''

        # Attempt to load histogram images
        hist_feats = None
        if use_histograms:
            try:
                imgs, labels_imgs, auxs, le_imgs = DDPM_utils._load_th2_images(
                    inputfiles=self._inputfiles, histogram_name=self.histogram_name,
                    tree_name=self.tree_name, aux_scalar_branch=self.aux_scalar_branch,
                    target=self.target, hist_shape=self.hist_shape,
                    x_edges=self.x_edges, y_edges=self.y_edges
                )
                # encode images
                feats_img = self.hist_to_features(imgs)
                # If image label encoder differs, merge
                if hasattr(self, "label_encoder") and le is not None and le_imgs is not None:
                    # try to map image labels into canonical encoder if possible
                    # if label sets are equal-sized and ordering matches, we can align; otherwise skip direct merge
                    pass
                # If number of hist images equals number of rows in df -> combine per-index
                if feats_img.shape[0] == df.shape[0]:
                    hist_feats = feats_img
                else:
                    # If number of hist images == number of input files, try to expand
                    if feats_img.shape[0] == len(self._inputfiles):
                        # attempt to map each row in df to its source file (we do not currently store file index)
                        # fallback: don't combine automatically (user should call with aligned datasets)
                        print("[xgb_train] Warning: hist images count equals number of files but not rows; skipping automatic join.")
                    else:
                        print("[xgb_train] Warning: histogram images found but count doesn't match rows; training on tree-only.")
            except Exception as e:
                print("[xgb_train] Could not load histograms (will train tree-only):", e)

        # Build final X
        if hist_feats is not None:
            X = np.concatenate([X_tree.to_numpy(), hist_feats], axis=1)
            feature_names = list(X_tree.columns) + getattr(self, "hist_feature_names", [])
        else:
            X = X_tree.to_numpy()
            feature_names = list(X_tree.columns)

        #Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state
        )
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        num_classes = len(le.classes_)

        #XGBoost params:
        xgb_params={
            'objective': 'multi:softmax',
            'eval_metric': loss,
            'num_class': num_classes,
            'max_depth': max_depth,
            'eta': eta,
            'subsample': subsample,
            'tree_method': tree_method
        }

        print("Training XGBoost model...")
        self.xgb_model = xgb.train(
            xgb_params, 
            dtrain, 
            num_boost_round=num_boost_round,
            evals=[(dtrain,'train'),(dtest,'test')],
            verbose_eval=True
        )

        #Evaluate
        preds = self.xgb_model.predict(dtest)
        acc = accuracy_score(y_test, preds)
        print(f"Test Accuracy: {acc: 4f}")
        # store a reference encoder for ddpm usage
        self.xgb_label_encoder = le

        return self.xgb_model

    def gbhm_train(
            self,n_estimators: int=100, learn_rate:float=0.1,
            max_depth: int=3,test_size: float=0.3,
            random_state: int=42,warm_start: bool=True,
            save: bool=True, use_histograms: bool = True,
        ):
        """
        Trains the Gradient Boosting model.

        Args:
            test_size (float): The proportion of the dataset to include in the test split.
            random_state (int): Controls the shuffling applied to the data.
            * Set Hyperparameters
             - n_estimators
             - learn_rate
             - max_depth
        """
        #Load and preprocess data
        data=self._load_data()
        X_tree=data[self.features]
        y_raw=data[self.target]

        '''
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
        else:
            # Update encoder with new labels
            new_classes = np.unique(y)
            all_classes = np.unique(np.concatenate([self.label_encoder.classes_, new_classes]))
            self.label_encoder.classes_ = all_classes
            y_encoded = self.label_encoder.transform(y)
        '''
        # Fit or reuse label encoder
        le = getattr(self, "label_encoder", None)
        if le is None:
            le = LabelEncoder()
            le.fit(y_raw)
            print("[gbhm_train] Built new LabelEncoder from data.")
        else:
            # If the existing encoder does not contain all labels, refit to union
            unique_new = np.unique(y_raw)
            if not set(unique_new).issubset(set(le.classes_)):
                print("[gbhm_train] Existing label encoder missing new labels -> refitting encoder to union.")
                union = list(sorted(set(le.classes_).union(set(unique_new)),
                                    key=lambda x: str(x)))
                le = LabelEncoder()
                le.fit(union)
        self.gbhm_label_encoder = le
        self.label_encoder = le
        y_encoded = le.transform(y_raw).astype(int)

        hist_feats = None
        if use_histograms:
            try:
                imgs, labels_imgs, auxs, le_imgs = DDPM_utils._load_th2_images(
                    inputfiles=self._inputfiles, histogram_name=self.histogram_name,
                    tree_name=self.tree_name, aux_scalar_branch=self.aux_scalar_branch,
                    target=self.target, hist_shape=self.hist_shape,
                    x_edges=self.x_edges, y_edges=self.y_edges
                )
                feats_img = self.hist_to_features(imgs)
                if feats_img.shape[0] == X_tree.shape[0]:
                    hist_feats = feats_img
                else:
                    print("[gbhm_train] hist image count != rows; skipping hist join.")
            except Exception as e:
                print("[gbhm_train] could not load histograms (training on tree-only):", e)

        if hist_feats is not None:
            X = np.concatenate([X_tree.to_numpy(), hist_feats], axis=1)
            self.gbhm_feature_names = list(X_tree.columns) + getattr(self, "hist_feature_names", [])
        else:
            X = X_tree.to_numpy()
            self.gbhm_feature_names = list(X_tree.columns)

        #Split data into training and testing sets
        X_train, X_test, y_train, y_test=train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state)

        #Initialize and train the Gradient Boosting model
        # If there is an existing model and warm_start True, continue training
        if self.gbhm_clf is not None and warm_start:
            print("[GBHM] Continuing training from existing GBHM (warm_start=True)...")
            # set n_estimators > current n_estimators and call fit again
            current_n = getattr(clf, "n_estimators", 0)
            new_n = max(current_n, n_estimators)
            self.gbhm_clf.set_params(warm_start=True, n_estimators=new_n, learning_rate=learn_rate)
            # Fit will continue from previous trees if warm_start True
            self.gbhm_clf.fit(X_train, y_train)
        else:
            print("[GBHM] Training new GBHM model...")
            self.gbhm_clf = GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=learn_rate,
                max_depth=max_depth,
                random_state=random_state,
                warm_start=warm_start,
            )
            self.gbhm_clf.fit(X_train, y_train)

        #Evaluate the model
        train_accuracy=accuracy_score(y_train, self.gbhm_clf.predict(X_train))
        test_accuracy=accuracy_score(y_test, self.gbhm_clf.predict(X_test))
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")

        return self.gbhm_clf

    #################################################################
    #                                                               #
    #   Denoising Diffusion Probalistic Model (DDPM Tensorflow)     #
    #                                                               #
    #################################################################
    def build_diffusion_model(self, n_classes: Optional[int] = None):
        '''
        Build conditinal UNet model for DDPM
        '''
        if n_classes is None and self.label_encoder is not None:
            n_classes = len(self.label_encoder.classes_)
        elif n_classes in None:
            # default (will be re-fit when training)
            n_classes = 32

        input_shape = (*self.hist_shape, 1)
        #Create model where class embedding will be handled via class id input
        self.diffusion_model = HybridUNetDiT_TF(input_shape=input_shape, base_channels=32)
        #compile will be handled in train
        # create EMA model copy
        self.ema_model = self.diffusion_model.init_ema()
        self.ema_model.set_weights(self.diffusion_model.get_weights())
        return self.diffusion_model

    def q_sample(self, x_start: tf.Tensor, t: tf.Tensor,
                 noise: tf.Tensor, alphas_cumprod_tf: tf.Tensor):
        '''
        Forward q(x_t | x_0)
        x_start: (B,H,W,C) float32 in [-1,1]
        t: (B,) int32
        noise: same shape as x_start
        alphas_cumprod_tf: tf.constant vector length T (float32)
        '''
        coef1 = DDPM_utils.extract(tf.sqrt(alphas_cumprod_tf), t, tf.shape(x_start))
        coef2 = DDPM_utils.extract(tf.sqrt(1.0 - alphas_cumprod_tf), t, tf.shape(x_start))
        return coef1 * x_start + coef2 * noise

    def train_ddpm(self, epochs=10, batch_size=8, lr=2e-4, beta_schedule='cosine',
                   timesteps=None, use_mixed_precision=False, clip_grad_norm=1.0,
                   base_ch=32, embed_dim=128, num_heads=4, num_layers=2, groups_gn=8,
                   time_emb=True, class_emb=True, num_classes=4, self_condition=False, ema_decay=0.999,
                   device=None, path='plots/ddpm_training', show=True):
        '''
        Train DDPM (predict noise) on the set of images + conditioning label + aux scaler
        - Normalizes images to [-1,1].
        - Uses MSE on noise prediction.
        - Gradient clipping and EMA applied (EMA updated per-epoch outside @tf.function).
        '''
        # Optional mixed precision
        if use_mixed_precision:
            policy = tf.keras.mixed_precision.Policy("mixed_float16")
            tf.keras.mixed_precision.set_global_policy(policy)
            print("[INFO] Mixed precision ON")

        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow not available for DDPM.")
        device = device or ("/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0")

        #------------------- Timesteps and schedule -----------------------------
        T = int(timesteps or getattr(self, "ddpm_timesteps", 1000))
        self.ddpm_timesteps = T
        betas = DDPM_utils.make_beta_schedule(T, schedule=beta_schedule).astype(np.float32)
        alphas = (1. - betas).astype(np.float32)
        alphas_cumprod = np.cumprod(alphas, axis=0).astype(np.float32)
        self.ddpm_betas = betas
        # store back on object
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        # store schedule on self
        self.ddpm_timesteps = T
        self.ddpm_betas = betas

        #imgs (N,H,W)
        imgs, labels_raw, auxs, le = DDPM_utils._load_th2_images(
            inputfiles=self._inputfiles, histogram_name=self.histogram_name,
            tree_name=self.tree_name, aux_scalar_branch = self.aux_scalar_branch,
            target = self.target,hist_shape=self.hist_shape,
            x_edges = self.x_edges, y_edges = self.y_edges
        )

        # --- Label encoder handling: reuse if available from xgb or gbhm, else create ---
        le_exist = None
        if getattr(self, "label_encoder", None) is not None:
            le_exist = self.label_encoder
            print("[DDPM] Using existing label_encoder on self.label_encoder")
        elif getattr(self, "xgb_label_encoder", None) is not None:
            le_exist = self.xgb_label_encoder
            self.label_encoder = le_exist
            print("[DDPM] Using xgb_label_encoder")

        if le_exist is None:
            le = LabelEncoder()
            le.fit(labels_raw)
            print("[DDPM] Built new label_encoder from DDPM labels")
        else:
            # extend existing encoder with unseen labels
            known = list(le_exist.classes_)
            incoming = list(np.unique(labels_raw))
            merged = np.unique(known + incoming)
            if not np.array_equal(merged, known):
                print(f"[DDPM] Extending label_encoder: {known} -> {merged}")
                le = LabelEncoder()
                le.fit(merged)
            else:
                le = le_exist

        self.label_encoder = le
        labels_enc = le.transform(labels_raw).astype(np.int32)
        num_classes = len(le.classes_)
        print(f"[DDPM] Using {num_classes} classes for conditioning")

        self.plotter = PerformancePlotter(save_dir=path)
        self.label_encoder = le
        N = imgs.shape[0]

        #Prepare tensors
        X = imgs[..., np.newaxis].astype(np.float32)
        # Normalize to [-1, 1] for better training
        X = (X - 0.5) * 2.0
        y = labels_enc.astype(np.int32)
        aux = auxs.astype(np.float32)
        print(f"[DDPM] Loaded {N} images, hist_shape={self.hist_shape}")

        #Dataset
        ds = tf.data.Dataset.from_tensor_slices((X,y,aux))
        ds = ds.shuffle(2048).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        #Build model if needed
        if self.diffusion_model is None or not self.diffusion_compiled:
            # if no model, build base model
            self.diffusion_model = HybridUNetDiT_TF(
                input_shape=(*self.hist_shape, 1),
                base_ch=base_ch,
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                time_emb=time_emb,
                class_emb=class_emb,
                aux_emb_dim=32,
                ema_decay=ema_decay,
                num_classes=len(np.unique(y)) if len(y)>0 else num_classes,
                self_condition=self_condition)
            print("[DDPM] Built new diffusion model")
            #Create EMA helper
            self.ema_helper = EMA(self.diffusion_model, decay=ema_decay)

        # Ensure the diffusion model knows about the schedule
        try:
            self.diffusion_model.betas = betas
            self.diffusion_model.alphas = alphas
            self.diffusion_model.alphas_cumprod = alphas_cumprod
        except Exception:
            pass

        optimizer = keras.optimizers.Adam(learning_rate=lr)
        if use_mixed_precision:
            # Wrap optimizer for mixed precision
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        mse = tf.keras.losses.MeanSquaredError()

        # Prepare TF constants
        alphas_cumprod_tf = tf.constant(alphas_cumprod, dtype=tf.float32)
        betas_tf = tf.constant(betas, dtype=tf.float32)
 
        # init EMA model (clone architecture, then copy weights)
        ema_model = None
        if ema_decay:
            # clone architecture (functional clone)
            try:
                ema_model = tf.keras.models.clone_model(self.diffusion_model)
                # build by calling once with dummy inputs so weights exist
                dummy_x = tf.zeros((1, *self.hist_shape, 1), dtype=tf.float32)
                dummy_t = tf.zeros((1,), dtype=tf.int32)
                dummy_y = tf.zeros((1,), dtype=tf.int32)
                dummy_aux = tf.zeros((1,1), dtype=tf.float32)
                _ = ema_model(dummy_x, dummy_t, dummy_y, dummy_aux, training=False)
                ema_model.set_weights(self.diffusion_model.get_weights())
                print("[DDPM] EMA model initialized")
            except Exception as e:
                print("[DDPM] EMA init failed:", e)
                ema_model = None

        train_losses = []
        # training loop (eager - no @tf.function to keep EMA simple)
        for epoch in range(epochs):
            t0 = time.time()
            epoch_loss = 0.0
            steps = 0
            for batch_x, batch_y, batch_aux in ds:
                batch_x = tf.cast(batch_x, tf.float32)
                batch_aux = tf.cast(batch_aux, tf.float32)
                batch_y = tf.cast(batch_y, tf.int32)
 
                # sample timesteps uniformly per sample
                B = tf.shape(batch_x)[0]
                t_rand = tf.random.uniform((B,), minval=0, maxval=T, dtype=tf.int32)
 
                noise = tf.random.normal(tf.shape(batch_x), dtype=tf.float32)
                x_noisy = self.q_sample(batch_x, t_rand, noise, alphas_cumprod_tf)
 
                with tf.GradientTape() as tape:
                    pred_noise = self.diffusion_model(x_noisy, t_rand, batch_y, batch_aux, training=True)
                    pred_noise = tf.cast(pred_noise, tf.float32)
                    loss = mse(noise, pred_noise)
                    # scaling by simple factor keeps numerics stable; usually not needed
                grads = tape.gradient(loss, self.diffusion_model.trainable_variables)
                # gradient clipping
                if clip_grad_norm is not None:
                    grads, _ = tf.clip_by_global_norm(grads, clip_grad_norm)
                optimizer.apply_gradients(zip(grads, self.diffusion_model.trainable_variables))
 
                epoch_loss += float(loss)
                steps += 1
 
            epoch_loss = epoch_loss / max(1, steps)
            train_losses.append(epoch_loss)
 
            # EMA weight update (eager)
            if ema_model is not None:
                new_weights = []
                cur_w = self.diffusion_model.get_weights()
                ema_w = ema_model.get_weights()
                for w_model, w_ema in zip(cur_w, ema_w):
                    new_w = ema_decay * w_ema + (1.0 - ema_decay) * w_model
                    new_weights.append(new_w)
                ema_model.set_weights(new_weights)
 
            print(f"[DDPM] Epoch {epoch+1}/{epochs} loss={epoch_loss:.6f} time={time.time()-t0:.1f}s")

        self.diffusion_compiled = True
        self.ema_model = ema_model
        # store schedule on diffusion model again
        try:
            self.diffusion_model.betas = betas
            self.diffusion_model.alphas = alphas
            self.diffusion_model.alphas_cumprod = alphas_cumprod
        except Exception:
            pass

        # keep training stats
        self.ddpm_train_history = {"loss": train_losses, "betas": betas}
        print('DDPM training is complete...')
        if show:
            DDPM_utils.plot_loss_curve(train_losses=train_losses,save_dir=path,show=show)
            #Plot beta schedule
            DDPM_utils.plot_beta_schedule(betas=betas,save_dir=path)

    def sample_ddpm(
            self,
            n_samples: int=4,
            class_ids: Optional[List[int]]=None, 
            aux_scalars: Optional[np.ndarray]=None,
            device=None
        ):
        '''
        Sample images from the DDPM reverse process conditioned on class_ids and aux_scalars.
        class_ids: list of class integers in original pdgID domain (will be label-encoded internally)
        aux_scalars: shape (n_samples,1) float array
        '''
        if self.diffusion_model is None or not self.diffusion_compiled:
            raise RuntimeError('No trained diffusion model available. Run train_ddpm first or load one...')

        device = device or ("/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0")
        T = self.ddpm_timesteps
        betas = tf.cast(self.betas, tf.float32)
        alphas = tf.cast(self.alphas, tf.float32)
        alphas_cumprod = tf.cast(self.alphas_cumprod, tf.float32)
        alphas_cumprod_prev = tf.cast(self.alphas_cumprod_prev, tf.float32)

        #Prepare conditioning
        if class_ids is None:
            class_ids = [0]*n_samples
        if aux_scalars is None:
            aux_scalars = np.zeros((n_samples,1),dtype=np.float32)
        else:
            #ensure correct shape
            aux_scalars = np.array(aux_scalars, dtype=np.float32)
            if aux_scalars.ndim == 1:
                aux_scalars = aux_scalars[:, None]

        if aux_scalars is None or np.any(np.isnan(aux_scalars)):
            print("aux_scalars None or invalid — defaulting to zeros")
            aux_scalars = np.zeros((n_samples, 1), dtype=np.float32)
        #If original labels were encoded, map provided class IDs (if they are raw pdgIDs)
        #If user passed pdgID values, convert via label_encoder
        cls_ids_arr = np.array(class_ids, dtype=np.float32)
        if self.label_encoder is not None:
            #if class_ids appear to be raw pdg (not encoded), attempt inverse map
            #try mapping provided values if they exist in encoder classes
            if np.any([val in self.label_encoder.classes_ for val in class_ids]):
                cls_ids_arr = self.label_encoder.transform(class_ids)
        cls_ids_arr = cls_ids_arr.astype(np.int32)

        #Start from pure noise
        shape = (n_samples, *self.hist_shape, 1)
        x_t = tf.random.normal(shape, dtype=tf.float32)

        for t_idx in reversed(range(T)):
            t_steps = tf.fill((n_samples), tf.cast(t_idx,tf.int32))
            #predict noise
            cls_ids_arr = tf.convert_to_tensor(cls_ids_arr, dtype=tf.int32)
            aux_scalars = tf.convert_to_tensor(aux_scalars, dtype=tf.float32)
            pred_noise = self.diffusion_model(x_t, t_steps, cls_ids_arr, aux_scalars,training=False)
            #Compute posterior mean and variance
            beta_t = betas[t_idx]
            sqrt_one_minus_alphas_cumprod_t = math.sqrt(1.0-alphas_cumprod[t_idx])
            sqrt_recip_alphas_t = 1.0/math.sqrt(alphas[t_idx])

            #estimate x0_pred
            x0_pred = (x_t-sqrt_one_minus_alphas_cumprod_t*pred_noise)*sqrt_recip_alphas_t

            #Clip x0_pred to [-1,1]
            x0_pred = tf.clip_by_value(x0_pred,-1.0,1.0)

            if t_idx > 0:
                #Compute posterior mean
                coef1 = (betas[t_idx]*np.sqrt(alphas_cumprod_prev[t_idx]))/(1.0-alphas_cumprod[t_idx])
                coef2 = ((1.0-alphas_cumprod_prev[t_idx])*np.sqrt(alphas[t_idx]))/(1.0-alphas_cumprod[t_idx])
                mean = coef1*x0_pred + coef2*x_t
                #Sample noise
                noise = tf.random.normal(shape, dtype=tf.float32)
                var = betas[t_idx]*(1.0 - alphas_cumprod_prev[t_idx])/(1.0 - alphas_cumprod[t_idx])
                x_t = mean + tf.sqrt(var)*noise
            else:
                x_t = x0_pred

        #(n_samples,H,W)
        out = x_t.numpy().squeeze(-1)
        #rescale back to [0,1] if necessary - model on [0,1]
        out = (out + 1.0) / 2.0
        out = np.clip(out, 0.0, 1.0)
        return out

    def save_ddpm(self, path='saved_models'):
        os.makedirs(path, exist_ok=True)
        if self.diffusion_model is None:
            raise RuntimeError('No diffusion model to save...')
        #Save weights + important arrays
        self.diffusion_model.save_weights(os.path.join(path, 'ddpm.weights.h5'))
        np.save(os.path.join(path,'ddpm_betas.npy'),self.betas)
        np.save(os.path.join(path,'ddpm_alphas_cumprod.npy'),self.alphas_cumprod)
        #Save label encoder
        if self.label_encoder is not None:
            joblib.dump(self.label_encoder,os.path.join(path,'label_encoder.pkl'))
        print(f'Saved DDPM checkpoint to {path}')

    def load_ddpm(self, path='saved_models'):
        #load arrays
        self.betas = np.load(os.path.join(path,'ddpm_betas.npy'))
        self.alphas_cumprod = np.load(os.path.join(path,'ddpm_alphas_cumprod.npy'))
        #rebuild model with hist_shape and load weights
        self.build_diffusion_model(n_classes=len(self.label_encoder.classes_) if self.label_encoder is not None else None)
        self.diffusion_model.load_weights(os.path.join(path,'ddpm.weights.h5'))
        #load label encoder if present
        le_path = os.path.join(path,'label_encoder.pkl')
        if os.path.exists(le_path):
            self.label_encoder = joblib.load(le_path)
        self.diffusion_compiled = True
        print(f'Loaded DDPM checkpoint from {path}')

    ###########################################################
    # Prediction helpers with robust reshaping / sanitization
    ###########################################################
    def _build_full_feature_array(self, hist_array: np.ndarray, target_feature_names: Optional[List[str]] = None, target_nfeat: Optional[int] = None):
        """
        Build a full feature array (N, target_nfeat) expected by classifier.
        - hist_array: (N, Hf) produced by encode_histograms()
        - target_feature_names: ordered list of feature names used by classifier (may include tree features and hist names)
        - target_nfeat: expected number of features (int)
        Strategy:
        - If target_feature_names available and self.hist_feature_names available, place hist columns at the matching names.
        - For tree features that classifier expected but we don't have (generated images), fill zeros.
        - If no mapping possible, and sizes match, return hist_array; else pad/truncate to target_nfeat.
        """
        N = hist_array.shape[0]
        # sanitize NaNs/Infs
        hist_array = np.nan_to_num(hist_array, nan=0.0, posinf=0.0, neginf=0.0)

        if target_feature_names is not None and self.hist_feature_names is not None:
            # Build columns: if target_feature_names contains hist_feature_names, map them.
            full = np.zeros((N, target_nfeat if target_nfeat is not None else len(target_feature_names)), dtype=np.float32)
            # try to detect hist positions
            for i, hname in enumerate(self.hist_feature_names):
                if hname in target_feature_names:
                    idx = target_feature_names.index(hname)
                    if idx < full.shape[1]:
                        full[:, idx] = hist_array[:, i]
            # any remaining hist columns that didn't map: append/truncate at the end if room
            mapped = [fn for fn in self.hist_feature_names if fn in target_feature_names]
            if len(mapped) < hist_array.shape[1]:
                # fill trailing columns where empty
                remaining = [i for i in range(full.shape[1]) if np.all(full[:, i] == 0.0)]
                j = 0
                for i_hist in range(hist_array.shape[1]):
                    hname = self.hist_feature_names[i_hist]
                    if hname in mapped:
                        continue
                    if j >= len(remaining):
                        break
                    full[:, remaining[j]] = hist_array[:, i_hist]
                    j += 1
            # if target_nfeat is None, return full trimmed to len(target_feature_names)
            return full
        else:
            # no mapping info: align by sizes
            if target_nfeat is None:
                return hist_array
            else:
                if hist_array.shape[1] == target_nfeat:
                    return hist_array
                elif hist_array.shape[1] > target_nfeat:
                    return hist_array[:, :target_nfeat]
                else:
                    pad = np.zeros((N, target_nfeat - hist_array.shape[1]), dtype=hist_array.dtype)
                    return np.concatenate([hist_array, pad], axis=1)

    def xgb_predict(self, new_data):
        """
        Makes predictions on new data.

        Args:
            new_data (pd.DataFrame): DataFrame containing the same features as the training data.

        Returns:
            np.ndarray: The predicted class labels.
        """
        if self.xgb_model is None:
            raise RuntimeError("No xgb_model loaded.")
        # Accept DataFrame or ndarray
        X = new_data
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.asarray(X)
        # sanitize
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # If xgb model was trained with a different feature count, fix shape by pad/truncate
        expected = getattr(self, 'xgb_num_features', None)
        if expected is not None:
            if X.shape[1] != expected:
                if X.shape[1] > expected:
                    X = X[:, :expected]
                else:
                    pad = np.zeros((X.shape[0], expected - X.shape[1]), dtype=X.dtype)
                    X = np.concatenate([X, pad], axis=1)

        dmat = xgb.DMatrix(X)
        preds = self.xgb_model.predict(dmat).astype(int)
        if getattr(self, "xgb_label_encoder", None) is not None:
            return self.xgb_label_encoder.inverse_transform(preds)
        if getattr(self, "label_encoder", None) is not None:
            return self.label_encoder.inverse_transform(preds)
        return preds

    def gbhm_predict(self, X: np.ndarray):
        if self.gbhm_clf is None:
            raise RuntimeError('No gbhm_clf loaded.')
        # Accept DataFrame or ndarray
        if isinstance(X, pd.DataFrame):
            Xnp = X.values
        else:
            Xnp = np.asarray(X)
        # sanitize
        Xnp = np.nan_to_num(Xnp, nan=0.0, posinf=0.0, neginf=0.0)

        # fix shape to expected
        expected = getattr(self, 'gbhm_num_features', None)
        if expected is not None and Xnp.shape[1] != expected:
            if Xnp.shape[1] > expected:
                Xnp = Xnp[:, :expected]
            else:
                pad = np.zeros((Xnp.shape[0], expected - Xnp.shape[1]), dtype=Xnp.dtype)
                Xnp = np.concatenate([Xnp, pad], axis=1)

        preds = self.gbhm_clf.predict(Xnp).astype(int)
        if self.gbhm_label_encoder is not None:
            return self.gbhm_label_encoder.inverse_transform(preds)
        if self.label_encoder is not None:
            return self.label_encoder.inverse_transform(preds)
        return preds

    def predict_from_hist(self, hist2d):
        '''
        Predict class labels given a ROOT TH2 hist (PyROOT/ROOT TH2).
        '''
        if self.xgb_model is None:
            raise RuntimeError("No trained model found. Run xgb_train() first or load saved model...")

        x_bins = hist2d.GetXaxis().GetNbins()
        y_bins = hist2d.GetYaxis().GetNbins()
        data = []

        for ix in range(1, x_bins + 1):
            x_center = hist2d.GetXaxis().GetBinCenter(ix)
            for iy in range(1, y_bins + 1):
                count = hist2d.GetBinContent(ix, iy)
                if count <= 0:
                    continue
                y_center = hist2d.GetYaxis().GetBinCenter(iy)
                data.append((x_center, y_center, count))

        if not data:
            print("*****Warning: Histogram is empty******")
            return None

        df = pd.DataFrame(data, columns=['posX', 'posY', 'weight'])
        feature_cols = [f for f in ['posX', 'posY'] if f in self.features]
        X = df[feature_cols]

        dmatrix = xgb.DMatrix(X, weight=df['weight'].to_numpy())
        preds = self.xgb_model.predict(dmatrix).astype(int)
        labels = self.label_encoder.inverse_transform(preds)

        df['prediction'] = labels
        return df

    def generate_and_classify(self, n_samples=100, aux_val=None, class_id=None):
        """
        High-level method: generate synthetic histograms via DDPM (if available),
        classify them using GBHM/XGB and return a small summary.
        If DDPM not present -> raises.
        """
        if self.diffusion_model is None:
            raise RuntimeError("No diffusion model available for generation.")
        if not hasattr(self.diffusion_model, "sample"):
            raise RuntimeError("Your diffusion model must implement .sample(...) for generation.")

        y_cond = None
        if class_id is not None:
            y_cond = np.array([class_id] * n_samples, dtype=np.int32)
        aux = None
        if aux_val is not None:
            aux = np.array([[aux_val]] * n_samples, dtype=np.float32)

        imgs = self.diffusion_model.sample(n_samples, class_id=y_cond, aux_val=aux)
        if imgs.ndim == 3:
            imgs_proc = imgs  # (N,H,W)
        else:
            imgs_proc = imgs[..., 0] if imgs.shape[-1] == 1 else np.squeeze(imgs)

        # Use encoder-based embeddings when available
        try:
            feature_array = self.encode_histograms(imgs_proc)
        except Exception as e:
            print("[generate_and_classify] encode_histograms failed, falling back to hist_to_features:", e)
            feature_array = self.hist_to_features(imgs_proc)

        # sanitize NaN/Inf
        feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=0.0, neginf=0.0)

        results = {}
        # gbhm: build full feature array according to gbhm_feature_names/self.gbhm_num_features
        if self.gbhm_clf is not None:
            try:
                target_names = getattr(self, 'gbhm_feature_names', None)
                target_n = getattr(self, 'gbhm_num_features', None)
                full_X = self._build_full_feature_array(feature_array, target_feature_names=target_names, target_nfeat=target_n)
                # ensure numeric numpy array
                results['gbhm'] = self.gbhm_predict(full_X)
            except Exception as e:
                print("[generate_and_classify] gbhm_predict failed:", e)
                results['gbhm'] = None

        # xgb: build full feature array according to xgb_feature_names/self.xgb_num_features
        if self.xgb_model is not None:
            try:
                target_names = getattr(self, 'xgb_feature_names', None)
                target_n = getattr(self, 'xgb_num_features', None)
                full_X = self._build_full_feature_array(feature_array, target_feature_names=target_names, target_nfeat=target_n)
                results['xgb'] = self.xgb_predict(full_X)
            except Exception as e:
                print("[generate_and_classify] xgb_predict failed:", e)
                results['xgb'] = None

        return results


    def get_xgb_feature_importances(self):
        """Returns the feature importances from the trained model."""
        if self.xgb_model is None:
            raise RuntimeError("Model has not been trained yet.")

        importances=pd.Series(self.xgb_model.feature_importances_,index=self.features).sort_values(ascending=False)
        return importances

    def save_xgb_model(self, path_prefix='saved_models/xgb_model'):
        '''
        Save xgb model and encoder:
        - {path_prefix}.json: XGBoost model
        - {path_prefix}_encoder.pkl: LabelEncoder
        '''

        if self.xgb_model is None:
            raise RuntimeError('No model tained yet. Must run xgb_train() first...')

        model_path = f"{path_prefix}xgb_model.json"
        self.xgb_model.save_model(model_path)

        #Save label encoder
        encoder_path = f"{path_prefix}_encoder.pkl"
        with open(encoder_path, "wb") as f:
            pickle.dump(self.label_encoder, f)

        print(f"Saved model to {model_path}")
        print(f"Saved label encoder to {encoder_path}")

    def load_xgb_model(self,path_prefix='saved_models/xgb_model'):
       model_path = os.path.join(path_prefix, "xgb_model.json")
       encoder_path = os.path.join(path_prefix, "_encoder.pkl")
       if not os.path.exists(model_path):
           raise FileNotFoundError("No saved model found to load.")
       self.xgb_model = xgb.Booster()
       self.xgb_model.load_model(model_path)
       if os.path.exists(encoder_path):
           self.label_encoder = joblib.load(encoder_path)
       print(f"Loaded model and encoder from {self.model_dir}")

    def predict_from_hist(self, hist2d):
        '''
        Predict class lebels given a ROOT TH2 hist
        '''
        if self.xgb_model is None:
            raise RuntimeError("No trained model found. Run xgb_train() first or load saved model...")

        #Extract bin centers and counts
        x_bins = hist2d.GetXaxis().GetNbins()
        y_bins = hist2d.GetYaxis().GetNbins()
        data = []

        for ix in range(1,x_bins+1):
            for iy in range(1,y_bins+1):
                count = hist2d.GetBinContent(ix, iy)
                if count <= 0:
                    continue #skip empty bins
                x_center = hist2d.GetXaxis().GetBinContent(ix)
                y_center = hist2d.GetYaxis().GetBinContent(iy)
                data.append((x_center, y_center, count))

        if not data:
            print("*****Warning: Histogram is empty******")
            return None

        #convert to DataFrame
        df = pd.DataFrame(data, columns=['posX','posY','weight'])

        feature_cols = [f for f in ['posX','posY'] if f in self.features]
        X = df[feature_cols]

        dmatrix = xgb.DMatrix(X, weight=df['weight'].to_numpy())

        #Predict
        preds = self.xgb_model.predict(dmatrix)
        labels = self.label_encoder.inverse_transform(preds.astype(int))

        #Attach predictions
        df['prediction'] = labels

        return df
    
    ###############################################################
    #######   Fine-tuning existing models from saved state  #######
    ###############################################################
    def xgb_finetune(self, new_inputfiles, num_boost_round=100,
            lr_tune=0.05, random_state=42,path_prefix='saved_models/xgb_model'):
        if self.xgb_model is None:
            print('No model in memory. Attempting to load existing model...')

        df = self._load_data_from_files(new_inputfiles)
        X_new = df[self.features + ['bin_index', 'radius']]
        y_new = df[self.target]

        #Use existing label encoder
        if self.label_encoder is None:
            le = LabelEncoder()
            y_encoded = le.fit_transform(y_new)
            self.label_encoder = le
        else:
            le = self.label_encoder
            unique_new_labels = np.setdiff1d(np.unique(y_new), le.classes_)
            if len(unique_new_labels) > 0:
                print(f"New particle IDs found: {unique_new_labels}. Updating label encoder.")
                le.classes_ = np.concatenate([le.classes_, unique_new_labels])
            y_encoded = le.transform(y_new)
        
        num_classes = len(le.classes_)
        
        dtrain = xgb.DMatrix(X_new, label=y_encoded)
        
        # --- Load config safely ---
        try:
            config = json.loads(self.xgb_model.save_config())
            learner_cfg = config.get("learner", {})
            gb_cfg = learner_cfg.get("gradient_booster", {})
        
            # Handle both new & old structures
            train_params = {}
            if isinstance(gb_cfg.get("updater"), list) and gb_cfg["updater"]:
                # Old format
                train_params = gb_cfg["updater"][0].get("train_param", {})
            elif "train_param" in gb_cfg:
                # Newer versions
                train_params = gb_cfg["train_param"]
        
            # Fallback defaults
            max_depth = int(train_params.get("max_depth", 6))
            subsample = float(train_params.get("subsample", 0.8))
        except Exception as e:
            print(f"⚠️ Warning: Could not parse config, using defaults. ({e})")
            max_depth, subsample = 6, 0.8
        
        params = {
            "objective": "multi:softmax",
            "num_class": num_classes,
            "max_depth": max_depth,
            "eta": lr_tune,
            "subsample": subsample,
            "tree_method": "hist",
            "eval_metric": "mlogloss",
        }
        
        print("Fine-tuning existing model...")
        self.xgb_model = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            xgb_model=self.xgb_model
        )
        
        #os.makedirs(path_prefix, exist_ok=True)
        #model_path = os.path.join(path_prefix, f"xgb_finetuned.json")
        #self.xgb_model.save_model(model_path)
        print(f"Fine-tuned model updated in memory...")
        
        return self.xgb_model

    def gbhm_finetune(self, new_inputfiles, n_estimators_add=50):
        """Fine-tune existing Gradient Boosting model with additional data"""
        print("Fine-tuning GBHM model...")
        data = self._load_data_from_files(new_inputfiles)
        X = data[self.features + ["bin_index", "radius"]]
        y = data[self.target]

        # Update label encoder
        new_classes = np.unique(y)
        all_classes = np.unique(np.concatenate([self.label_encoder.classes_, new_classes]))
        self.label_encoder.classes_ = all_classes
        y_encoded = self.label_encoder.transform(y)

        # Continue training
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3)
        self.gbhm_clf.n_estimators += n_estimators_add
        self.gbhm_clf.fit(X_train, y_train)

        print("Fine-tuning complete.")
        test_acc = accuracy_score(y_test, self.gbhm_clf.predict(X_test))
        print(f"New Test Accuracy: {test_acc:.4f}")

    ###############################################################
    ## ddpm finetune
    ###############################################################
    def ddpm_finetune(self, new_inputfiles: Union[str, List[str]], epochs: int = 3,
                      batch_size: int = 8, lr: float = 2e-4):
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow not available for DDPM.")

        if self.diffusion_model is None:
            # if no model, build base model
            self.diffusion_model = DDPM_utils.make_unet2d(input_shape=(*self.hist_shape, 1))
        # temporarily swap files, load images, then restore
        old_files = self._inputfiles
        self._inputfiles = new_inputfiles if isinstance(new_inputfiles, list) else [new_inputfiles]
        imgs, labels_enc, auxs, le = DDPM_utils._load_th2_images(
            inputfiles=self._inputfiles, histogram_name=self.histogram_name,
            tree_name=self.tree_name, aux_scalar_branch = self.aux_scalar_branch,
            target = self.target,hist_shape=self.hist_shape,
            x_edges = self.x_edges, y_edges = self.y_edges
        )
        self._inputfiles = old_files

        # continue training using same train loop as train_ddpm but smaller epochs
        opt = keras.optimizers.Adam(learning_rate=lr)
        T = self.ddpm_timesteps
        alphas_cum = tf.constant(self.alphas_cumprod, dtype=tf.float32)
        X = imgs[..., np.newaxis].astype(np.float32)
        X = (X - 0.5) * 2.0  # Normalize to [-1, 1]
        y = labels_enc.astype(np.int32)
        aux = auxs.astype(np.float32)
        ds = tf.data.Dataset.from_tensor_slices((X, y, aux)).shuffle(1024).batch(batch_size).prefetch(2)

        @tf.function
        def train_step(x0, labels, aux_scalar):
            batch_size_local = tf.shape(x0)[0]
            t = tf.random.uniform((batch_size_local,), minval=0, maxval=T, dtype=tf.int32)
            noise = tf.random.normal(tf.shape(x0))
            coef1 = DDPM_utils.extract(tf.sqrt(alphas_cum), t, tf.shape(x0))
            coef2 = DDPM_utils.extract(tf.sqrt(1.0 - alphas_cum), t, tf.shape(x0))
            x_noisy = coef1 * x0 + coef2 * noise
            with tf.GradientTape() as tape:
                noise_pred = self.diffusion_model([x_noisy, t, labels, aux_scalar], training=True)
                loss = tf.reduce_mean(tf.square(noise - noise_pred))
            grads = tape.gradient(loss, self.diffusion_model.trainable_variables)
            opt.apply_gradients(zip(grads, self.diffusion_model.trainable_variables))
            return loss

        print(f"Fine-tuning DDPM on {X.shape[0]} images for {epochs} epochs...")
        for epoch in range(epochs):
            avg_loss = 0.0
            steps = 0
            t0 = time.time()
            for bx, by, bau in ds:
                loss_val = train_step(bx, by, bau)
                avg_loss += float(loss_val)
                steps += 1
            avg_loss /= max(1, steps)
            print(f"[DDPM fine-tune] epoch {epoch+1}/{epochs} loss={avg_loss:.6f} time={time.time()-t0:.1f}s")

        self.save_ddpm()
        print("DDPM fine-tune complete.")
        
    ###############################################################
    # -------- Joint performance plot & summary helper ---------
    ###############################################################
    def plot_and_report_joint_performance(self, path: str = "plots/joint_performance",
                                          save=True):
        """
        Gathers metrics available on this object (from _gbhm_metrics, _xgb_metrics, and ddpm training records)
        and calls PerformancePlotter.plot_joint_performance + generates a text summary.
        """
        DDPM_utils._ensure_dir(path)
        metrics = {}
        # GBHM metrics
        if self.gbhm_clf is not None:
            # If training data still available, compute simple accuracy on a held-out split
            try:
                df = self._load_data()
                X = df[self.features]
                y_raw = df[self.target].values
                le = self.gbhm_label_encoder or self.label_encoder
                if le is not None:
                    y_enc = le.transform(y_raw).astype(int)
                    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)
                    metrics['gbhm_accuracy'] = float(accuracy_score(y_test, self.gbhm_clf.predict(X_test)))
                else:
                    metrics['gbhm_accuracy'] = None
            except Exception as e:
                metrics['gbhm_accuracy'] = None
        else:
            metrics['gbhm_accuracy'] = None

        # XGB metrics
        if self.xgb_model is not None:
            try:
                df = self._load_data()
                X = df[self.features + ['bin_index','radius']]
                y_raw = df[self.target].values
                le = self.xgb_label_encoder or self.label_encoder
                if le is not None:
                    y_enc = le.transform(y_raw).astype(int)
                    _, X_test, _, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)
                    dtest = xgb.DMatrix(X_test)
                    preds = self.xgb_model.predict(dtest).astype(int)
                    metrics['xgb_accuracy'] = float(accuracy_score(y_test, preds))
                else:
                    metrics['xgb_accuracy'] = None
            except Exception:
                metrics['xgb_accuracy'] = None
        else:
            metrics['xgb_accuracy'] = None

        # DDPM metrics - attempt to get simple reconstruction MSE if plotter/metrics exist
        metrics['ddpm_recon_mse'] = None
        if self.plotter is not None and hasattr(self.plotter, "compute_ddpm_recon_mse"):
            try:
                metrics['ddpm_recon_mse'] = float(self.plotter.compute_ddpm_recon_mse(self))
            except Exception:
                metrics['ddpm_recon_mse'] = None

        # Call plotting helper if present
        if self.plotter is not None and hasattr(self.plotter, "plot_joint_performance"):
            try:
                self.plotter.plot_joint_performance(metrics.get('gbhm_accuracy'), metrics.get('xgb_accuracy'), metrics.get('ddpm_recon_mse'), save=save)
            except Exception as e:
                print("[plot_and_report_joint_performance] plotting failed:", e)

        # Summarize
        summary = {
            "gbhm_accuracy": metrics.get('gbhm_accuracy'),
            "xgb_accuracy": metrics.get('xgb_accuracy'),
            "ddpm_recon_mse": metrics.get('ddpm_recon_mse'),
            "label_encoder_classes": list(self.label_encoder.classes_) if self.label_encoder is not None else []
        }
        if save:
            DDPM_utils.save_pickle(summary,
                                   os.path.join(path, "combined_summary.pkl"))

        return summary

    # -------------------------
    # Utility: update canonical encoder after training
    # -------------------------
    def _update_label_encoder_from(self, le: LabelEncoder):
        """
        Make given encoder canonical across self (xgb + gbhm).
        """
        if le is None:
            return
        self.label_encoder = le
        self.xgb_label_encoder = le
        self.gbhm_label_encoder = le
        print("[_update_label_encoder_from] canonical label encoder updated.")

    ###############################################################
    # Helpers for fine-tune loading
    ###############################################################
    def _load_data_from_files(self, files):
        """Wrapper to reuse same data-loading logic for fine-tuning."""
        old_files = self._inputfiles
        self._inputfiles = files if isinstance(files, list) else [files]
        df = self._load_data()
        self._inputfiles = old_files
        return df

    ###############################################################
    #####      Save Both XGB & GBHM and Load  models       ########
    ###############################################################
    def save_models(self, path_prefix="saved_models/hybrid_model"):
        """Save model and encoder to disk."""
        base = os.path.splitext(path_prefix)[0]
        DDPM_utils._ensure_dir(os.path.dirname(base) or ".")
        if self.gbhm_clf is not None:
            joblib.dump(self.gbhm_clf, base + ".gbhm.pkl")
            if getattr(self, "gbhm_label_encoder", None) is not None:
                save_pickle(self.gbhm_label_encoder, base + ".gbhm.label_encoder.pkl")
        if self.xgb_model is not None:
            # save_model uses xgb Booster.save_model
            try:
                self.xgb_model.save_model(base + ".xgb.json")
            except Exception:
                # fallback to joblib
                joblib.dump(self.xgb_model, base + ".xgb.pkl")
        if self.label_encoder is not None:
            save_pickle(self.label_encoder, base + ".label_encoder.pkl")
        print(f"[save] Saved models/encoders using prefix {base}")

    def load_models(self, path_prefix="saved_models/hybrid_model"):
        """Load model and encoder from disk."""
        if os.path.exists(f"{path_prefix}_gbhm.pkl"):
            self.model = joblib.load(f"{path_prefix}_gbhm.pkl")
            print("Loaded GBHM model.")
        if os.path.exists(f"{path_prefix}_xgb.json"):
            self.xgb_model = xgb.Booster()
            self.xgb_model.load_model(f"{path_prefix}_xgb.json")
            print("Loaded XGB model.")
        if os.path.exists(f"{path_prefix}_encoder.pkl"):
            self.label_encoder = joblib.load(f"{path_prefix}_encoder.pkl")
            print("Loaded LabelEncoder.")
        print("Model state fully restored.")

    ###############################################################
    #####      Merge Label Encoders of Various models      ########
    ###############################################################
    def _merge_label_encoders_and_transform(self, existing_le: LabelEncoder, new_labels: np.ndarray) -> Tuple[LabelEncoder, np.ndarray]:
        """
        Merge classes between an existing LabelEncoder and new observed labels.
        Returns new LabelEncoder fitted on the union, and encoded new labels with new mapping.
        """
        # If existing_le is None, fit new on unique(new_labels)
        uniques_new = np.unique(new_labels)
        if existing_le is None:
            le = LabelEncoder()
            le.fit(uniques_new)
            return le, le.transform(new_labels).astype(int)

        # union classes
        existing_classes = list(existing_le.classes_)
        union = list(sorted(set(existing_classes).union(set(uniques_new)),
                            key=lambda x: str(x)))
        le_new = LabelEncoder()
        le_new.fit(union)
        encoded = le_new.transform(new_labels).astype(int)
        return le_new, encoded
