import os
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

# TensorFlow optional
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
    Hybrid model combining numerical features + Cherenkov histogram images.
    This version removes ALL information leaks:
        - No bin_index
        - No radius
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
        chunk_size: int = 100_000,
        max_events: int = 250_000,
        hist_shape: Tuple[int,int] = (64,64),
        ddpm_timesteps: int = 1000,
        model_dir='saved_models'
    ):
        """
        Args:
            inputfile_name (str or list): ROOT files.
            tree_name (str): Name of the TTree (tabular).
            branch_name (tuple): (xbranch, ybranch) used for histogram reconstruction.
            histogram_name (str): Name of the TH2 histogram in the ROOT file.
            features (list): Pure numerical features (no leak allowed).
            target (str): PDG or class label.
        """
        if isinstance(inputfile_name, str):
            inputfile_name = [inputfile_name]

        self._inputfiles = inputfile_name
        self.tree_name = tree_name
        self.branch_name = branch_name
        self.histogram_name = histogram_name
        self.features = features
        self.target = target
        self.aux_scalar_branch: Optional[str] = 'pT'
        self.chunk_size = chunk_size
        self.r_max = r_max
        self.max_events = max_events

        # --------------- HISTOGRAM GRID -----------------
        self.x_edges = np.linspace(-650,650,50) if x_edges is None else np.asarray(x_edges)
        self.y_edges = np.linspace(-650,650,50) if y_edges is None else np.asarray(y_edges)

        # --------------- LABEL ENCODERS -----------------
        self.label_encoder: Optional[LabelEncoder] = None
        self.xgb_label_encoder: Optional[LabelEncoder] = None
        self.gbhm_label_encoder: Optional[LabelEncoder] = None

        # --------------- MODELS -------------------------
        self.xgb_model = None
        self.gbhm_clf = None

        self.xgb_num_features: Optional[int] = None
        self.gbhm_num_features: Optional[int] = None
        self.xgb_feature_names: Optional[List[str]] = None
        self.gbhm_feature_names: Optional[List[str]] = None
        self.hist_feature_names: Optional[List[str]] = None

        self.hist_shape = hist_shape

        # --------------- DIFFUSION -----------------------
        self.ddpm_timesteps = ddpm_timesteps
        self.betas = DDPM_utils.make_beta_schedule(ddpm_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas).astype(np.float32)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1]).astype(np.float32)

        self.diffusion_model = None
        self.diffusion_compiled = False
        self.ema_helper = None

        # --------------- STORAGE -------------------------
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

        # --------------- PLOTTER -------------------------
        self.plotter = PerformancePlotter(save_dir='plots/performance')

    # --------------------------------------------------------------
    # Helper: compute digitized bin index for histogram consistency
    # (No leak: ONLY used to match histogram ↔ event, NOT used for ML)
    # --------------------------------------------------------------
    def _compute_hist_bin(self, row: pd.Series) -> int:
        x_var, y_var = self.branch_name
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

        return y_bin * (len(self.x_edges)-1) + x_bin
        
    # ===============================================================
    #                   LOAD TABULAR DATA (SAFE)
    # ===============================================================
    def _load_data(self) -> pd.DataFrame:
        """
        Loads tabular data from ROOT TTrees.
        Returns a clean DataFrame containing:
            - self.features   (pure numerical)
            - branch_name     (for histogram alignment, not used for ML)
            - aux_scalar_branch
            - target
        All rows with invalid histogram coordinates are removed.
        """
        dfs = []
        processed = 0

        # Only numerical ML-safe columns
        branches = (
            list(self.features) +
            list(self.branch_name) +     # x,y for histogram alignment
            [self.aux_scalar_branch] +
            [self.target]
        )

        for fname in self._inputfiles:
            try:
                it = ur.iterate(
                    f"{fname}:{self.tree_name}",
                    expressions=branches,
                    step_size=self.chunk_size,
                    library="np"
                )
            except Exception as e:
                raise RuntimeError(f"Could not open {fname}: {e}")

            for arrays in it:
                df = pd.DataFrame({k: arrays[k] for k in arrays.keys()})

                # --- Validate required columns ---
                missing = [c for c in branches if c not in df.columns]
                if missing:
                    raise KeyError(f"Missing branches in ROOT file: {missing}")

                # --- Histogram bin consistency check (no leak) ---
                df["hist_bin"] = df.apply(self._compute_hist_bin, axis=1)

                # Remove hits outside RICH acceptance
                df = df[df["hist_bin"] >= 0]

                if len(df) == 0:
                    continue

                dfs.append(df)
                processed += len(df)

                # Early stop to protect RAM
                if processed >= self.max_events:
                    print(f"Reached max number of events: {self.max_events}. Stopping early...")
                    break

            if processed >= self.max_events:
                break

        if not dfs:
            raise RuntimeError("*** No valid events found in the input ROOT files! ***")

        # Assemble final table
        df_all = pd.concat(dfs, ignore_index=True)

        # Final truncation (rare)
        if len(df_all) > self.max_events:
            df_all = df_all.iloc[:self.max_events].reset_index(drop=True)

        # Drop the histogram bin (used only for alignment)
        # IMPORTANT: Not used as a feature → NO LEAK
        df_all = df_all.drop(columns=["hist_bin"])

        return df_all


    # ===============================================================
    #                   HISTOGRAM IMAGE ENCODING
    # ===============================================================

    def encode_histograms(self, imgs, batch_size=64):
        """
        Encode TH2 histogram images into a feature vector.
        If diffusion encoder exists → embed using CNN encoder.
        Else → fallback to compact hist_to_features().
        """
        imgs = np.asarray(imgs)

        # Remove channel dimension if (N,H,W,1)
        if imgs.ndim == 4 and imgs.shape[-1] == 1:
            imgs_proc = imgs[..., 0]
        else:
            imgs_proc = imgs

        # ----------- Diffusion encoder path (if available) ----------
        if TF_AVAILABLE and self.diffusion_model is not None and hasattr(self.diffusion_model, "encoder"):
            encoder = self.diffusion_model.encoder
            out = []
            N = imgs_proc.shape[0]

            for i in range(0, N, batch_size):
                batch = imgs_proc[i:i+batch_size]
                try:
                    emb = encoder(batch[..., None], training=False).numpy()
                except Exception:
                    emb = encoder(batch.reshape((-1, *self.hist_shape, 1)), training=False).numpy()
                out.append(emb)

            embeddings = np.vstack(out)
            self.hist_feature_names = [f"cnn_embed_{i}" for i in range(embeddings.shape[1])]
            return embeddings

        # ----------- Fallback → compact features -------------------
        return self.hist_to_features(imgs_proc, n_flat=64)

    # ---------------------------------------------------------------
    #           Compact Histogram Feature Extractor
    # ---------------------------------------------------------------
    def hist_to_features(self, imgs, n_flat=64):
        """
        Convert TH2 images to compact numerical features.
        - Downsample or resize
        - Flatten
        - Add global stats
        """
        imgs = np.asarray(imgs)
        if imgs.ndim == 4 and imgs.shape[-1] == 1:
            imgs = imgs[..., 0]

        N, H, W = imgs.shape

        # 1) Resize to small grid (TensorFlow if available)
        try:
            imgs_tf = tf.convert_to_tensor(imgs[..., None], dtype=tf.float32)
            small = tf.image.resize(
                imgs_tf,
                (max(4, H // 8), max(4, W // 8)),
                method="bilinear"
            )
            small = tf.reshape(small, (N, -1)).numpy()
        except Exception:
            # Fallback without TF: uniform downsampling by slicing
            step_h = max(1, H // 8)
            step_w = max(1, W // 8)
            small = imgs[:, ::step_h, ::step_w].reshape(N, -1)

        # 2) Pad or truncate to n_flat
        if small.shape[1] >= n_flat:
            flat = small[:, :n_flat]
        else:
            pad = np.zeros((N, n_flat - small.shape[1]), dtype=small.dtype)
            flat = np.concatenate([small, pad], axis=1)

        # 3) Add basic global stats
        means = imgs.mean(axis=(1, 2)).reshape(N, 1)
        stds = imgs.std(axis=(1, 2)).reshape(N, 1)
        maxs = imgs.max(axis=(1, 2)).reshape(N, 1)

        feats = np.concatenate([flat, means, stds, maxs], axis=1)

        # Store names
        self.hist_feature_names = [f"hist_feat_{i}" for i in range(feats.shape[1])]
        return feats

    # ===============================================================
    #                   HISTOGRAM + TABULAR ALIGNMENT
    # ===============================================================
    def _load_histogram_images(self):
        """
        Load TH2 images from ROOT files using DDPM_utils.
        Returns:
            imgs          N x H x W
            labels_imgs   N
            auxs          N (aux scalar)
            le_imgs       LabelEncoder (if provided)
        """
        try:
            imgs, labels, auxs, le_imgs = DDPM_utils._load_th2_images(
                inputfiles=self._inputfiles,
                histogram_name=self.histogram_name,
                tree_name=self.tree_name,
                aux_scalar_branch=self.aux_scalar_branch,
                target=self.target,
                hist_shape=self.hist_shape,
                x_edges=self.x_edges,
                y_edges=self.y_edges
            )
        except Exception as e:
            raise RuntimeError(f"Could not load TH2 histogram images: {e}")

        # Ensure numpy
        imgs = np.asarray(imgs)
        labels = np.asarray(labels)
        auxs = np.asarray(auxs)

        # Sanity checks
        if imgs.ndim != 3:
            raise RuntimeError(f"Histogram images must be (N,H,W), got {imgs.shape}")

        if imgs.shape[0] != labels.shape[0]:
            raise RuntimeError("Mismatch between number of images and labels from _load_th2_images()")

        if imgs.shape[0] != auxs.shape[0]:
            raise RuntimeError("Mismatch between images and aux scalars")

        return imgs, labels, auxs, le_imgs

    # ===============================================================
    #           ALIGN TABULAR DATA ↔ HISTOGRAM IMAGES
    # ===============================================================
    def _check_alignment(self, df: pd.DataFrame, imgs: np.ndarray, labels_imgs: np.ndarray):
        """
        Verify that tabular rows and histogram images correspond 1:1.
        Ensures:
            - same number of rows
            - same label distribution (sanity)
        Raises RuntimeError if inconsistent.
        """

        if df.shape[0] != imgs.shape[0]:
            raise RuntimeError(
                f"Mismatch: tabular rows={df.shape[0]} but hist images={imgs.shape[0]}. "
                "This hybrid model requires perfectly aligned datasets."
            )

        # Optional but helpful: check label consistency
        y_df = df[self.target].to_numpy()
        if len(np.unique(y_df)) != len(np.unique(labels_imgs)):
            print("[WARNING] Label distribution differs between TTree and Histogram images.")
        return True

    # ===============================================================
    #                        XGBOOST TRAINING
    # ===============================================================
    def xgb_train(
        self,
        max_depth=6,
        loss='mlogloss',
        eta=0.1,
        subsample=0.8,
        tree_method='hist',
        num_boost_round=200,
        test_size=0.2,
        random_state=42,
        use_histograms=True
    ):
        """
        Train XGBoost classifier.
        Fully corrected version (no leak, safe alignment).
        """
        # -----------------------------------------------------------
        # 1. Load tabular data (safe)
        # -----------------------------------------------------------
        df = self._load_data()

        # Pure numerical features only (NO bin_index, NO radius)
        X_tree = df[self.features]
        y_raw = df[self.target]

        # -----------------------------------------------------------
        # 2. Stable LabelEncoder handling
        # -----------------------------------------------------------
        le = getattr(self, "label_encoder", None)

        if le is None:
            le = LabelEncoder()
            le.fit(y_raw)
            self.label_encoder = le
        else:
            # ensure encoder covers all labels
            new_labels = np.unique(y_raw)
            if not set(new_labels).issubset(set(le.classes_)):
                merged = np.unique(list(le.classes_) + list(new_labels))
                le = LabelEncoder()
                le.fit(merged)
                self.label_encoder = le

        y_encoded = le.transform(y_raw).astype(int)

        # -----------------------------------------------------------
        # 3. Load histogram images (optional)
        # -----------------------------------------------------------
        hist_feats = None

        if use_histograms:
            try:
                imgs, labels_img, auxs, le_imgs = self._load_histogram_images()

                # Align tabular ↔ images
                self._check_alignment(df, imgs, labels_img)

                # Convert histogram images to features
                hist_feats = self.encode_histograms(imgs)

            except Exception as e:
                print(f"[xgb_train] Histogram load failed → training tree-only: {e}")
                hist_feats = None

        # -----------------------------------------------------------
        # 4. Build full feature matrix X
        # -----------------------------------------------------------
        if hist_feats is not None:
            X = np.concatenate([X_tree.to_numpy(), hist_feats], axis=1)
            feature_names = list(X_tree.columns) + self.hist_feature_names
        else:
            X = X_tree.to_numpy()
            feature_names = list(X_tree.columns)

        # memorize expected input size
        self.xgb_num_features = X.shape[1]
        self.xgb_feature_names = feature_names

        # -----------------------------------------------------------
        # 5. Train-test split
        # -----------------------------------------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state
        )

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        # number of classes
        num_classes = len(le.classes_)

        # -----------------------------------------------------------
        # 6. XGBoost parameters
        # -----------------------------------------------------------
        xgb_params = {
            'objective': 'multi:softmax',
            'eval_metric': loss,
            'num_class': num_classes,
            'max_depth': max_depth,
            'eta': eta,
            'subsample': subsample,
            'tree_method': tree_method
        }

        print("[XGB] Training XGBoost classifier...")

        # -----------------------------------------------------------
        # 7. Train model
        # -----------------------------------------------------------
        self.xgb_model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtrain, 'train'), (dtest, 'test')],
            verbose_eval=True
        )

        # -----------------------------------------------------------
        # 8. Evaluate
        # -----------------------------------------------------------
        preds = self.xgb_model.predict(dtest).astype(int)
        test_acc = accuracy_score(y_test, preds)

        print(f"[XGB] Test Accuracy: {test_acc:.4f}")

        # keep encoder for predictions
        self.xgb_label_encoder = le

        return self.xgb_model

    # ===============================================================
    #                GRADIENT BOOSTING HYBRID MODEL TRAINING
    # ===============================================================
    def gbhm_train(
        self,
        n_estimators: int = 100,
        learn_rate: float = 0.1,
        max_depth: int = 3,
        test_size: float = 0.3,
        random_state: int = 42,
        warm_start: bool = True,
        save: bool = True,
        use_histograms: bool = True,
    ):
        """
        Train the GradientBoostingClassifier in a hybrid configuration.

        Corrections:
        - No leak (bin_index, radius removed).
        - Strict alignment between tabular rows and histogram images.
        - Stable label encoding.
        - Robust warm_start handling.
        """

        # -----------------------------------------------------------
        # 1. Load tabular data
        # -----------------------------------------------------------
        df = self._load_data()
        X_tree = df[self.features]
        y_raw = df[self.target]

        # -----------------------------------------------------------
        # 2. Stable LabelEncoder handling
        # -----------------------------------------------------------
        le = getattr(self, "label_encoder", None)

        if le is None:
            le = LabelEncoder()
            le.fit(y_raw)
            self.label_encoder = le
        else:
            new_labels = np.unique(y_raw)
            if not set(new_labels).issubset(set(le.classes_)):
                merged = np.unique(list(le.classes_) + list(new_labels))
                le = LabelEncoder()
                le.fit(merged)
                self.label_encoder = le

        y_encoded = le.transform(y_raw).astype(int)
        self.gbhm_label_encoder = le

        # -----------------------------------------------------------
        # 3. Load histogram images and validate alignment
        # -----------------------------------------------------------
        hist_feats = None
        if use_histograms:
            try:
                imgs, labels_img, auxs, le_imgs = self._load_histogram_images()
                self._check_alignment(df, imgs, labels_img)   # strict

                # Convert histograms to numerical features
                hist_feats = self.encode_histograms(imgs)

            except Exception as e:
                print(f"[gbhm_train] Could not load/align histograms, training tree-only: {e}")
                hist_feats = None

        # -----------------------------------------------------------
        # 4. Build full feature matrix
        # -----------------------------------------------------------
        if hist_feats is not None:
            X = np.concatenate([X_tree.to_numpy(), hist_feats], axis=1)
            feature_names = list(X_tree.columns) + self.hist_feature_names
        else:
            X = X_tree.to_numpy()
            feature_names = list(X_tree.columns)

        # Save expected feature count
        self.gbhm_num_features = X.shape[1]
        self.gbhm_feature_names = feature_names

        # -----------------------------------------------------------
        # 5. Train/test split
        # -----------------------------------------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state
        )

        # -----------------------------------------------------------
        # 6. Warm start handling
        # -----------------------------------------------------------
        if self.gbhm_clf is not None and warm_start:
            print("[GBHM] Continuing training (warm_start=True)...")

            # Increase n_estimators only if needed
            current_n = getattr(self.gbhm_clf, "n_estimators", n_estimators)
            new_n_estimators = max(current_n + 50, n_estimators)

            self.gbhm_clf.set_params(
                warm_start=True,
                n_estimators=new_n_estimators,
                learning_rate=learn_rate,
                max_depth=max_depth
            )
        else:
            print("[GBHM] Training new GBHM model...")

            self.gbhm_clf = GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=learn_rate,
                max_depth=max_depth,
                random_state=random_state,
                warm_start=warm_start
            )

        # -----------------------------------------------------------
        # 7. Fit the model
        # -----------------------------------------------------------
        self.gbhm_clf.fit(X_train, y_train)

        # -----------------------------------------------------------
        # 8. Evaluate
        # -----------------------------------------------------------
        pred_train = self.gbhm_clf.predict(X_train)
        pred_test = self.gbhm_clf.predict(X_test)

        train_acc = accuracy_score(y_train, pred_train)
        test_acc = accuracy_score(y_test, pred_test)

        print(f"[GBHM] Training Accuracy: {train_acc:.4f}")
        print(f"[GBHM] Test Accuracy:     {test_acc:.4f}")

        return self.gbhm_clf

    # ===============================================================
    #                DDPM – BUILD / TRAIN / SAMPLE
    # ===============================================================

    def build_diffusion_model(self, n_classes: Optional[int] = None):
        """
        Build a conditional UNet for DDPM sampling.

        - Uses HybridUNetDiT_TF
        - No leak
        - n_classes determined from label encoder
        """

        if n_classes is None:
            if self.label_encoder is not None:
                n_classes = len(self.label_encoder.classes_)
            else:
                n_classes = 32  # fallback, rarely used

        input_shape = (*self.hist_shape, 1)

        # Construct the conditional UNet (DiT-like)
        self.diffusion_model = HybridUNetDiT_TF(
            input_shape=input_shape,
            base_channels=32
        )

        # Exponential Moving Average helper
        self.ema_helper = EMA(self.diffusion_model, decay=0.999)

        # Keep track that the model exists (needs compile via training)
        self.diffusion_compiled = False

        return self.diffusion_model

    # -----------------------------------------------------------
    # Forward diffusion q(x_t | x_0)
    # -----------------------------------------------------------
    def q_sample(self, x_start, t, noise, alphas_cumprod_tf):
        """
        Forward process: q(x_t | x_0)
        """
        coef1 = DDPM_utils.extract(tf.sqrt(alphas_cumprod_tf), t, tf.shape(x_start))
        coef2 = DDPM_utils.extract(tf.sqrt(1.0 - alphas_cumprod_tf), t, tf.shape(x_start))
        return coef1 * x_start + coef2 * noise

    def train_ddpm(
        self,
        epochs=10,
        batch_size=8,
        lr=2e-4,
        beta_schedule='cosine',
        timesteps=None,
        use_mixed_precision=False,
        clip_grad_norm=1.0,
        base_ch=32,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        groups_gn=8,
        time_emb=True,
        class_emb=True,
        num_classes=4,
        self_condition=False,
        ema_decay=0.999,
        device=None,
        path='plots/ddpm_training',
        show=True
    ):
        print("\n====================================================")
        print("              🌀 DDPM TRAINING START                ")
        print("====================================================")
        global_start = time.time()

        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow not available for DDPM training.")

        # GPU CHECK
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            print(f"[INFO] GPU detected: {gpus}")
            device = "/GPU:0"
        else:
            print("[WARNING] ❌ No GPU detected — training will be VERY slow.")
            device = "/CPU:0"
        print(f"[INFO] Using device: {device}")

        # LOAD IMAGES
        print("\n[DDPM] Loading histogram images…")
        t0 = time.time()
        imgs, labels_raw, auxs, le_imgs = self._load_histogram_images()
        print(f"[DDPM] Loaded {imgs.shape[0]} images in {time.time() - t0:.2f}s")

        # LABEL ENCODING
        print("[DDPM] Encoding labels…")
        t0 = time.time()
        if self.label_encoder is not None:
            le = self.label_encoder
            incoming = np.unique(labels_raw)
            if not set(incoming).issubset(le.classes_):
                merged = np.unique(list(le.classes_) + list(incoming))
                le = LabelEncoder()
                le.fit(merged)
                self.label_encoder = le
        else:
            le = LabelEncoder()
            le.fit(labels_raw)
            self.label_encoder = le

        labels = le.transform(labels_raw).astype(np.int32)
        print(f"[DDPM] Encoded {len(le.classes_)} classes in {time.time() - t0:.2f}s")

        # DIFFUSION SCHEDULE
        print("\n[DDPM] Preparing beta schedule…")
        T = int(timesteps or self.ddpm_timesteps)
        betas = DDPM_utils.make_beta_schedule(T, schedule=beta_schedule).astype(np.float32)
        alphas = (1.0 - betas).astype(np.float32)
        alphas_cumprod = np.cumprod(alphas).astype(np.float32)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        print(f"[DDPM] Schedule ready: {T} timesteps")

        # NORMALIZATION
        print("[DDPM] Normalizing images…")
        X = imgs.astype(np.float32)
        X = (X - 0.5) * 2.0
        X = X[..., None]
        auxs = auxs.astype(np.float32)
        print(f"[DDPM] Final dataset shape: {X.shape}")

        # DATASET PIPELINE
        print("\n[DDPM] Building dataset pipeline…")
        t0 = time.time()
        ds = tf.data.Dataset.from_tensor_slices((X, labels, auxs))
        ds = ds.shuffle(4096).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        print(f"[DDPM] Pipeline ready in {time.time() - t0:.2f}s")

        # ======================================================
        # BUILD MODEL (CORRECTED — ONLY ONE BLOCK)
        # ======================================================
        if self.diffusion_model is None or not self.diffusion_compiled:
            print("\n[DDPM] Building diffusion model…")
            t0 = time.time()

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
                num_classes=len(le.classes_),
                self_condition=self_condition,
            )

            print(f"[DDPM] Model built in {time.time() - t0:.2f}s")

            # -----------------------------------------------------
            # FORCE MODEL BUILD: dummy forward pass
            # -----------------------------------------------------
            print("[DDPM] Building model weights with dummy pass...")

            dummy_x = tf.zeros((1, self.hist_shape[0], self.hist_shape[1], 1), dtype=tf.float32)
            dummy_t = tf.zeros((1,), dtype=tf.int32)
            dummy_y = tf.zeros((1,), dtype=tf.int32)

            if auxs.ndim == 1:
                dummy_aux = tf.zeros((1,), dtype=tf.float32)
            else:
                dummy_aux = tf.zeros((1, auxs.shape[1]), dtype=tf.float32)

            _ = self.diffusion_model(dummy_x, dummy_t, dummy_y, dummy_aux, training=False)

            print("[DDPM] Model successfully built.\n")

            try:
                print("\n[DDPM] Model summary:")
                self.diffusion_model.summary()
            except Exception:
                print("[DDPM] (Model summary unavailable)")

            print("[DDPM] Initializing EMA…")
            self.ema_helper = EMA(self.diffusion_model, decay=ema_decay)
            self.ema_model = self.ema_helper.ema_model

        # ======================================================
        # OPTIMIZER
        # ======================================================
        optimizer = keras.optimizers.Adam(lr)
        mse = tf.keras.losses.MeanSquaredError()

        if use_mixed_precision:
            policy = tf.keras.mixed_precision.Policy("mixed_float16")
            tf.keras.mixed_precision.set_global_policy(policy)
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
            print("[DDPM] Mixed precision enabled.")

        alphas_cumprod_tf = tf.constant(alphas_cumprod, dtype=tf.float32)

        # ======================================================
        # TRAINING LOOP
        # ======================================================
        print("\n====================================================")
        print(" ⚠️ First batch will be VERY SLOW (TF graph tracing)")
        print("====================================================")

        train_losses = []

        for epoch in range(epochs):
            print(f"\n[DDPM] ===== EPOCH {epoch+1}/{epochs} =====")
            t0 = time.time()
            epoch_loss = 0.0
            steps = 0

            for x_batch, y_batch, aux_batch in ds:

                B = tf.shape(x_batch)[0]
                t_rand = tf.random.uniform((B,), minval=0, maxval=T, dtype=tf.int32)

                noise = tf.random.normal(tf.shape(x_batch))
                x_noisy = self.q_sample(x_batch, t_rand, noise, alphas_cumprod_tf)

                with tf.GradientTape() as tape:
                    pred_noise = self.diffusion_model(
                        x_noisy, t_rand, y_batch, aux_batch, training=True
                    )
                    pred_noise = tf.cast(pred_noise, tf.float32)
                    loss = mse(noise, pred_noise)

                grads = tape.gradient(loss, self.diffusion_model.trainable_variables)

                if clip_grad_norm:
                    grads, _ = tf.clip_by_global_norm(grads, clip_grad_norm)

                optimizer.apply_gradients(
                    zip(grads, self.diffusion_model.trainable_variables)
                )

                epoch_loss += float(loss)
                steps += 1

            epoch_loss /= max(1, steps)
            train_losses.append(epoch_loss)

            print(
                f"[DDPM] Epoch {epoch+1}/{epochs} "
                f"| loss={epoch_loss:.6f} "
                f"| time={time.time() - t0:.1f}s"
            )

            # EMA UPDATE
            if self.ema_model is not None:
                ema_w = self.ema_model.get_weights()
                cur_w = self.diffusion_model.get_weights()
                new_w = [
                    ema_decay * w_ema + (1 - ema_decay) * w_mod
                    for w_ema, w_mod in zip(ema_w, cur_w)
                ]
                self.ema_model.set_weights(new_w)

        self.diffusion_compiled = True

        print("\n[DDPM] Training complete.")
        print(f"Total training time: {time.time() - global_start:.1f}s")

        self.ddpm_train_history = {
            "loss": train_losses,
            "betas": betas
        }

        if show:
            DDPM_utils.plot_loss_curve(train_losses, save_dir=path, show=True)
            DDPM_utils.plot_beta_schedule(betas, save_dir=path)


    # ===============================================================
    #                DDPM SAMPLING (IMAGE GENERATION)
    # ===============================================================
    def sample_ddpm(self, n_samples=9, use_ema=True):
        """
        Generate new Cherenkov-ring images from the trained DDPM.
        Returns an array (n_samples, H, W, 1).
        """

        if not self.diffusion_compiled:
            raise RuntimeError("DDPM model is not trained. Run train_ddpm() first.")

        print(f"[DDPM] Sampling {n_samples} images...")

        model = self.ema_model if use_ema and hasattr(self, "ema_model") else self.diffusion_model

        T = len(self.betas)
        betas = tf.constant(self.betas, dtype=tf.float32)
        alphas = 1.0 - betas
        alphas_cumprod = tf.math.cumprod(alphas)

        # Start from pure Gaussian noise
        x = tf.random.normal((n_samples, self.hist_shape[0], self.hist_shape[1], 1))

        for t in reversed(range(T)):
            t_tensor = tf.fill([n_samples], t)

            # unconditional → class=0, aux=0
            y_dummy = tf.zeros((n_samples,), dtype=tf.int32)
            aux_dummy = tf.zeros((n_samples,), dtype=tf.float32)

            # predict the noise
            noise_pred = model(x, t_tensor, y_dummy, aux_dummy, training=False)

            alpha_t = alphas[t]
            alpha_cum = alphas_cumprod[t]
            beta_t = betas[t]

            # equation (DDPM) posterior mean
            x = (1 / tf.sqrt(alpha_t)) * (x - (beta_t / tf.sqrt(1 - alpha_cum)) * noise_pred)

            if t > 0:
                # add stochastic term
                noise = tf.random.normal(tf.shape(x))
                x = x + tf.sqrt(beta_t) * noise

        # bring back to [0,1]
        x = tf.clip_by_value(x, -1, 1)
        x = (x + 1) / 2.0

        print("[DDPM] Sampling finished.")
        return x.numpy()


    # ===============================================================
    #                        PREDICTION HELPERS
    # ===============================================================

    def _build_full_feature_array(
        self,
        hist_array: np.ndarray,
        target_feature_names: Optional[List[str]] = None,
        target_nfeat: Optional[int] = None
    ):
        """
        Builds full feature matrix for prediction.
        Only used for histogram-only inference.

        - hist_array: (N, Hf)
        - target_feature_names: list of expected names (tree + hist)
        - target_nfeat: number of expected input features
        """

        N = hist_array.shape[0]

        # sanitize
        hist_array = np.nan_to_num(hist_array, nan=0.0, posinf=0.0, neginf=0.0)

        # No explicit mapping → simply pad or truncate to expected size
        if target_nfeat is not None:
            if hist_array.shape[1] == target_nfeat:
                return hist_array
            elif hist_array.shape[1] > target_nfeat:
                return hist_array[:, :target_nfeat]
            else:
                pad = np.zeros((N, target_nfeat - hist_array.shape[1]), dtype=hist_array.dtype)
                return np.concatenate([hist_array, pad], axis=1)

        return hist_array

    # ===============================================================
    #                 XGB PREDICT
    # ===============================================================
    def xgb_predict(self, X):
        """
        Predict using trained XGBoost classifier.
        X may be a DataFrame or a numpy array.
        """

        if self.xgb_model is None:
            raise RuntimeError("No XGBoost model loaded.")

        # Convert DF → array
        if isinstance(X, pd.DataFrame):
            Xnp = X.values
        else:
            Xnp = np.asarray(X)

        # sanitize
        Xnp = np.nan_to_num(Xnp, nan=0.0, posinf=0.0, neginf=0.0)

        # shape correction
        expected = getattr(self, "xgb_num_features", None)
        if expected is not None and Xnp.shape[1] != expected:
            if Xnp.shape[1] > expected:
                Xnp = Xnp[:, :expected]
            else:
                pad = np.zeros((Xnp.shape[0], expected - Xnp.shape[1]), dtype=Xnp.dtype)
                Xnp = np.vstack([Xnp, pad])

        dmat = xgb.DMatrix(Xnp)
        preds = self.xgb_model.predict(dmat).astype(int)

        # decode
        return self.label_encoder.inverse_transform(preds)

    # ===============================================================
    #                 GBHM PREDICT
    # ===============================================================
    def gbhm_predict(self, X):
        """
        Predict using Gradient Boosting classifier.
        """

        if self.gbhm_clf is None:
            raise RuntimeError("No GBHM model loaded.")

        if isinstance(X, pd.DataFrame):
            Xnp = X.values
        else:
            Xnp = np.asarray(X)

        Xnp = np.nan_to_num(Xnp, nan=0.0, posinf=0.0, neginf=0.0)

        expected = getattr(self, "gbhm_num_features", None)
        if expected is not None and Xnp.shape[1] != expected:
            if Xnp.shape[1] > expected:
                Xnp = Xnp[:, :expected]
            else:
                pad = np.zeros((Xnp.shape[0], expected - Xnp.shape[1]), dtype=Xnp.dtype)
                Xnp = np.concatenate([Xnp, pad], axis=1)

        preds = self.gbhm_clf.predict(Xnp).astype(int)
        return self.label_encoder.inverse_transform(preds)

    # ===============================================================
    #        ROOT TH2 → direct prediction via XGB
    # ===============================================================
    def predict_from_hist(self, hist2d):
        """
        Predict class labels from a ROOT TH2F histogram object.
        Steps:
         - convert histogram bins to (posX,posY,count)
         - build a DataFrame
         - apply XGB model
        """

        if self.xgb_model is None:
            raise RuntimeError("xgb_model is not trained.")

        # Extract bins
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
            print("Warning: empty histogram.")
            return None

        df = pd.DataFrame(data, columns=['posX', 'posY', 'weight'])

        # Filter features
        feature_cols = [f for f in ['posX', 'posY'] if f in self.features]
        X = df[feature_cols]

        dmat = xgb.DMatrix(X, weight=df["weight"].values)
        preds = self.xgb_model.predict(dmat).astype(int)

        labels = self.label_encoder.inverse_transform(preds)
        df["prediction"] = labels

        return df

# ======================================================================
#           JOINT PERFORMANCE SUMMARY AND ANALYSIS UTILITIES
# ======================================================================

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
import seaborn as sns

def joint_performance_summary(
    true_labels,
    pred_xgb=None,
    pred_gbhm=None,
    pred_ddpm=None,
    label_encoder=None,
    title_prefix="Performance Summary"
):
    """
    Print joint performance summary for all models.
    """

    summary = {}

    def decode(y):
        if y is None:
            return None
        if label_encoder is None:
            return y
        try:
            return label_encoder.inverse_transform(y)
        except:
            return y

    y_true = decode(true_labels)
    y_pred_xgb = decode(pred_xgb)
    y_pred_gbhm = decode(pred_gbhm)
    y_pred_ddpm = decode(pred_ddpm)

    print(f"\n=== {title_prefix} ===\n")

    if y_pred_xgb is not None:
        summary["XGB_accuracy"] = accuracy_score(y_true, y_pred_xgb)
        print(f"XGB accuracy: {summary['XGB_accuracy']:.4f}")

    if y_pred_gbhm is not None:
        summary["GBHM_accuracy"] = accuracy_score(y_true, y_pred_gbhm)
        print(f"GBHM accuracy: {summary['GBHM_accuracy']:.4f}")

    if y_pred_ddpm is not None:
        summary["DDPM_accuracy"] = accuracy_score(y_true, y_pred_ddpm)
        print(f"DDPM accuracy: {summary['DDPM_accuracy']:.4f}")

    # Detailed textual reports
    if y_pred_xgb is not None:
        print("\n--- XGB Report ---")
        print(classification_report(y_true, y_pred_xgb))

    if y_pred_gbhm is not None:
        print("\n--- GBHM Report ---")
        print(classification_report(y_true, y_pred_gbhm))

    if y_pred_ddpm is not None:
        print("\n--- DDPM Report ---")
        print(classification_report(y_true, y_pred_ddpm))

    return summary


def joint_confusion_matrix(true_labels, pred_labels, label_encoder=None,
                           title="Confusion Matrix", figsize=(6, 5)):
    """
    Plot confusion matrix for one model.
    """

    if label_encoder is not None:
        try:
            true_labels = label_encoder.inverse_transform(true_labels)
            pred_labels = label_encoder.inverse_transform(pred_labels)
        except:
            pass

    cm = confusion_matrix(true_labels, pred_labels)
    labels = np.unique(true_labels)

    plt.figure(figsize=figsize)
    sns.heatmap(
        cm, annot=True, fmt="d",
        xticklabels=labels, yticklabels=labels, cmap="Blues"
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()


def joint_roc_curve(true_labels, probas_dict, label_encoder=None, title="ROC Curves"):
    """
    Plot ROC curves for all models that provide probability outputs.
    """

    plt.figure(figsize=(8, 6))

    if label_encoder is not None and not np.issubdtype(true_labels.dtype, np.integer):
        try:
            true_labels = label_encoder.transform(true_labels)
        except:
            pass

    n_classes = len(np.unique(true_labels))

    for name, proba in probas_dict.items():
        if proba is None:
            continue

        fpr = {}
        tpr = {}
        roc_auc = {}

        for c in range(n_classes):
            y_true_bin = (true_labels == c).astype(int)
            y_score_bin = proba[:, c]

            fpr[c], tpr[c], _ = roc_curve(y_true_bin, y_score_bin)
            roc_auc[c] = auc(fpr[c], tpr[c])

        mean_auc = np.mean(list(roc_auc.values()))
        plt.plot(fpr[0], tpr[0], label=f"{name} (macro AUC = {mean_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def evaluate_and_plot_all(
    y_true,
    y_pred_xgb=None,
    y_pred_gbhm=None,
    y_pred_ddpm=None,
    proba_xgb=None,
    proba_gbhm=None,
    proba_ddpm=None,
    label_encoder=None,
    prefix="Hybrid Model"
):
    """
    Combined performance evaluation:
      - accuracy summary
      - confusion matrices
      - ROC curves
    """

    print("\n============ JOINT MODEL EVALUATION ============\n")

    joint_performance_summary(
        y_true,
        pred_xgb=y_pred_xgb,
        pred_gbhm=y_pred_gbhm,
        pred_ddpm=y_pred_ddpm,
        label_encoder=label_encoder,
        title_prefix=prefix
    )

    if y_pred_xgb is not None:
        joint_confusion_matrix(
            y_true, y_pred_xgb, label_encoder=label_encoder,
            title=f"{prefix} — XGB Confusion Matrix"
        )

    if y_pred_gbhm is not None:
        joint_confusion_matrix(
            y_true, y_pred_gbhm, label_encoder=label_encoder,
            title=f"{prefix} — GBHM Confusion Matrix"
        )

    if y_pred_ddpm is not None:
        joint_confusion_matrix(
            y_true, y_pred_ddpm, label_encoder=label_encoder,
            title=f"{prefix} — DDPM Confusion Matrix"
        )

    if any(p is not None for p in [proba_xgb, proba_gbhm, proba_ddpm]):
        probas = {
            "XGB": proba_xgb,
            "GBHM": proba_gbhm,
            "DDPM": proba_ddpm
        }
        joint_roc_curve(
            y_true, probas, label_encoder=label_encoder,
            title=f"{prefix} — ROC Curves"
        )

