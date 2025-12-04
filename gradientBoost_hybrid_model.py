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
        self.x_edges = np.linspace(-650,650,65) if x_edges is None else np.asarray(x_edges)
        self.y_edges = np.linspace(-650,650,65) if y_edges is None else np.asarray(y_edges)

        # --------------- LABEL ENCODERS -----------------
        self.label_encoder: Optional[LabelEncoder] = None
        self.xgb_label_encoder: Optional[LabelEncoder] = None
        self.gbhm_label_encoder: Optional[LabelEncoder] = None
        self.hgbm_label_encoder: Optional[LabelEncoder] = None

        # --------------- MODELS -------------------------
        self.xgb_model = None
        self.gbhm_clf = None
        self.hgbm_clf = None

        self.xgb_num_features: Optional[int] = None
        self.gbhm_num_features: Optional[int] = None
        self.hgbm_num_features: Optional[int] = None   # NEW ✔️

        self.xgb_feature_names: Optional[List[str]] = None
        self.gbhm_feature_names: Optional[List[str]] = None
        self.hgbm_feature_names: Optional[List[str]] = None  # NEW ✔️
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
    #          LOAD HISTOGRAM IMAGES (NEW — FROM posX,posY)
    # ===============================================================
    def _load_histogram_images(self):
        """
        Build TH2 histogram images directly from the TTree (posX,posY).
        Ensures PERFECT alignment with the tabular data.
        """
        # 1) Load tabular rows (same filtering, same number of events)
        df = self._load_data()

        # 2) Build TH2 images from posX,posY
        imgs = self._build_histogram_images_from_tree(df)

        # 3) Extract labels and auxiliary scalar
        labels = df[self.target].values
        auxs   = df[self.aux_scalar_branch].values

        # 4) Build a minimal LabelEncoder for consistency
        le = LabelEncoder().fit(labels)

        return imgs, labels, auxs, le


    def _build_histogram_images_from_tree(self, df):
        """
        Build TH2-like images from (posX,posY) coordinates.
        Perfect alignment: one image per event.
        """
        N = df.shape[0]
        H, W = self.hist_shape

        imgs = np.zeros((N, H, W), dtype=np.float32)

        x_edges = self.x_edges
        y_edges = self.y_edges

        xbins = len(x_edges) - 1
        ybins = len(y_edges) - 1

        if xbins != H or ybins != W:
            raise RuntimeError("x_edges / y_edges inconsistent with hist_shape")

        xs = df[self.branch_name[0]].values
        ys = df[self.branch_name[1]].values

        x_idx = np.digitize(xs, x_edges) - 1
        y_idx = np.digitize(ys, y_edges) - 1

        valid = (x_idx >= 0) & (x_idx < W) & (y_idx >= 0) & (y_idx < H)

        imgs[valid, y_idx[valid], x_idx[valid]] = 1.0

        return imgs
        
    # ===============================================================
    #        CONVERT HISTOGRAM IMAGES → FLATTENED FEATURES
    # ===============================================================
    def encode_histograms(self, imgs: np.ndarray) -> np.ndarray:
        """
        Convert histogram images (N, H, W) → flattened features (N, H*W).

        This ensures:
          - deterministic mapping
          - no leak
          - numerical-friendly feature format
        """

        if imgs.ndim != 3:
            raise RuntimeError(f"encode_histograms: expected (N,H,W), got {imgs.shape}")

        N, H, W = imgs.shape
        feats = imgs.reshape(N, H * W).astype(np.float32)

        # store hist feature names once
        if self.hist_feature_names is None:
            self.hist_feature_names = [f"hist_{i}" for i in range(H * W)]

        return feats



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
        use_histograms=False
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
    #        HIST-GRADIENT BOOSTING (HGBM) – FAST MODERN BOOSTER
    # ===============================================================
    def hgbm_train(
        self,
        learning_rate: float = 0.1,
        max_depth: int = None,
        max_iter: int = 300,
        test_size: float = 0.3,
        random_state: int = 42,
        use_histograms: bool = True,
    ):
        """
        Train a HistGradientBoostingClassifier (modern, very fast).
        Equivalent to LightGBM-style histogram boosting.

        Key points:
          - STRICT alignment with TH2 images (same as GBHM)
          - VERY FAST (seconds vs minutes)
          - Supports TH2 + tabular features
          - No warm_start needed (training is extremely fast)
        """

        from sklearn.ensemble import HistGradientBoostingClassifier

        # -----------------------------------------------------------
        # 1. Load tabular data
        # -----------------------------------------------------------
        df = self._load_data()
        X_tree = df[self.features]
        y_raw = df[self.target]

        # -----------------------------------------------------------
        # 2. Stable LabelEncoder (same as XGB + GBHM)
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
        self.hgbm_label_encoder = le

        # -----------------------------------------------------------
        # 3. Load histogram images and align
        # -----------------------------------------------------------
        hist_feats = None
        if use_histograms:
            try:
                imgs, labels_img, auxs, le_imgs = self._load_histogram_images()

                # MUST match perfectly
                self._check_alignment(df, imgs, labels_img)

                # Flatten TH2 → features
                hist_feats = self.encode_histograms(imgs)

            except Exception as e:
                print(f"[HGBM] Histogram load failed → training tree-only: {e}")
                hist_feats = None

        # -----------------------------------------------------------
        # 4. Build final feature matrix
        # -----------------------------------------------------------
        if hist_feats is not None:
            X = np.concatenate([X_tree.to_numpy(), hist_feats], axis=1)
            feature_names = list(X_tree.columns) + self.hist_feature_names
        else:
            X = X_tree.to_numpy()
            feature_names = list(X_tree.columns)

        # Save expected shape
        self.hgbm_num_features = X.shape[1]
        self.hgbm_feature_names = feature_names

        # -----------------------------------------------------------
        # 5. Train/test split
        # -----------------------------------------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state
        )

        # -----------------------------------------------------------
        # 6. Train HGBM (VERY FAST)
        # -----------------------------------------------------------
        print("[HGBM] Training HistGradientBoostingClassifier...")

        self.hgbm_clf = HistGradientBoostingClassifier(
            learning_rate=learning_rate,
            max_depth=max_depth,
            max_iter=max_iter,
            random_state=random_state,
            verbose=1,             # shows progress bar
        )

        self.hgbm_clf.fit(X_train, y_train)

        # -----------------------------------------------------------
        # 7. Evaluate
        # -----------------------------------------------------------
        pred_train = self.hgbm_clf.predict(X_train)
        pred_test  = self.hgbm_clf.predict(X_test)

        train_acc = accuracy_score(y_train, pred_train)
        test_acc  = accuracy_score(y_test, pred_test)

        print(f"[HGBM] Training Accuracy: {train_acc:.4f}")
        print(f"[HGBM] Test Accuracy:     {test_acc:.4f}")

        return self.hgbm_clf


    def hgbm_predict(self, X):
        """
        Predict using HistGradientBoostingClassifier.
        """

        if not hasattr(self, "hgbm_clf") or self.hgbm_clf is None:
            raise RuntimeError("No HGBM model loaded.")

        if isinstance(X, pd.DataFrame):
            Xnp = X.values
        else:
            Xnp = np.asarray(X)

        Xnp = np.nan_to_num(Xnp, nan=0.0, posinf=0.0, neginf=0.0)

        expected = getattr(self, "hgbm_num_features", None)
        if expected is not None and Xnp.shape[1] != expected:
            if Xnp.shape[1] > expected:
                Xnp = Xnp[:, :expected]
            else:
                pad = np.zeros((Xnp.shape[0], expected - Xnp.shape[1]), dtype=Xnp.dtype)
                Xnp = np.concatenate([Xnp, pad], axis=1)

        preds = self.hgbm_clf.predict(Xnp).astype(int)
        return self.label_encoder.inverse_transform(preds)


    # ===============================================================
    #           DDPM TRAINING (VERY SIMPLE VERSION)
    # ===============================================================
    def train_ddpm(self, imgs, epochs=1, batch_size=128):
        """
        Minimal DDPM training loop — simplified noise prediction.
        imgs must be shape (N, 64, 64, 1)
        """
        import tensorflow as tf
        from tensorflow import keras
        import numpy as np

        if imgs.ndim == 3:
            imgs = imgs[..., None]

        optimizer = keras.optimizers.Adam(1e-4)

        for epoch in range(epochs):
            print(f"[DDPM] Epoch {epoch+1}/{epochs}")

            for i in range(0, len(imgs), batch_size):
                batch = imgs[i:i+batch_size]

                # add random Gaussian noise
                noise = np.random.randn(*batch.shape).astype("float32")

                with tf.GradientTape() as tape:
                    pred = self.ddpm_model(batch, training=True)
                    loss = tf.reduce_mean((pred - noise)**2)

                grads = tape.gradient(loss, self.ddpm_model.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.ddpm_model.trainable_variables))

                # EMA update
                for w, w_ema in zip(self.ddpm_model.weights, self.ema_model.weights):
                    w_ema.assign(0.999 * w_ema + 0.001 * w)

        print("[DDPM] Training complete.")
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
    #      CNN ENCODER (PATTERN RECOGNITION SUR LES IMAGES TH2)
    # ===============================================================
    def _build_cnn_encoder(self, latent_dim=64):
        """
        Petit CNN pour encoder les images TH2 → vecteur latent.
        """
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow requis pour CNN.")

        H, W = self.hist_shape
        inp = layers.Input(shape=(H, W, 1))

        x = layers.Conv2D(32, 3, activation="relu", padding="same")(inp)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
        x = layers.GlobalAveragePooling2D()(x)

        latent = layers.Dense(latent_dim, activation="linear")(x)

        self.cnn_encoder = keras.Model(inputs=inp, outputs=latent)
        print(f"[Hybrid] CNN encoder built, latent_dim={latent_dim}")
        return self.cnn_encoder

    # ===============================================================
    #                OPTIONAL DENOISING VIA DDPM (EMA)
    # ===============================================================
    def _denoise_images(self, imgs, batch_size=1000):
        """
        DDPM denoising en mini-batches pour éviter un OOM GPU.
        imgs : (N, 64, 64)
        """
        import tensorflow as tf
        import numpy as np

        N = imgs.shape[0]

        if not hasattr(self, "ema_model") or self.ema_model is None:
            print("[DDPM] No EMA model → no denoising applied.")
            return imgs

        print(f"[DDPM] Denoising {N} images in batches of {batch_size}...")

        # Préparation des constantes DDPM
        betas = tf.constant(self.betas, dtype=tf.float32)
        alphas = 1.0 - betas
        alphas_cumprod = tf.math.cumprod(alphas)
        T = len(betas)

        out_list = []

        # Traitement batch par batch
        for i in range(0, N, batch_size):
            batch = imgs[i:i + batch_size][..., None].astype("float32")
            x = tf.convert_to_tensor((batch - 0.5) * 2.0)

            # Light denoising : derniers 20% des steps DDPM
            for t in reversed(range(int(T * 0.8), T)):
                t_tensor = tf.fill([x.shape[0]], t)
                eps = self.ema_model(x, training=False)

                alpha_t = alphas[t]
                alpha_cum = alphas_cumprod[t]
                beta_t = betas[t]

                x = (1.0 / tf.sqrt(alpha_t)) * (
                    x - (beta_t / tf.sqrt(1.0 - alpha_cum)) * eps
                )

            # Retour dans [0,1]
            x = (x + 1.0) / 2.0
            x = tf.squeeze(tf.clip_by_value(x, 0.0, 1.0), axis=-1).numpy()

            out_list.append(x)

            print(f"[DDPM] Denoised batch {i} → {i + len(batch)}")

        return np.concatenate(out_list, axis=0)

    # ===============================================================
    #      HYBRID PR-CNN + XGBOOST (FAST VERSION — df/imgs PROVIDED)
    # ===============================================================
    def xgb_prcnn_train(
        self,
        df,
        imgs,
        test_size=0.25,
        latent_dim=64,
        use_denoising=False,
        num_boost_round=300,
        max_depth=6,
        eta=0.1,
        subsample=0.8
    ):
        print("\n================ HYBRID PR-CNN-XGB TRAINING =================")

        # 1) tabular features
        X_tree = df[self.features].to_numpy()
        y_raw  = df[self.target].to_numpy()

        # LabelEncoder
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder().fit(y_raw)
        y = self.label_encoder.transform(y_raw).astype(int)

        # 2) optional denoising
        if use_denoising and hasattr(self, "ema_model"):
            imgs = self._denoise_images(imgs)

        # 3) build CNN encoder
        print("[PR-CNN] Building CNN encoder...")
        cnn = self._build_cnn_encoder(latent_dim)
        imgs_tf = imgs[..., None].astype(np.float32)

        print("[PR-CNN] Extracting latent vectors...")
        Z_img = cnn.predict(imgs_tf, batch_size=256, verbose=1)

        # 4) final features
        X_full = np.concatenate([X_tree, Z_img], axis=1)

        # 5) train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_full, y, test_size=test_size, random_state=42
        )

        # 6) XGBoost
        print("[PR-CNN] Training XGBoost...")
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest  = xgb.DMatrix(X_test,  label=y_test)

        params = {
            'objective': 'multi:softmax',
            'num_class': len(self.label_encoder.classes_),
            'max_depth': max_depth,
            'eta': eta,
            'subsample': subsample,
            'tree_method': 'hist'
        }

        self.prcnn_model = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtest, "test")],
            verbose_eval=True
        )

        preds = self.prcnn_model.predict(dtest).astype(int)
        acc = accuracy_score(y_test, preds)

        print(f"[PR-CNN] Final accuracy = {acc:.4f}")

        self.cnn_encoder = cnn
        return self.prcnn_model, acc


    # ===============================================================
    #                 HYBRID PR-CNN + XGB — PREDICT
    # ===============================================================
    def xgb_prcnn_predict(self, df, imgs, use_denoising=False):

        if not hasattr(self, "prcnn_model"):
            raise RuntimeError("PR-CNN-XGB model not trained.")

        if use_denoising and hasattr(self, "ema_model"):
            imgs = self._denoise_images(imgs)

        # tree features
        X_tree = df[self.features].to_numpy()

        # CNN embedding
        imgs_tf = imgs[..., None].astype(np.float32)
        Z_img = self.cnn_encoder.predict(imgs_tf, batch_size=256, verbose=0)

        X_full = np.concatenate([X_tree, Z_img], axis=1)
        preds_int = self.prcnn_model.predict(xgb.DMatrix(X_full)).astype(int)

        return self.label_encoder.inverse_transform(preds_int)
        
    # ===============================================================
    #                 DDPM — BUILD MODEL + EMA
    # ===============================================================
    def build_diffusion_model(self, img_shape=(64,64,1), num_timesteps=400):
        """
        Build a minimal UNet-like DDPM model + EMA model.
        Compatible with train_ddpm() and _denoise_images().
        """
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers

        self.img_shape = img_shape
        self.num_timesteps = num_timesteps

        # ----- Simple UNet -----
        inputs = keras.Input(shape=img_shape)

        x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)

        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = layers.UpSampling2D()(x)

        outputs = layers.Conv2D(1, 3, padding="same")(x)

        # Store DDPM model
        self.ddpm_model = keras.Model(inputs, outputs, name="DDPM_UNet")

        # ----- EMA COPY -----
        self.ema_model = keras.models.clone_model(self.ddpm_model)
        self.ema_model.set_weights(self.ddpm_model.get_weights())

        print("[DDPM] Diffusion model + EMA created.")
        return self.ddpm_model, self.ema_model

     # ===============================================================
    #                 DDPM — TRAIN (NO CONDITIONAL INPUT)
    # ===============================================================
    def train_ddpm(self, imgs, epochs=1, batch_size=128):
        """
        Train DDPM model for denoising. Extremely simplified.
        imgs must be (N,64,64) or (N,64,64,1)
        """
        import tensorflow as tf
        from tensorflow import keras
        import numpy as np
        import time

        # Ensure shape (N,64,64,1)
        if imgs.ndim == 3:
            imgs = imgs[..., None]

        optimizer = keras.optimizers.Adam(1e-4)
        N = len(imgs)

        for ep in range(epochs):
            print(f"[DDPM] Epoch {ep+1}/{epochs}")
            t0 = time.time()

            # Shuffle indices for robustness
            idx = np.random.permutation(N)

            for j, i in enumerate(range(0, N, batch_size)):
                batch_idx = idx[i : i + batch_size]
                batch = imgs[batch_idx]

                noise = np.random.randn(*batch.shape).astype("float32")

                with tf.GradientTape() as tape:
                    pred = self.ddpm_model(batch, training=True)
                    loss = tf.reduce_mean((pred - noise)**2)

                grads = tape.gradient(loss, self.ddpm_model.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.ddpm_model.trainable_variables))

                # ----- EMA update -----
                for w, w_ema in zip(self.ddpm_model.weights, self.ema_model.weights):
                    w_ema.assign(0.999 * w_ema + 0.001 * w)

                # ----- Progress print -----
                if j % 20 == 0:
                    print(f"[DDPM] Step {i:6d}/{N} — Loss={float(loss):.5f}")

            print(f"[DDPM] Epoch {ep+1} finished in {time.time()-t0:.1f}s")

        print("[DDPM] Training complete.")

    # ===============================================================
    #            DDPM — SAVE + LOAD + ENSURE SCHEDULE
    # ===============================================================
    def save_ddpm(self, path=None):
        if path is None:
            path = f"{self.model_dir}/ddpm_ema.h5"
        self.ema_model.save(path)
        print(f"[DDPM] EMA saved at {path}")

    def load_ddpm(self, path=None):
        from tensorflow import keras
        import os

        if path is None:
            path = f"{self.model_dir}/ddpm_ema.h5"

        if not os.path.exists(path):
            print("[DDPM] No saved EMA model found.")
            return False

        print(f"[DDPM] Loading EMA from {path}")
        self.ema_model = keras.models.load_model(path, compile=False)
        self.ensure_ddpm_schedule()
        return True

    def ensure_ddpm_schedule(self):
        self.betas = DDPM_utils.make_beta_schedule(self.ddpm_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas).astype(np.float32)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1]).astype(np.float32)



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










