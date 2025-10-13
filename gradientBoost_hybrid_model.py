import  os
import math
import time
from datetime import datetime

from utilityddpm import DDPM_utils

import pickle
import uproot as ur
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from typing import List, Tuple, Union, Optional

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
        self.gbhm_model = None
        self.label_encoder = None

        self.hist_shape = hist_shape
        self.model = None
        self.ddpm_timesteps = ddpm_timesteps
        self.betas = DDPM_utils.make_beta_schedule(self.ddpm_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas,axis=0).astype(np.float32)
        self.alphas_cumprod_prev = np.append(1.0,self.alphas_cumprod[:-1]).astype(np.float32)
        self.shape = hist_shape #target image size for diffusion
        self.diffusion_model = DDPM_utils.make_unet2d(input_shape=(64,64,1))

        #Model dir
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)


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

    def xgb_train(
            self,max_depth=6,loss='mlogloss',eta=0.1,subsample=0.8,tree_method='hist',
            num_boost_round=200,test_size=0.2, 
            random_state=42
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
        X = df[self.features + ['bin_index', 'radius']]
        y = df[self.target]

        #Map labels -> 0..N-1
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        self.label_encoder = le

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
            #'num_class': len(np.unique(y)),
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

        return self.xgb_model

    def gbhm_train(
            self,n_estimators=100,learn_rate=0.1,max_depth=3,
            test_size=0.3,random_state=42
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
        X=data[self.features + ["bin_index", "radius"]]
        y=data[self.target]

        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
        else:
            # Update encoder with new labels
            new_classes = np.unique(y)
            all_classes = np.unique(np.concatenate([self.label_encoder.classes_, new_classes]))
            self.label_encoder.classes_ = all_classes
            y_encoded = self.label_encoder.transform(y)

        #Split data into training and testing sets
        X_train, X_test, y_train, y_test=train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state)

        #Initialize and train the Gradient Boosting model
        print("Training Gradient Boosting model...")
        self.gbhm_model=GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learn_rate,
            max_depth=max_depth,
            random_state=random_state
        )
        self.gbhm_model.fit(X_train,y_train)

        #Evaluate the model
        train_accuracy=accuracy_score(y_train, self.gbhm_model.predict(X_train))
        test_accuracy=accuracy_score(y_test, self.gbhm_model.predict(X_test))
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        return self.gbhm_model

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

        input_shape = (self.hist_shape, 1)
        #Create model where class embedding will be handled via class id input
        self.diffusion_model = self.ddmp.make_unet2d(imput_shape=input_shape, base_channels=32)
        #compile will be handled in train
        self.n_classes = n_classes
        return self.diffusion_model

    def q_sample(self, x_start, t, noise):
        '''
        Forward diffusion q(x_t | x_0). Uses precomputed alphas_cumprod
        x_start: (B,H,W,1) in float32
        t: int timesteps(B,)
        noise: same shape as x_start
        '''
        sqrt_alphas_cumprod = tf.sqrt(tf.constant(self.alphas_cumprod))
        sqrt_one_minus_alphas_cumprod = tf.sqrt(tf.constant(1.0 - self.alphas_cumprod))
        coef1 = extract(sqrt_alphas_cumprod, t, x_start.shape)
        coef2 = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return coef1*x_start + coef2*noise

    def train_ddpm(self, epochs=20, batch_size=1, lr=2e-4, device=None):
        '''
        Train DDPM (predict noise) on the set of images + conditioning label + aux scaler
        '''
        device = device or ("/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0")
        #imgs (N,H,W)
        imgs, labels_enc, auxs, le = DDPM_utils._load_th2_images(
            inputfiles=self._inputfiles, histogram_name=self.histogram_name,
            tree_name=self.tree_name, aux_scalar_branch = self.aux_scalar_branch,
            target = self.target,hist_shape=self.hist_shape,
            x_edges = self.x_edges, y_edges = self.y_edges
        )
        self.label_encoder = le
        N = imgs.shape[0]

        #Prepare tensors
        X = imgs[..., np.newaxis].astype(np.float32)
        y = labels_enc.astype(np.int32)
        aux = auxs.astype(np.float32)

        #build diffusion model if not built
        if self.diffusion_model is None:
            self.build_diffusion_model(n_classes=len(np.unique(y)))
        optimizer = keras.optimizers.Adam(learning_rate=lr)

        #training loop:
        T = self.ddpm_timesteps
        betas = tf.constant(self.betas)
        alphas_cumprod = tf.constant(self.alphas_cumprod)

        #Dataset
        ds = tf.data.Dataset.from_tensor_slices((X,y,aux)).shuffle(1024).batch(batch_size)

        @tf.function
        def train_step(x0, labels, aux_scalar):
            batch_size_local = tf.shape(x0)[0]
            #timesteps
            t = tf.random.uniform((batch_size_local,),minval=0, maxval=T,dtype=tf.int32)
            noise = tf.random.normal(tf.shape(x0))
            coef1 = DDPM_utils.extract(tf.sqrt(self.alphas_cumprod), t, x0.shape)
            coef2 = DDPM_utils.extract(tf.sqrt(1.0 - self.alphas_cumprod), t, x0.shape)
            x_noisy = coef1*x0 + coef2*noise

            with tf.GradientTape() as tape:
                #Model expects inputs: [imp,t,class_id,aux_scalar]
                noise_pred = self.diffusion_model([x_noisy, t, labels, aux_scalar],training=True)
                loss = tf.reduce_mean(tf.square(noise - noise_pred))
            grads = tape.gradient(loss, self.diffusion_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.diffusion_model.trainable_variables))
            
            return loss
        
        #Training loop
        for epoch in range(epochs):
            t0 = time.time()
            avg_loss = 0.0
            steps = 0
            for batch_x, batch_y, batch_aux in ds:
                loss_val = train_step(batch_x, batch_y, batch_aux)
                avg_loss += float(loss_val)
                steps += 1
            avg_loss /= max(1,steps)
            print(f'[DDPM] Epoch {epoch+1}/{epochs} - loss: {avg_loss:.6f} - time: {time.time()-t0:.1f}s')
        self.diffusion_compile = True
        print('DDPM training is complete...')

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
            raise RuntimeError('No trained diffusion model available. Run train_ddpm firs or load one...')

        device = device or ('/GPU:0' if tf.config.list_physicsl_devices('GPU') else '/CPU:0')
        T = self.ddpm_timesteps
        betas = self.betas
        alphas = self.alphas
        alphas_cumprod = self.alphas_cumprod
        alphas_cumprod_prev = self.alphas_cumprod_prev

        #Prepare conditioning
        if class_ids is None:
            class_ids = [0]*n_samples
        if aux_scalars is None:
            aux = np.zeros((n_samples,1),dtype=np.float32)

        #If original labels were encoded, map provided class IDs (if they are raw pdgIDs)
        #If user passed pdgID values, convert via label_encoder
        cls_ids_arr = np.array(class_ids, dtype=float32)
        if self.label_encoder is not None:
            #if class_ids appear to be raw pdg (not encoded), attempt inverse map
            #try mapping provided values if they exist in encoder classes
            if np.any([val in self.label_encoder.classes_ for val in class_ids]):
                cls_ids_arr = self.label_encoder.transform(class_ids)
        cls_ids_arr = cls_ids_arr.astype(np.int32)

        #Start from pure noise
        shape = (n_samples, *self.hist_shape, 1)
        x_t = tf.random.normal(shape, dtype=tf.float32)

        for t_idx in reverse(range(T)):
            t_step = tf.fill((n_samples), tf.cast(t_idx,tf.int32))
            #predict noise
            pred_noise = self.diffusion_model([x_t, t_steps, cls_ids_arr, aux_scalars],training=False)
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
                mean = coef1*x0_pred + coef1*x_t
                #Sample noise
                noise = tf.random.normal(shape, dtype=tf.float32)
                var = betas[t_idx]*(1.0 - alphas_cumprod_prev[t_idx])/(1.0 - alphas_cumprod[t_idx])
                x_t = mean + tf.sqrt(var)*noise
            else:
                x_t = x0_pred

        #(n_samples,H,W)
        out = x_t.numpy().squeeze(-1)
        #rescale back to [0,1] if necessary - model on [0,1]
        out = np.clip(out, 0.0, 1.0)
        return out

    def save_ddpm(self, path='saved_models'):
        if self.diffusion_model in None:
            raise RuntimeError('No diffusion model to save...')
        os.makedirs(path, exists_ok=True)
        #Save weights + important arrays
        self.diffusion_model.save_weights(os.path.join(path, 'ddpm_ckpt_unet_weights.h5'))
        np.save(os.path.join(path,'betas.npy'),self.betas)
        np.save(os.path.join(path,'alphas_cumprod.npy'),self.alphas_cumprod)
        #Save label encoder
        if self.label_encoder is not None:
            joblib.dump(self.label_encoder,os.path.join(path,'label_encoder.pkl'))
        print(f'Saved DDPM checkpoint to {path}')

    def load_ddpm(self, path='saved_models'):
        #load arrays
        self.betas = np.load(os.path.join(path,'ddpm_ckpt_betas.npy'))
        self.alphas_cumprod = np.load(os.path.join(path,'ddpm_ckpt_alphas_cumprod.np'))
        #rebuild model with hist_shape and load weights
        self.build_diffusion_model(n_classes=len(self.label_encoder.classes_) if self.label_encoder is not None else None)
        self.diffusion_model.load_weights(os.path.join(path,'unet_weights.h5'))
        #load label encoder if present
        le_path = os.path.join(path,'ddpm_ckpt_label_encoder.pkl')
        if os.path.exists(le_path):
            self.laebl_encoder = joblib.load(le_path)
        self.diffusion_compiled = True
        print(f'Loaded DDPM checkpoint from {path}')

    def xgb_predict(self, new_data):
        """
        Makes predictions on new data.

        Args:
            new_data (pd.DataFrame): DataFrame containing the same features as the training data.

        Returns:
            np.ndarray: The predicted class labels.
        """
        if self.xgb_model is None:
            raise RuntimeError("Model has not been trained yet.")

        # Ensure the new data has the same columns as the training data
        missing_cols=set(self.features) - set(new_data.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in new data: {missing_cols}")

        return self.xgb_model.predict(new_data[self.features])

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
        laebls = self.label_encoder.inverse_transform(preds.astype(int))

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
        self.gbhm_model.n_estimators += n_estimators_add
        self.gbhm_model.fit(X_train, y_train)

        print("Fine-tuning complete.")
        test_acc = accuracy_score(y_test, self.gbhm_model.predict(X_test))
        print(f"New Test Accuracy: {test_acc:.4f}")

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
    def save(self, path_prefix="saved_models/hybrid_model"):
        """Save model and encoder to disk."""
        os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
        if self.model:
            joblib.dump(self.model, f"{path_prefix}_gbhm.pkl")
        if self.xgb_model:
            self.xgb_model.save_model(f"{path_prefix}_xgb.json")
        joblib.dump(self.label_encoder, f"{path_prefix}_encoder.pkl")
        print(f"Saved model and encoder --> {path_prefix}_*")

    def load(self, path_prefix="saved_models/hybrid_model"):
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
