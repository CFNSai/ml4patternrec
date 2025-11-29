import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, Model, Sequential
from tensorflow.keras.layers import (
    Conv2D, Conv2DTranspose, Dense, Dropout, GroupNormalization, Activation,
    UpSampling2D, GlobalAveragePooling2D, Multiply, Add, Input
)
from typing import List, Tuple, Union, Optional

class GroupNorm(layers.Layer):
    """Custom Group Normalization (as tfa.layers.GroupNormalization replacement)."""
    def __init__(self, groups: int = 8, epsilon: float = 1e-5, **kwargs):
        super().__init__(**kwargs)
        self.groups = groups
        self.epsilon = epsilon

    def build(self, input_shape):
        channels = int(input_shape[-1])
        self.gamma = self.add_weight(name="gamma", shape=(channels,), initializer="ones", trainable=True)
        self.beta = self.add_weight(name="beta", shape=(channels,), initializer="zeros", trainable=True)
        super().build(input_shape)

    def call(self, x):
        # x: (B,H,W,C)
        B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        G = tf.minimum(self.groups, C)
        x = tf.reshape(x, [B, H, W, G, C // G])
        mean, var = tf.nn.moments(x, axes=[1, 2, 4], keepdims=True)
        x = (x - mean) / tf.sqrt(var + self.epsilon)
        x = tf.reshape(x, [B, H, W, C])
        return x * self.gamma + self.beta


#########################################
# FiLM (Feature-wise linear modulation)
#########################################
class FiLM(layers.Layer):
    """Feature-wise Linear Modulation using embeddings (time/class)."""
    def __init__(self, channels):
        super().__init__()
        self.scale = Dense(channels, name="film_scale")
        self.shift = Dense(channels, name="film_shift")

    def call(self, x, cond):
        gamma = self.scale(cond)
        beta = self.shift(cond)
        gamma = tf.reshape(gamma, [-1, 1, 1, gamma.shape[-1]])
        beta = tf.reshape(beta, [-1, 1, 1, beta.shape[-1]])
        return x * (1 + gamma) + beta

############################################
# Small Transformer block (for bottleneck)
############################################
class TransformerBlock(layers.Layer):
    """
    A fully build-safe Transformer block for image/UNet applications.
    """
    def __init__(self, embed_dim, num_heads, ff_dim, groups_gn=8, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        #--------- hyperparameters --------
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.ff_dim = int(ff_dim)
        self.groups_gn = int(groups_gn)
        self.dropout_rate = float(dropout)

        #--------- sublayers (not built yet) --------
        self.att = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=max(1,self.embed_dim // max(1,self.num_heads)),
            name='mha'
        )

        # Projection for cond tokens so they match the model embedding dim
        self.cond_proj = Dense(self.embed_dim, activation='linear', name='cond_proj_for_mha')
        # Norm layers (will choose at runtime)
        self.ln1 = layers.LayerNormalization(epsilon=1e-6, name="ln1")
        self.ln2 = layers.LayerNormalization(epsilon=1e-6, name="ln2")
        #self.gn1 = GroupNorm(groups=self.groups_gn)
        #self.gn2 = GroupNorm(groups=self.groups_gn)

        # Feedforward
        self.ffn = Sequential([
            Dense(ff_dim, activation="gelu"),
            Dense(embed_dim)
        ])

        self.dropout1 = Dropout(self.dropout_rate)
        self.dropout2 = Dropout(self.dropout_rate)

        # Flags set in build()
        self._built_sublayers = None

    #-----------------------------------------------------------
    #      BUILD: correct MHA.build() behavior
    #-----------------------------------------------------------
    def build(self, input_shape):
        """
        Supports both (B,T,C) and (B,H,W,C).
        Correctly initializes MHA with necessary query/value shapes.
        """

        if not self._built_sublayers:
            # Dummy input: (1, tokens, embed_dim)
            dummy = tf.zeros((1, 16, self.embed_dim), dtype=tf.float32)
        _ = self.att(dummy, dummy, dummy)
        _ = self.ffn(dummy)
        _ = self.ln1(dummy)
        _ = self.ln2(dummy)

        super().build(input_shape)

    #-----------------------------------------------------------
    #    CALL: automatic spatial flattening + GroupNorm
    #-----------------------------------------------------------
    def call(self, x, cond_tokens: Optional[tf.Tensor] = None, *, training=False):
        is_spatial = (tf.rank(x) == 4)

        # Flatten if spatial
        if is_spatial:
            B, H, W, C = tf.unstack(tf.shape(x))
            x = tf.reshape(x, (B, H * W, C))

        #---------- Attention -------
        attn = self.att(x, x, training=training)
        attn = self.dropout1(attn, training=training)
        # Residual + Norm (use LayerNorm for sequence tokens)
        x = self.ln1(x + attn)

        # Cross-attention if cond_tokens present
        if cond_tokens is not None:
            cond = cond_tokens
            if cond.ndim == 2:
                cond = tf.expand_dims(cond, axis=1)  # (B,1,cond_dim)
            # project cond to embed_dim keys/values
            cond_proj = self.cond_proj(cond)
            cross = self.att(query=x, value=cond_proj, key=cond_proj, training=training)
            cross = self.dropout2(cross, training=training)
            x = self.ln2(x + cross)

        #--------- FFN -------
        ffn = self.ffn(x)
        ffn = self.dropout2(ffn, training=training)
        x = self.ln2(x + ffn)

        return x


    #-----------------------------------------------------------
    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "groups_gn": self.groups_gn,
            "dropout": self.dropout_rate,
        })
        return cfg


#########################
# Time embedding helper
#########################
class TimeEmbed(layers.Layer):
    """Sinusoidal + small MLP time embedding that accepts dynamic batch dims."""
    def __init__(self, dim=128, **kwargs):
        super().__init__(**kwargs)
        self.dim = int(dim)
        self.dense1 = layers.Dense(self.dim, activation="swish")
        self.dense2 = layers.Dense(self.dim, activation="swish")

    def call(self, t):
        # Ensure t has shape (B,1)
        t = tf.reshape(t, [-1, 1])           # safe in graph + eager
        t = tf.cast(t, tf.float32)

        # Sinusoidal embedding (vectorized, numeric-stable)
        half = self.dim // 2
        freqs = tf.exp(-np.log(10000.0) * (tf.range(0, half, dtype=np.float32) / float(half)))
        freqs = tf.reshape(freqs, (1, -1))  # (1, half)
        args = t * freqs  # broadcast (B, half)
        sincos = tf.concat([tf.sin(args), tf.cos(args)], axis=-1)  # (B, dim or dim-1)
        if self.dim % 2 == 1:
            # pad to dim
            sincos = tf.pad(sincos, [[0, 0], [0, 1]])
        x = self.dense1(sincos)
        x = self.dense2(x)
        return x

##################
# Small AuxEmbed
##################
class AuxEmbed(layers.Layer):
    """Small MLP for scalar auxiliary condition (pT/energy)."""
    def __init__(self, dim=32, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = layers.Dense(dim, activation="swish")
        self.dense2 = layers.Dense(dim, activation="swish")

    def call(self, aux):
        aux = tf.reshape(aux, [-1, 1])   # (B,1)
        aux = tf.cast(aux, tf.float32)
        x = self.dense1(aux)
        x = self.dense2(x)
        return x  # (B, dim)

####################################################################
# CNNEncoder: Keras encoder mapping HxW (or HxWx1) -> embedding
####################################################################
class CNNEncoder(tf.keras.Model):
    """
    Lightweight CNN encoder which maps (H,W[,1]) -> embedding vector.
    Designed for joint training with the DDPM UNet.
    """
    def __init__(self, embedding_dim=128, name="cnn_encoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.embedding_dim = embedding_dim
        self.conv1 = layers.Conv2D(32, 3, padding='same', activation='relu')
        self.pool1 = layers.MaxPool2D(2)
        self.conv2 = layers.Conv2D(64, 3, padding='same', activation='relu')
        self.pool2 = layers.MaxPool2D(2)
        self.conv3 = layers.Conv2D(128, 3, padding='same', activation='relu')
        self.pool3 = layers.MaxPool2D(2)
        self.gap = layers.GlobalAveragePooling2D()
        self.dense = layers.Dense(self.embedding_dim, activation=None, name='embed_dense')
        self.norm = layers.LayerNormalization()

    def call(self, x, training=False):
        # Accept (N,H,W) or (N,H,W,1)
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        if x.ndim == 3:
            x = tf.expand_dims(x, axis=-1)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.gap(x)
        x = self.dense(x)
        x = self.norm(x)
        return x

    def build_graph(self, input_shape=(None,64,64,1)):
        x = tf.keras.Input(shape=input_shape[1:])
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

##############################
# Hybrid UNet-DiT Architecture
##############################
class HybridUNetDiT_TF(Model):
    def __init__(
        self,
        input_shape=(64, 64, 1),
        base_ch: int = 32,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        groups_gn: int = 8,
        time_emb: bool = True,
        class_emb: bool = True,
        aux_emb_dim: Optional[int] = None,
        num_classes: int = 4,
        self_condition: bool = False,
        ema_decay: float = 0.999,
        cond_tokens: int = 4,
        name: str = "HybridUNetDiT_TF",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.input_shape = tuple(input_shape)
        self.base_ch = base_ch
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.groups_gn = groups_gn
        self.time_emb = bool(time_emb)
        self.class_emb = bool(class_emb)
        self.aux_emb_dim = aux_emb_dim or (embed_dim // 4)
        self.num_classes = num_classes
        self.self_condition = bool(self_condition)
        self.ema_decay = ema_decay
        self.cond_tokens = int(cond_tokens)

        # embeddings
        self.time_embedding = TimeEmbed(dim=embed_dim) if self.time_emb else None
        self.aux_mlp = AuxEmbed(dim=self.aux_emb_dim)
        self.aux_proj = Dense(self.embed_dim, activation="swish")  # project aux -> embed_dim (FiLM cond)
        self.class_embedding = layers.Embedding(self.num_classes, self.embed_dim) if self.class_emb else None
        self.encoder = CNNEncoder(embedding_dim=self.embed_dim)

        # encoder conv blocks (no input_shape passed)
        in_ch = 2 if self.self_condition else 1
        self.enc1 = self._conv_block(self.base_ch, name="enc1")
        self.enc2 = self._conv_block(self.base_ch * 2, name="enc2")
        self.enc3 = self._conv_block(self.base_ch * 4, name="enc3")
        self.pool = layers.MaxPool2D(pool_size=2, name="pool")

        # FiLM modules for encoder/decoder
        self.film1 = FiLM(self.base_ch)
        self.film2 = FiLM(self.base_ch * 2)
        self.film3 = FiLM(self.base_ch * 4)

        # Bottleneck transformer
        self.bottleneck_proj = Dense(self.embed_dim)
        self.transformer_stack = [TransformerBlock(
            self.embed_dim, num_heads, self.embed_dim * 4,
            groups_gn=groups_gn) for _ in range(self.num_layers)]
        self.bottleneck_unproj = Dense(self.base_ch * 4)

        # decoder (upsampling)
        self.up1 = self._up_block(self.base_ch * 4, self.base_ch * 2)
        self.dec1 = self._conv_block(self.base_ch * 2, name="dec1")
        self.up2 = self._up_block(self.base_ch * 2, self.base_ch)
        self.dec2 = self._conv_block(self.base_ch, name="dec2")
        self.out_conv = layers.Conv2D(1, 1, activation=None, name="out_conv")

        # small bottleneck-to-cond projection
        self.bottleneck_cond_proj =  Dense(self.embed_dim * max(1, self.cond_tokens),
                                          activation="swish",
                                          name="bottleneck_cond_proj")
        self.encoder_proj = Dense(self.embed_dim, activation=None, name="encoder_proj")
    def _conv_block(self, filters, in_ch=None, name='conv_block'):
        layers_list = []
        if in_ch is not None:
            layers_list.append(Conv2D(filters, 3, padding="same", activation=None))
        else:
            layers_list.append(Conv2D(filters, 3, padding="same", activation=None))
        layers_list.append(GroupNorm(groups=self.groups_gn))
        layers_list.append(Activation("gelu"))
        layers_list.append(Conv2D(filters, 3, padding="same", activation=None))
        layers_list.append(GroupNorm(groups=self.groups_gn))
        layers_list.append(Activation("gelu"))
        return Sequential(layers_list, name=name)

    def _up_block(self, in_ch, out_ch, name='up_block'):
        return Sequential([
            UpSampling2D(size=(2, 2), interpolation="bilinear"),
            Conv2D(out_ch, 3, padding="same", activation=None),
            GroupNorm(groups=self.groups_gn),
            Activation("gelu"),
        ],name=name)

    def call(self, x, t=None, y=None, aux=None, *, training=False, use_ema=False):
        """
        x: (B,H,W,1) or (B,H,W,2) if self-conditioning used
        t: (B,) or (B,1) int timesteps
        y: (B,) class ids (int) or None
        aux: (B,1) float or None
        """
        # ---- unpack list/tuple input ----
        if isinstance(x, (list, tuple)):
            if len(x) == 4:
                x, t, y, aux = x
            else:
                raise ValueError(f"Expected 4 inputs (x, t, y, aux) but got {len(x)}")
 
        x = tf.cast(x, tf.float32)
        batch = tf.shape(x)[0]
 
        # ---- build conditioning embedding cond (time + class + aux) ----
        cond = tf.zeros((batch, self.embed_dim), dtype=tf.float32)
 
        if self.time_embedding is not None and t is not None:
            cond += self.time_embedding(t)
 
        if self.class_embedding is not None and y is not None:
            y = tf.cast(y, tf.int32)
            cond += self.class_embedding(y)
 
        if aux is not None:
            aux_emb = self.aux_mlp(aux)     # (B, aux_dim)
            cond += self.aux_proj(aux_emb)  # (B, embed_dim)
 
        # ---- CNN encoder conditioning (cross-attention tokens) ----
        enc_input = x[..., :1] if x.shape[-1] >= 1 else x
        enc_input = tf.image.resize(enc_input, (self.input_shape[0], self.input_shape[1]))
        enc_emb = self.encoder(enc_input, training=training)  # (B, embed_dim)
 
        # Ensure encoder embedding has correct dim
        if int(enc_emb.shape[-1]) != int(self.embed_dim):
            enc_emb = self.encoder_proj(enc_emb)  # linear projection to embed_dim

        # Produce cond_tokens in one projection: cond_flat shape (B, cond_tokens*embed_dim)
        cond_flat = self.bottleneck_cond_proj(enc_emb)  # (B, cond_tokens * embed_dim)

        # Reshape into (B, cond_tokens, embed_dim) — safe because bottleneck_cond_proj output matches this size
        cond_tokens = tf.reshape(cond_flat, (batch, int(self.cond_tokens), int(self.embed_dim)))

        # ---- encoder path + FiLM conditioning using cond (time/class/aux) ----
        e1 = self.enc1(x)
        e1 = self.film1(e1, cond)

        p1 = self.pool(e1)
        e2 = self.enc2(p1)
        e2 = self.film2(e2, cond)

        p2 = self.pool(e2)
        e3 = self.enc3(p2)
        e3 = self.film3(e3, cond)

        # Bottleneck: flatten spatial dims, project, apply transformer stack with cond_tokens
        shp = tf.shape(e3)
        h, w, c = shp[1], shp[2], shp[3]
        seq = tf.reshape(e3, (batch, h * w, c))
        seq = self.bottleneck_proj(seq)  # (B, hw, embed_dim)

        # Pass cond_tokens into each transformer block (blocks expect signature (x, cond_tokens, *, training=False))
        for block in self.transformer_stack:
            seq = block(seq, cond_tokens, training=training)

        seq = self.bottleneck_unproj(seq)
        z = tf.reshape(seq, (batch, h, w, c))

        # Decoder + skip connections
        d1 = self.up1(z)
        d1 = tf.concat([d1, e2], axis=-1)
        d1 = self.dec1(d1)

        d2 = self.up2(d1)
        d2 = tf.concat([d2, e1], axis=-1)
        d2 = self.dec2(d2)

        out = self.out_conv(d2)
        return out

    # -------------------------
    # Encoder save/load helpers
    # -------------------------
    def save_encoder(self, path):
        os.makedirs(path, exist_ok=True)
        enc_path = os.path.join(path, 'cnn_encoder')
        try:
            self.encoder.save(enc_path, include_optimizer=False, save_format='tf')
        except Exception:
            # fallback: weights
            wpath = os.path.join(path, 'cnn_encoder.weights.h5')
            self.encoder.save_weights(wpath)
        # try save cond token proj and any small modules
        try:
            self.cond_token_proj.save_weights(os.path.join(path, 'cond_token_proj.weights.h5'))
        except Exception:
            pass
        # encoder classifier (if you had one) is not part of this file
        print(f"[HybridUNetDiT_TF] Saved encoder to {path}")

    def load_encoder(self, path):
        enc_path = os.path.join(path, 'cnn_encoder')
        if os.path.exists(enc_path):
            try:
                self.encoder = tf.keras.models.load_model(enc_path, compile=False)
                print(f"[HybridUNetDiT_TF] Loaded encoder from {enc_path}")
            except Exception:
                # try weights-only
                wpath = os.path.join(path, 'cnn_encoder.weights.h5')
                if os.path.exists(wpath):
                    self.encoder = CNNEncoder(embedding_dim=self.embed_dim)
                    self.encoder.build(tf.TensorShape((None, *self.input_shape)))
                    self.encoder.load_weights(wpath)
                    print(f"[HybridUNetDiT_TF] Loaded encoder weights from {wpath}")
                else:
                    raise FileNotFoundError(f"No encoder found at {path}")
        else:
            raise FileNotFoundError(f"Encoder path does not exist: {enc_path}")

    ##################################
    #       DDPM Sampler
    ##################################
    def sample(self, n_samples, class_id=None,
               aux_val=None):
        """
        DDPM reverse sampling loop.
        Args:
            n_samples: number of generated samples
            class_id: int or None       -> if model uses class conditioning
            aux_val: float or None      -> if model uses auxiliary scalar
        Returns:
            images in [-1,1], shape (n_samples, H, W, 1)
        """
        if not hasattr(self, "betas") or not hasattr(self, "alphas_cumprod"):
            raise RuntimeError("betas / alphas_cumprod were not set on the model. "
                               "Call train_ddpm() first or assign DDPM schedule.")
 
        betas = tf.constant(self.betas, dtype=tf.float32)
        alphas = 1.0 - betas
        alphas_cumprod = tf.constant(self.alphas_cumprod, dtype=tf.float32)
 
        T = betas.shape[0]
        H, W, _ = self.input_shape
 
        # start from Gaussian noise
        x = tf.random.normal((n_samples, H, W, 1), dtype=tf.float32)
 
        # conditioning tensors
        if class_id is not None:
            y = tf.convert_to_tensor(class_id, dtype=tf.int32)

            # scalar: broadcast to (n_samples,)
            if y.shape.rank == 0:
                y = tf.fill((n_samples,), y)
  
            # shape (n_samples,1): squeeze
            elif y.shape.rank == 2 and y.shape[1] == 1:
                y = tf.squeeze(y, axis=1)
  
            # ensure (n_samples,)
            y = tf.reshape(y, (n_samples,))
        else:
            y = None
 
        if aux_val is not None:
            aux = tf.convert_to_tensor(aux_val, dtype=tf.float32)

            # If aux is scalar, broadcast to (n_samples, 1)
            if aux.shape.rank == 0:
                aux = tf.fill((n_samples, 1), aux)
 
            # If aux shaped like (n_samples, 1), keep as is
            elif aux.shape.rank == 2 and aux.shape[1] == 1:
                if aux.shape[0] != n_samples:
                    # broadcast if single value repeated needed
                    aux = tf.fill((n_samples, 1), aux[0, 0])
 
            # If aux shaped like (n_samples,), reshape to (n_samples,1)
            elif aux.shape.rank == 1:
                aux = tf.reshape(aux, (n_samples, 1))
 
            # Safety: final shape = (n_samples, 1)
            aux = tf.reshape(aux, (n_samples, 1))
        else:
            aux = None
 
        # reverse diffusion -----------------------------------------------------
        for t_inv in range(T):
            t = T - t_inv - 1
            t_tf = tf.fill((n_samples,), tf.cast(t, tf.int32))
 
            # predict noise eps_t
            eps_pred = self(x, t_tf, y, aux, training=False)
 
            # DDPM formula
            alpha_t = alphas[t]
            alpha_bar_t = alphas_cumprod[t]
            beta_t = betas[t]
 
            # removal of noise term
            x0_pred = (x - tf.sqrt(1.0 - alpha_bar_t) * eps_pred) / tf.sqrt(alpha_bar_t)
 
            # sample from q(x_{t-1} | x_t, x0_pred)
            if t > 0:
                noise = tf.random.normal(shape=tf.shape(x), dtype=tf.float32)
            else:
                noise = tf.zeros_like(x)
 
            coef1 = tf.sqrt(alphas[t])
            coef2 = (1 - alphas[t]) / tf.sqrt(1 - alphas_cumprod[t])
 
            x = coef1 * x0_pred + coef2 * eps_pred + tf.sqrt(beta_t) * noise
 
        return x  # still in [-1,1] range

    # small config helpers for saving/restore
    def get_config(self):
        base = super().get_config()
        base.update({
            "input_shape": self.input_shape,
            "base_ch": self.base_ch,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "groups_gn": self.groups_gn,
            "num_classes": self.num_classes,
            "self_condition": self.self_condition,
            "ema_decay": self.ema_decay,
        })
        return base

    @classmethod
    def from_config(cls, config):
        return cls(**config)

##################################
# EMA Wrapper
##################################
class EMA:
    """Exponential Moving Average for model parameters."""
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.ema_model = keras.models.clone_model(model)

        # Force both models to build
        dummy_in = tf.zeros((1, 64, 64, 1), dtype=tf.float32)
        dummy_t = tf.zeros((1,), dtype=tf.int32)
        dummy_y = tf.zeros((1,), dtype=tf.int32)
        dummy_aux = tf.zeros((1, 1), dtype=tf.float32)
        try:
            _ = model(dummy_in, dummy_t, dummy_y, dummy_aux, training=False)
            _ = self.ema_model(dummy_in, dummy_t, dummy_y, dummy_aux, training=False)
        except Exception as e:
            print(f"[EMA] fallback build: {e}")
            _ = model(dummy_in, training=False)
            _ = self.ema_model(dummy_in, training=False)

        # Initialize EMA weights
        for ema_var, var in zip(self.ema_model.weights, model.weights):
            ema_var.assign(var)

    @tf.function
    def update(self, model):
        """Graph-safe EMA update using assign ops."""
        for ema_var, var in zip(self.ema_model.weights, model.weights):
            ema_var.assign(self.decay * ema_var + (1.0 - self.decay) * var)
