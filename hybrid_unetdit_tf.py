import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
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
    def __init__(self, embed_dim, num_heads, ff_dim, groups_gn=8,dropout=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        # keep both norms available
        self.gn1 = GroupNorm(groups=groups_gn)
        self.gn2 = GroupNorm(groups=groups_gn)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn = Sequential([
            Dense(ff_dim, activation="gelu"),
            Dense(embed_dim)
        ])
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def call(self, x, training=False):
        attn = self.att(x, x)
        attn = self.dropout1(attn, training=training)
        x = self.norm1(x + attn)
        ffn = self.ffn(x)
        ffn = self.dropout2(ffn, training=training)
        x = self.norm2(x + ffn)
        return x

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
        name: str = "HybridUNetDiT_TF",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.input_shape_ = tuple(input_shape)
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

        # embeddings
        self.time_embedding = TimeEmbed(dim=embed_dim) if self.time_emb else None
        self.aux_mlp = AuxEmbed(dim=self.aux_emb_dim)
        self.aux_proj = Dense(self.embed_dim, activation="swish")  # project aux -> embed_dim (FiLM cond)
        self.class_embedding = layers.Embedding(self.num_classes, self.embed_dim) if self.class_emb else None

        # encoder conv blocks (no input_shape passed)
        in_ch = 2 if self.self_condition else 1
        self.enc1 = self._conv_block(self.base_ch)
        self.enc2 = self._conv_block(self.base_ch * 2)
        self.enc3 = self._conv_block(self.base_ch * 4)
        self.pool = layers.MaxPool2D(pool_size=2)

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
        self.up1 = Sequential([UpSampling2D(size=(2, 2)),
                               Conv2D(self.base_ch * 2, 3, padding="same"),
                               GroupNorm(groups_gn), Activation("gelu")])
        self.up2 = Sequential([UpSampling2D(size=(2, 2)),
                               Conv2D(self.base_ch, 3, padding="same"),
                               GroupNorm(groups_gn), Activation("gelu")])

        # final conv
        self.out_conv = Conv2D(1, 1, padding="same")

        # small bottleneck-to-cond projection
        self.bottleneck_cond_proj = Dense(self.embed_dim, activation="swish")

    def _conv_block(self, filters):
        return Sequential([
            Conv2D(filters, 3, padding='same'),
            GroupNorm(groups=self.groups_gn),
            Activation('gelu'),
            Conv2D(filters, 3, padding='same'),
            GroupNorm(groups=self.groups_gn),
            Activation('gelu'),
        ])

    def call(self, x, t=None, y=None, aux_scalar=None, training=False, use_ema=False):
        """
        x: (B,H,W,1) or (B,H,W,2) if self-conditioning used
        t: (B,) or (B,1) int timesteps
        y: (B,) class ids (int) or None
        aux_scalar: (B,1) float or None
        """
        batch = tf.shape(x)[0]

        # build conditioning vector cond: (B, embed_dim)
        cond = tf.zeros((batch, self.embed_dim), dtype=tf.float32)

        if self.time_embedding is not None and t is not None:
            cond += self.time_embedding(t)  # ensures (B,embed_dim)

        if self.class_embedding is not None and y is not None:
            y = tf.cast(y, tf.int32)
            c_emb = self.class_embedding(y)  # (B, embed_dim)
            cond += c_emb

        if aux_scalar is not None:
            aux_emb = self.aux_mlp(aux_scalar)  # (B, aux_dim)
            aux_emb = self.aux_proj(aux_emb)    # project to embed_dim
            cond += aux_emb

        # broadcast cond to spatial dims for FiLM where needed
        cond_proj = self.bottleneck_cond_proj(cond)  # (B, embed_dim)

        # encoder
        e1 = self.enc1(x)
        e1 = self.film1(e1, cond_proj)
        e2 = self.enc2(self.pool(e1))
        e2 = self.film2(e2, cond_proj)
        e3 = self.enc3(self.pool(e2))
        e3 = self.film3(e3, cond_proj)

        # bottleneck transformer. Flatten spatial dims
        shp = tf.shape(e3)
        h, w, c = shp[1], shp[2], shp[3]
        seq = tf.reshape(e3, (batch, -1, c))  # (B, h*w, c)
        seq = self.bottleneck_proj(seq)      # (B, h*w, embed_dim)
        for layer in self.transformer_stack:
            seq = layer(seq, training=training)
        seq = self.bottleneck_unproj(seq)    # (B, h*w, c)
        z = tf.reshape(seq, (batch, h, w, c))

        # decoder + skip connections
        d1 = self.up1(z)
        d1 = tf.concat([d1, e2], axis=-1)
        d2 = self.up2(d1)
        d2 = tf.concat([d2, e1], axis=-1)

        out = self.out_conv(d2)  # predicted noise (B,H,W,1)
        return out

    # small config helpers for saving/restore
    def get_config(self):
        base = super().get_config()
        base.update({
            "input_shape": self.input_shape_,
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
