"""
BiLSTM + Bahdanau Attention Model
===================================
Binary classification: Alzheimer's Disease (1) vs Healthy (0)

Architecture:
  Input(T, F)
    → BiLSTM(128) + Dropout
    → BiLSTM(64)  + Dropout
    → Bahdanau Attention
    → Dense(128, ReLU) → BatchNorm → Dropout
    → Dense(64,  ReLU)
    → Dense(2,   Softmax)
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Bidirectional,
    Dropout, BatchNormalization, Layer,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K


# ─────────────────────────────────────────────
# Bahdanau Attention
# ─────────────────────────────────────────────

class BahdanauAttention(Layer):
    """
    Additive (Bahdanau) soft-attention over BiLSTM hidden states.

    For each time step t, computes an importance score e_t:
        e_t  = V · tanh(W · h_t)
        α_t  = softmax(e_t)          ← attention weight
        c    = Σ α_t · h_t           ← context vector

    The context vector c is a weighted sum of all hidden states,
    where the model has learned to focus on clinically relevant
    time steps (e.g., visits showing memory or cognitive decline).
    """

    def __init__(self, units: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W = Dense(units, use_bias=False, name="attn_W")
        self.V = Dense(1,     use_bias=False, name="attn_V")

    def call(self, hidden_states):
        # hidden_states: (batch, T, hidden_dim)
        score           = self.V(tf.nn.tanh(self.W(hidden_states)))  # (batch, T, 1)
        attention_weights = tf.nn.softmax(score, axis=1)              # (batch, T, 1)
        context         = attention_weights * hidden_states            # (batch, T, hidden)
        context         = tf.reduce_sum(context, axis=1)              # (batch, hidden)
        attn_squeezed   = tf.squeeze(attention_weights, axis=-1)      # (batch, T)
        return context, attn_squeezed

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units})
        return cfg


# ─────────────────────────────────────────────
# Model Builder
# ─────────────────────────────────────────────

def build_model(
    sequence_length : int   = 10,
    n_features      : int   = 32,
    n_classes       : int   = 2,
    lstm_units_1    : int   = 128,
    lstm_units_2    : int   = 64,
    attention_units : int   = 64,
    dense_units     : int   = 128,
    dropout_rate    : float = 0.3,
    rec_dropout     : float = 0.2,
    l2_reg          : float = 1e-4,
    learning_rate   : float = 1e-3,
) -> Model:
    """
    Build and compile the BiLSTM + Attention binary classifier.

    Parameters
    ----------
    sequence_length  : T — number of (pseudo) time steps per patient
    n_features       : Number of input clinical features
    n_classes        : 2 (AD vs Healthy)
    lstm_units_1/2   : Units per direction in each BiLSTM layer
    attention_units  : Hidden size of attention scoring network
    dense_units      : Size of first dense head layer
    dropout_rate     : Feed-forward dropout
    rec_dropout      : Recurrent (intra-LSTM) dropout
    l2_reg           : L2 weight regularisation coefficient
    learning_rate    : Adam initial LR

    Returns
    -------
    Compiled Keras Model
    """

    inputs = Input(shape=(sequence_length, n_features), name="clinical_input")

    # ── BiLSTM Layer 1 ─────────────────────────────────────
    # return_sequences=True passes all T hidden states to layer 2
    x = Bidirectional(
        LSTM(
            lstm_units_1,
            return_sequences=True,
            dropout=dropout_rate,
            recurrent_dropout=rec_dropout,
            kernel_regularizer=l2(l2_reg),
            name="lstm1",
        ),
        name="bilstm_1",
    )(inputs)
    x = Dropout(dropout_rate, name="drop_1")(x)

    # ── BiLSTM Layer 2 ─────────────────────────────────────
    x = Bidirectional(
        LSTM(
            lstm_units_2,
            return_sequences=True,     # still need all steps for attention
            dropout=dropout_rate,
            recurrent_dropout=rec_dropout,
            kernel_regularizer=l2(l2_reg),
            name="lstm2",
        ),
        name="bilstm_2",
    )(x)
    x = Dropout(dropout_rate, name="drop_2")(x)

    # ── Bahdanau Attention ──────────────────────────────────
    attention = BahdanauAttention(units=attention_units, name="attention")
    context_vector, attention_weights = attention(x)

    # ── Classification Head ─────────────────────────────────
    x = Dense(
        dense_units, activation="relu",
        kernel_regularizer=l2(l2_reg),
        name="dense_1",
    )(context_vector)
    x = BatchNormalization(name="bn_1")(x)
    x = Dropout(0.4, name="drop_head")(x)

    x = Dense(
        64, activation="relu",
        kernel_regularizer=l2(l2_reg),
        name="dense_2",
    )(x)

    outputs = Dense(n_classes, activation="softmax", name="output")(x)

    # ── Compile ─────────────────────────────────────────────
    model = Model(inputs=inputs, outputs=outputs, name="BiLSTM_Attn_AD")

    model.compile(
        optimizer=Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    return model


def build_interpretable_model(base_model: Model) -> Model:
    """
    Returns a model that outputs BOTH predictions AND attention weights.
    Use this at inference time to explain which time steps mattered most.
    """
    attn_layer = base_model.get_layer("attention")
    context, weights = attn_layer.output
    return Model(
        inputs  = base_model.input,
        outputs = [base_model.output, weights],
        name    = "BiLSTM_Attn_AD_Interpretable",
    )


if __name__ == "__main__":
    m = build_model()
    m.summary()
    total = sum(p.numpy() for p in m.trainable_weights if hasattr(p, 'numpy'))
    print(f"\nTotal trainable parameters: {m.count_params():,}")
