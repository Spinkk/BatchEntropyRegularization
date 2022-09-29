import tensorflow as tf
from models.model import LBE_Model
from layers.layers import *

def get_dense_model(input_shape,
                    n_layers,
                    width,
                    n_out,
                    out_act=None,
                    activation="relu",
                    with_lbe=True,
                    LBE_alpha=0.5,
                    LBE_alpha_min=0.5,
                    LBE_beta=1.0,
                    LBE_strength=1.5,
                    initializer="glorot_uniform"):
    if with_lbe:
        Dense_Layer = Dense_LBE
        Model = LBE_Model
    else:
        Dense_Layer = tf.keras.layers.Dense
        Model = tf.keras.Model
    inputs = tf.keras.Input(input_shape)
    flattened_inputs = tf.keras.layers.Flatten()(inputs)

    if with_lbe:
        x = Dense_Layer(width, activation=activation, kernel_initializer=initializer,
                        LBE_alpha=LBE_alpha,
                        LBE_alpha_min=0., LBE_beta=1.0)(flattened_inputs)
        for _ in range(n_layers):
            x = Dense_Layer(width, activation=activation,
                            kernel_initializer=initializer, LBE_alpha=LBE_alpha,
                            LBE_alpha_min=0., LBE_beta=1.0)(x)
    else:
        x = Dense_Layer(width, activation=activation, kernel_initializer=initializer)(flattened_inputs)
        for _ in range(n_layers):
            x = Dense_Layer(width, activation=activation, kernel_initializer=initializer)(x)

    out = tf.keras.layers.Dense(n_out)(x)

    return Model(inputs, out)

def get_conv2d_model(input_shape,
                    n_layers,
                    n_out=10,
                    out_act=None,
                    n_filters=128,
                    kernel_size=3,
                    activation="relu",
                    with_lbe=True,
                    LBE_alpha=0.5,
                    LBE_alpha_min=0.5,
                    LBE_beta=1.0,
                    LBE_strength=1.5):
    if with_lbe:
        Conv2D = Conv2D_LBE
        Model = LBE_Model

    else:
        Conv2D = tf.keras.layers.Conv2D
        Model = tf.keras.Model

    inputs = tf.keras.Input(input_shape)
    if with_lbe:
        x = Conv2D(filters=n_filters,
                        kernel_size=kernel_size,
                        padding="same",
                        activation=activation,
                        LBE_alpha=LBE_alpha,
                        LBE_alpha_min=0.,
                        LBE_beta=1.0)(inputs)
        # add a lot more layers to test if lbe makes it trainable
        for i in range(n_layers-1):
            x = Conv2D(filters=n_filters,
                         kernel_size=kernel_size,
                         padding="same",
                         activation=activation,
                         LBE_alpha=LBE_alpha,
                         LBE_alpha_min=0.,
                         LBE_beta=1.0)(x)
    else:
        x = Conv2D(filters=n_filters,
                        kernel_size=kernel_size,
                        padding="same",
                        activation=activation)(inputs)
        # add a lot more layers to test if lbe makes it trainable
        for i in range(n_layers-1):
            x = Conv2D(filters=n_filters,
                         kernel_size=kernel_size,
                         padding="same",
                         activation=activation)(x)
    x = tf.keras.layers.GlobalMaxPooling2D()(x)

    out = tf.keras.layers.Dense(n_out)(x)
    model = Model(inputs=inputs, outputs=out)
    return model
