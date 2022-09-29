import json
import os
import tensorflow as tf
from datasets.datasets import get_mnist, get_cifar10, get_cifar100
from models.functional_api_models import get_dense_model, get_conv2d_model
from experiments.training_callbacks import *
from experiments.plot_functions import plot_logs_classification

# Experiment parameters
EXPERIMENT_NAME="FunctionalAPI_1"

# Dataset and task parameters
dataset_function=get_mnist
input_shape = (28,28,1)
loss_function=tf.keras.losses.CategoricalCrossentropy(from_logits=True)
N_OUT=10
OUTPUT_ACTIVATION=None

# Train a baseline model without LBE regularization?
train_baseline=True

# Architecture parameters
MODEL_NO_LBE=get_dense_model
MODEL=get_dense_model
N_LAYERS = 10
WIDTH = 256
ACTIVATIONS="relu"

# LBE-Regularization parameters
LBE_STRENGTH = 0.2*7.5 # global beta parameter (scaling average of lbe errors)
LBE_BETA = 1  # local beta parameter
LBE_ALPHA = 1.5*2*2 # initial layer batch entropy target value
LBE_ALPHA_MIN = 0.5 # minimal layer batch entropy target value

# Optimization related parameters
INITIALIZER = "glorot_uniform"
N_EPOCHS = 4
BATCH_SIZE = 512
LEARNING_RATE = 1e-4
ADAM_BETA1 = 0.95
ADAM_EPSILON = 1e-7
GLOBAL_CLIPNORM = 1.0 # not used in the original paper (set to False to disable)

# Callbacks (e.g. what to log, when to adapt hyperparameters during training)
LR_PATIENCE = 10
lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    patience=LR_PATIENCE,
    factor=0.5,
    min_lr = 1e-7)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=f"experiments/results/{EXPERIMENT_NAME}/model_weights",
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)


logging_callback = tf.keras.callbacks.TensorBoard(log_dir=f"./logs/{EXPERIMENT_NAME}")
CALLBACKS = [logging_callback, lr_reducer, model_checkpoint_callback]

# Log the setup to a config json
config={"N_LAYERS": N_LAYERS,
        "WIDTH": WIDTH,
        "ACTIVATIONS": ACTIVATIONS,
        "LBE_STRENGTH":LBE_STRENGTH,
        "LBE_ALPHA": LBE_ALPHA,
        "LBE_ALPHA_MIN": LBE_ALPHA_MIN,
        "INITIALIZER": INITIALIZER,
        "BATCH_SIZE": BATCH_SIZE,
        "LEARNING_RATE": LEARNING_RATE,
        "patience": LR_PATIENCE,
        "ADAM_BETA1": ADAM_BETA1,
        "ADAM_EPSILON": ADAM_EPSILON,
        "GLOBAL_CLIPNORM": GLOBAL_CLIPNORM,}

os.makedirs(f'experiments/results/{EXPERIMENT_NAME}/',exist_ok=True)
with open(f'experiments/results/{EXPERIMENT_NAME}/config.json', 'w') as fp:
    json.dump(config, fp)

# load data
train_ds, val_ds = dataset_function(batch_size=BATCH_SIZE)

if train_baseline:
    print("Without Layer Batch Entropy Regularization:\n\n")
    model = MODEL_NO_LBE(
              input_shape=input_shape,
              n_layers=N_LAYERS,
              width=WIDTH,
              activation=ACTIVATIONS,
              n_out=N_OUT,
              out_act=OUTPUT_ACTIVATION,
              initializer=INITIALIZER)
    if GLOBAL_CLIPNORM:
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE,
            beta_1=ADAM_BETA1,
            epsilon=ADAM_EPSILON,
            global_clipnorm=GLOBAL_CLIPNORM)
    else:
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE,
            beta_1=ADAM_BETA1)

    model.compile(optimizer=optimizer, loss=loss_function)

    without_lbe_history = model.fit(train_ds,
        validation_data=val_ds,
        batch_size=BATCH_SIZE,
        epochs=N_EPOCHS,
        callbacks=CALLBACKS)
else:
    without_lbe_history=None

# Train the same network with LBE regularization
print("\nWith Layer Batch Entropy Regularization:\n\n")
model = MODEL(input_shape=input_shape,
          n_layers=N_LAYERS,
          width=WIDTH,
          activation=ACTIVATIONS,
          n_out=N_OUT,
          out_act=OUTPUT_ACTIVATION,
          LBE_alpha_min=LBE_ALPHA_MIN,
          LBE_alpha=LBE_ALPHA,
          LBE_beta=LBE_BETA,
          LBE_strength=LBE_STRENGTH,
          initializer=INITIALIZER)

if GLOBAL_CLIPNORM:
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE,
        beta_1=ADAM_BETA1,
        epsilon=ADAM_EPSILON,
        global_clipnorm=GLOBAL_CLIPNORM)
else:
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=ADAM_BETA1)

model.compile(optimizer=optimizer, loss=loss_function)

with_lbe_history = model.fit(train_ds,
    validation_data=val_ds,
    batch_size=BATCH_SIZE,
    epochs=N_EPOCHS,
    callbacks=CALLBACKS)

# save accuracy, loss, and LBE visualizations
plot_logs_classification(EXPERIMENT_NAME, with_lbe_history, without_lbe_history)
