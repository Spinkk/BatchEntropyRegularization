import tensorflow as tf
from BatchEntropyRegularization import LBERegularizer

class FNN(tf.keras.Model):
    def __init__(self, n_layers, use_LBE=True, LBE_strength=0.2, **kwargs):
        """
        A feedforward neural network with `n_layers`, optionally regularized with a
        layer batch entropy loss (set use_LBE=True and LBE_strength > 0.)
        """
        super(FNN, self).__init__(**kwargs)

        self.use_LBE = use_LBE
        self.LBE_strength = LBE_strength

        #set up loss and accuracy trackers
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        if use_LBE:
            self.loss_tracker_lbe = tf.keras.metrics.Mean(name="lbe")
            self.loss_tracker_ce = tf.keras.metrics.Mean(name="ce")

        self.accuracy = tf.keras.metrics.CategoricalAccuracy(name="accuracy")

        self.loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

        # define regularizers for each layer
        if use_LBE:
            self.lbe_regs = [LBERegularizer(lbe_alpha=0.5,
                                            lbe_alpha_min=0.3,
                                            lbe_beta=LBE_strength)
                            for _ in range(n_layers)]

            self.flatten_reg = LBERegularizer(lbe_alpha=0.5,
                                        lbe_alpha_min=0.3,
                                        lbe_beta=LBE_strength)

            self.flatten = tf.keras.layers.Flatten(activity_regularizer=self.flatten_reg)

        else:
            self.lbe_regs = [None for _ in range(n_layers)]
            self.flatten = tf.keras.layers.Flatten()

        # define layers, including the activity regularizers
        self.layer_list = [
            tf.keras.layers.Dense(32,
                                  activation="relu",
                                  activity_regularizer=reg)
                          for reg in self.lbe_regs
        ]

        self.dense = tf.keras.layers.Dense(10, activation="softmax")

    def call(self, x, training=False):
        x = self.flatten(x)
        for layer in self.layer_list:
            x = layer(x)
        return self.dense(x)

    @property
    def metrics(self):
        if self.use_LBE:
            return [self.loss_tracker, self.loss_tracker_lbe, self.loss_tracker_ce, self.accuracy]
        else:
            return [self.loss_tracker, self.accuracy]

    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset_state()

    @tf.function
    def train_step(self, data: tf.Tensor):

        inputs, targets = data

        if self.use_LBE:
            with tf.GradientTape() as tape:
                output = self(inputs, training=True)
                ce = self.loss_function(targets, output)
                lbe = tf.reduce_sum(self.losses, axis=None) * ce
                loss = ce + lbe
        else:
            with tf.GradientTape() as tape:
                output = self(inputs, training=True)
                ce = self.loss_function(targets, output)
                loss = ce

        grads = tape.gradient(loss, self.trainable_variables)

        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # update metric states
        self.loss_tracker.update_state(loss)
        if self.use_LBE:
            self.loss_tracker_lbe.update_state(lbe)
            self.loss_tracker_ce.update_state(ce)
        self.accuracy.update_state(targets, output)

        if self.use_LBE:
            return {"loss" : self.loss_tracker.result(),
                    "lbe"  : self.loss_tracker_lbe.result(),
                    "ce"   : self.loss_tracker_ce.result(),
                    "accuracy": self.accuracy.result()}
        else:
            return {"loss" : self.loss_tracker.result(),
                    "accuracy": self.accuracy.result()}

    @tf.function
    def test_step(self, data: tf.Tensor):

        inputs, targets = data

        output = self(inputs, training=False)
        ce = self.loss_function(targets, output)
        if self.use_LBE:
            lbe = tf.reduce_sum(self.losses, axis=None) * ce
            loss = ce + lbe
        else:
            loss = ce

        # update metric states
        self.loss_tracker.update_state(loss)
        if self.use_LBE:
            self.loss_tracker_lbe.update_state(lbe)
            self.loss_tracker_ce.update_state(ce)
        self.accuracy.update_state(targets, output)

        if self.use_LBE:
            return {"loss": self.loss_tracker.result(),
                    "lbe"  : self.loss_tracker_lbe.result(),
                    "ce"  : self.loss_tracker_ce.result(),
                    "accuracy": self.accuracy.result()}
        else:
            return {"loss" : self.loss_tracker.result(),
                    "accuracy": self.accuracy.result()}
