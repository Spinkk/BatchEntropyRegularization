import tensorflow as tf
from BatchEntropyRegularization import LBERegularizer

class FNN_LBE(tf.keras.Model):
    def __init__(self, n_layers, width, activation="relu", n_out=10,
                 out_act="softmax", LBE_alpha_min=0.3, LBE_alpha=0.5, LBE_beta=1,
                 LBE_strength=0.2, initializer="glorot_uniform", **kwargs):
        """
        A feedforward neural network with `n_layers`, optionally regularized with a
        layer batch entropy loss (set use_LBE=True and LBE_strength > 0.)
        """
        super(FNN_LBE, self).__init__(**kwargs)

        self.LBE_strength = LBE_strength

        self.n_layers = n_layers
        self.width = width

        #set up loss and accuracy trackers
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.loss_tracker_lbe = tf.keras.metrics.Mean(name="lbe")
        self.loss_tracker_maxlbe = tf.keras.metrics.Mean(name="max_lbe")
        self.loss_tracker_ce = tf.keras.metrics.Mean(name="ce")
        self.accuracy = tf.keras.metrics.CategoricalAccuracy(name="accuracy")

        # define regularizers for each layer
        self.lbe_regs = [
                        LBERegularizer(lbe_alpha=LBE_alpha,
                            lbe_alpha_min=LBE_alpha_min,
                            lbe_beta=LBE_beta)
                        for _ in range(n_layers)]

        self.flatten_reg = LBERegularizer(lbe_alpha=LBE_alpha,
                              lbe_alpha_min=LBE_alpha_min,
                              lbe_beta=LBE_beta)

        self.flatten = tf.keras.layers.Flatten(activity_regularizer=self.flatten_reg)

        # define layers, including the activity regularizers
        self.layer_list = [
            tf.keras.layers.Dense(width,
                activation=activation,
                kernel_initializer=initializer,
                activity_regularizer=reg)
            for reg in self.lbe_regs]

        self.dense = tf.keras.layers.Dense(n_out, activation=out_act)

    def call(self, x, training=False):
        x = self.flatten(x)
        for layer in self.layer_list:
            x = layer(x)
        return self.dense(x)

    @property
    def metrics(self):
        return [self.loss_tracker,
                self.loss_tracker_lbe,
                self.loss_tracker_maxlbe,
                self.loss_tracker_ce,
                self.accuracy]

    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset_state()

    @tf.function
    def train_step(self, data):

        inputs, targets = data
        with tf.GradientTape() as tape:
            output = self(inputs, training=True)
            ce = self.compiled_loss(targets, output)
            reg_losses = tf.convert_to_tensor(self.losses)
            be_losses = reg_losses[:,0] # get lbe loss from regularization losses
            other_regularization_losses = reg_losses[:,1:] # all other reg losses
            lbe =  tf.reduce_mean(be_losses, axis=None) * ce
            loss = ce + (self.LBE_strength * lbe) + other_regularization_losses

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # update metric states
        self.loss_tracker.update_state(loss)
        self.loss_tracker_lbe.update_state(lbe)
        self.loss_tracker_maxlbe.update_state(tf.reduce_max(self.losses)*ce)
        self.loss_tracker_ce.update_state(ce)
        self.accuracy.update_state(targets, output)

        return {metric.name : metric.result() for metric in self.metrics}

    @tf.function
    def test_step(self, data: tf.Tensor):

        inputs, targets = data
        output = self(inputs, training=False)
        ce  = self.compiled_loss(targets, output)
        reg_losses = tf.convert_to_tensor(self.losses)
        be_losses = reg_losses[:,0]
        other_regularization_losses = reg_losses[:,1:]
        lbe =  tf.reduce_mean(be_losses, axis=None) * ce
        loss = ce + (self.LBE_strength * lbe) + other_regularization_losses

        # update metric states
        self.loss_tracker.update_state(loss)
        self.loss_tracker_lbe.update_state(lbe)
        self.loss_tracker_maxlbe.update_state(tf.reduce_max(self.losses)*ce)
        self.loss_tracker_ce.update_state(ce)
        self.accuracy.update_state(targets, output)

        return {metric.name : metric.result() for metric in self.metrics}

class FNN(tf.keras.Model):
    def __init__(self, n_layers, width, n_out=10, activation="relu", out_act="softmax",
                 initializer="glorot_uniform", **kwargs):
        """
        A feedforward neural network with `n_layers` of `width` units.
        """
        super(FNN, self).__init__(**kwargs)

        self.n_layers = n_layers
        self.width = width

        #set up loss and accuracy trackers
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.accuracy = tf.keras.metrics.CategoricalAccuracy(name="accuracy")

        self.flatten = tf.keras.layers.Flatten()

        # define layers
        self.layer_list = [
            tf.keras.layers.Dense(width,
                                  activation="relu",
                                  kernel_initializer=initializer,
                                  activity_regularizer=None)
                          for _ in range(self.n_layers)
        ]

        self.dense = tf.keras.layers.Dense(n_out, activation=out_act)

    def call(self, x, training=False):
        x = self.flatten(x)
        for layer in self.layer_list:
            x = layer(x)
        return self.dense(x)

    @property
    def metrics(self):
        return [self.loss_tracker, self.accuracy]

    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset_state()

    @tf.function
    def train_step(self, data: tf.Tensor):

        inputs, targets = data

        with tf.GradientTape() as tape:
            output = self(inputs, training=True)
            ce = self.compiled_loss(targets, output) + tf.reduce_sum(self.losses)
            loss = ce

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # update metric states
        self.loss_tracker.update_state(loss)
        self.accuracy.update_state(targets, output)

        return {metric.name : metric.result() for metric in self.metrics}

    @tf.function
    def test_step(self, data: tf.Tensor):

        inputs, targets = data

        output = self(inputs, training=False)
        ce = self.compiled_loss(targets, output) + tf.reduce_sum(self.losses)
        loss = ce

        # update metric states
        self.loss_tracker.update_state(loss)
        self.accuracy.update_state(targets, output)

        return {metric.name : metric.result() for metric in self.metrics}
