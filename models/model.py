import tensorflow as tf
class LBE_Model(tf.keras.Model):
    """
    This is a version of tf.keras.Model that has useful metrics and
    the required train_step and test_step methods used for
    LBE activity regularization. It can be used just like tf.keras.Model
    with the exception that layers in the model need to have no kernel
    regularization and they need to use LBE activity regularization!

    If you wish to use other (or more) metrics, you should modify
    the code for this class accordingly.

    """
    __doc__ += tf.keras.Model.__doc__
    def __init__(self, inputs, outputs, LBE_strength=1.5, **kwargs):
        super().__init__(inputs, outputs)

        # global scaling of lbe losses (instead of beta)
        self.LBE_strength = LBE_strength

        # add metrics here
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.loss_tracker_lbe = tf.keras.metrics.Mean(name="lbe")
        self.loss_tracker_maxlbe = tf.keras.metrics.Mean(name="max_lbe")
        self.loss_tracker_ce = tf.keras.metrics.Mean(name="ce")
        self.accuracy = tf.keras.metrics.CategoricalAccuracy(name="accuracy")

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

    #@tf.function
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

    #@tf.function
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
