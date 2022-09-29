import tensorflow as tf
import math as m

class LBERegularizer(tf.Module):
    """
    Batch entropy regularization (Peer et al 2022) module, to be used
    as an activity regularizer in tf.keras.layers.Layer objects within
    models created with the subclassing API.

    To equip a model with LBERegularizer, first instantiate an LBERegularizer
    for each layer, then use them as activity regularizers when
    instantiating the layers.

    This regularizer does **not** work if you don't instantiate the regularizers
    external to the layers' activity_regularizer arguments. This means that
    it also **does not work in the Sequential and Functional API** style of models,
    unless you subclass the layers that you want to use and include the regularizer
    as a submodule!

    """
    def __init__(self, lbe_alpha=0.5, lbe_alpha_min=0.3,
                lbe_beta=0.2, other_activity_regularizer=None):

        # learnable parameter (batch entropy target for the layer)
        self.lbe_alpha_p = tf.Variable(lbe_alpha, dtype=tf.float32,
                                     trainable=True, name="lbe_alpha")
        # regularization strength for that layer
        self.lbe_beta = lbe_beta
        # minimal batch entropy target for the layer
        self.lbe_alpha_min = lbe_alpha_min
        self.flatten = tf.keras.layers.Flatten()

        self.other_activity_regularizer=other_activity_regularizer

    def __call__(self, x):
        if self.other_activity_regularizer is not None:
            lbe = self.lbe(x)
            return tf.convert_to_tensor([lbe, self.other_activity_regularizer(x)])
        else:
            lbe = self.lbe(x)
            return [lbe, tf.zeros_like(lbe)]

    def get_config(self):
        return {'lbe_alpha': float(self.lbe_alpha_p),
                "lbe_alpha_min": float(self.lbe_alpha_min),
                "lbe_beta": float(self.lbe_beta),
                "other_activity_regularizer": self.other_activity_regularizer}

    def batch_entropy(self, x):
        """ Estimate the differential entropy by assuming a gaussian distribution of
            values for different samples of a mini-batch.
        """
        x = self.flatten(x)
        # compute std across batch dimension
        x_std = tf.math.reduce_std(x, axis=0)
        # compute entropy for each unit
        entropies = 0.5 * tf.math.log(m.pi * m.e * x_std**2 + 1)
        # take mean of entropy across units
        return tf.reduce_mean(entropies)

    def lbe(self, x):
        """ Estimate the squared error between the differential entropy of
            the layer and a target value that is learned but at least as
            high as ´lbe_alpha_min´. The error is scaled by a coefficient beta.
        """
        #restrict alpha parameter to be positive
        lbe_alpha_l = tf.abs(self.lbe_alpha_p)
        # compute squared error between the learned desired batch entropy level
        lbe_l = (self.batch_entropy(x) - tf.maximum(self.lbe_alpha_min, lbe_alpha_l))**2
        # scale with lbe coefficient beta
        return lbe_l * self.lbe_beta
