# +
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
    it also **does not work in the Sequential and Functional API** style of models!
    
    It works only in models that rely solely on the subclassing API.
    
    Example (Subclassing API):
    
    >>> class FNN(tf.keras.Model):
    >>>     def __init__(self, n_layers, **kwargs):
    >>>         super(FNN, self).__init__(**kwargs)
    >>>         
    >>>         # define metrics to keep track of losses and accuracy
    >>>         self.loss_tracker    = tf.keras.metrics.Mean(name="loss")
    >>>         self.loss_tracker_lbe = tf.keras.metrics.Mean(name="lbe")
    >>>         self.loss_tracker_ce = tf.keras.metrics.Mean(name="ce")
    >>>         self.accuracy = tf.keras.metrics.CategoricalAccuracy(name="accuracy")
    >>>         self.loss_function = tf.keras.losses.CategoricalCrossentropy()
    >>>                 
    >>>         # define regularizers for each layer
    >>>         self.lbe_regs = [LBERegularizer()
    >>>                         for _ in range(n_layers)]
    >>>     
    >>>         self.flatten_reg = LBERegularizer()
    >>>     
    >>>         self.flatten = tf.keras.layers.Flatten(
    >>>                                         activity_regularizer=self.flatten_reg)
    >>>         
    >>>         # define layers, using the activity regularizers
    >>>         self.layer_list = [
    >>>             tf.keras.layers.Dense(32,
    >>>                                   activation="relu",
    >>>                                   activity_regularizer=lbe_reg)
    >>>                           for lbe_reg in self.lbe_regs]
    >>>          
    >>>         self.out = tf.keras.layers.Dense(10, activation="softmax")
    >>>            
    >>>     def call(self, x, training=False):
    >>>         x = self.flatten(x)
    >>>         for layer in self.layer_list:
    >>>             x = layer(x)
    >>>         return self.out(x)
    >>>     
    >>>     @property
    >>>     def metrics(self):
    >>>         return [self.loss_tracker, 
    >>>                 self.loss_tracker_lbe, 
    >>>                 self.loss_tracker_ce, 
    >>>                 self.accuracy]
    >>>     
    >>>     def reset_metrics(self):
    >>>         for metric in self.metrics:
    >>>             metric.reset_state()
    >>>     
    >>>     @tf.function
    >>>     def train_step(self, data: tf.Tensor):
    >>>     
    >>>         inputs, targets = data
    >>>     
    >>>         with tf.GradientTape() as tape:
    >>>             output = self(inputs, training=True)
    >>>             ce = self.loss_function(targets, output)
    >>>             lbe = tf.reduce_sum(self.losses, axis=None) * ce
    >>>             loss = ce + lbe
    >>>     
    >>>         grads = tape.gradient(loss, self.trainable_variables)
    >>>     
    >>>         self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
    >>>     
    >>>         # update metric states
    >>>         self.loss_tracker.update_state(loss)
    >>>         self.loss_tracker_lbe.update_state(lbe)
    >>>         self.loss_tracker_ce.update_state(ce)
    >>>         self.accuracy.update_state(targets, output)
    >>>     
    >>>         return {"loss" : self.loss_tracker.result(),
    >>>                 "lbe"  : self.loss_tracker_lbe.result(),
    >>>                 "ce"   : self.loss_tracker_ce.result(),
    >>>                 "accuracy": self.accuracy.result()}
    >>>     
    >>>     @tf.function
    >>>     def test_step(self, data: tf.Tensor):
    >>>     
    >>>         inputs, targets = data
    >>>     
    >>>         output = self(inputs, training=False)
    >>>         ce = self.loss_function(targets, output)
    >>>         lbe = tf.reduce_sum(self.losses, axis=None) * ce
    >>>         loss = ce + lbe
    >>>     
    >>>         # update metric states
    >>>         self.loss_tracker.update_state(loss)
    >>>         self.loss_tracker_lbe.update_state(lbe)
    >>>         self.loss_tracker_ce.update_state(ce)
    >>>         self.accuracy.update_state(targets, output)
    >>>     
    >>>         return {"loss": self.loss_tracker.result(),
    >>>                 "lbe"  : self.loss_tracker_lbe.result(),
    >>>                 "ce"  : self.loss_tracker_ce.result(),
    >>>                 "accuracy": self.accuracy.result()}
    
    NOT WORKING (Functional API):
    >>> n_layers=400
    >>> inputs = tf.keras.Input(shape=(784,))
    >>> lbe_regularizers = [LBERegularizer() for _ in range(n_layers)]
    >>> layers = [tf.keras.layers.Dense(32, activation="relu", 
                                    activity_regularizer=lbe_reg) for lbe_reg 
                  in lbe_regularizers]
    >>> out_layer = tf.keras.layers.Dense(10, activation="softmax")
    
    >>> x = inputs
    >>> for layer in layers:
    >>>     x = layer(x)
    >>> out = out_layer(x)
    >>> model = tf.keras.Model(inputs=inputs, outputs=out)
    
    """
    def __init__(self, lbe_alpha=0.5, lbe_alpha_min=0.3,
                lbe_beta=0.2):
        
        # learnable parameter (batch entropy target for the layer)
        self.lbe_alpha_p = tf.Variable(lbe_alpha, dtype=tf.float32,
                                     trainable=True, name="lbe_alpha")
        # regularization strength for that layer
        self.lbe_beta = lbe_beta
        
        # minimal batch entropy target for the layer
        self.lbe_alpha_min = lbe_alpha_min
        
        self.flatten = tf.keras.layers.Flatten()
        
    def __call__(self, x):
        return self.lbe(x)

    def get_config(self):
        return {'lbe_alpha': float(self.lbe_alpha_p),
                "lbe_alpha_min": float(self.lbe_alpha_min),
                "lbe_beta": float(self.lbe_beta)}

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

