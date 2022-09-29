import tensorflow as tf
from BatchEntropyRegularization import LBERegularizer

class Conv2D_LBE(tf.keras.layers.Conv2D):
    def __init__(self,
        *args,
        LBE_alpha=6.,
        LBE_alpha_min=0.5,
        LBE_beta=1.,
        **kwargs):

        if "kernel_regularizer" in kwargs.keys():
            if kwargs["kernel_regularizer"] is not None:
                raise NotImplementedError('Kernel regularizers are currently not supported.')

        if "activity_regularizer" not in kwargs.keys():
            kwargs["activity_regularizer"] = None

        self.other_activity_regularizer=kwargs["activity_regularizer"] # e.g. L1

        self.LBE= LBERegularizer(lbe_alpha=LBE_alpha,
                                    lbe_alpha_min=LBE_alpha_min,
                                    lbe_beta=LBE_beta,
                                    other_activity_regularizer=self.other_activity_regularizer)
        kwargs["activity_regularizer"] = self.LBE
        super().__init__(*args, **kwargs)

class Dense_LBE(tf.keras.layers.Dense):
    def __init__(self,
        *args,
        LBE_alpha=6.,
        LBE_alpha_min=0.5,
        LBE_beta=1.,
        **kwargs):

        if "kernel_regularizer" in kwargs.keys():
            if kwargs["kernel_regularizer"] is not None:
                raise NotImplementedError('Kernel regularizers are currently not supported.')

        if "activity_regularizer" not in kwargs.keys():
            kwargs["activity_regularizer"] = None

        self.other_activity_regularizer=kwargs["activity_regularizer"] # e.g. L1
        self.LBE= LBERegularizer(lbe_alpha=LBE_alpha,
                                    lbe_alpha_min=LBE_alpha_min,
                                    lbe_beta=LBE_beta,
                                    other_activity_regularizer=self.other_activity_regularizer)
        kwargs["activity_regularizer"] = self.LBE
        super().__init__(*args, **kwargs)


# ... (more layers in the same schema as above should go here)
