import tensorflow as tf
import numpy as np
class ReduceTemperatureOnPlateau(tf.keras.callbacks.Callback):
    """Reduce the temperature hyperparameter of a model when a metric has stopped improving.
    This can be useful to control the relative scale of custom regularization losses across different layers.
    Preferably this kind of adjustment is learned via meta learning, but meta learning is hard to implement in tensorflow.

    This callback monitors a
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the temperature is reduced.
    Example:
    ```python
    reduce_temperature = ReduceTemperatureOnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5, min_temperature=0.001)
    model.fit(X_train, Y_train, callbacks=[reduce_temperature])
    ```
    Args:
        monitor: quantity to be monitored.
        factor: factor by which the temperature will be reduced.
          `new_temperature = temperature * factor`.
        patience: number of epochs with no improvement after which temperature
          will be reduced.
        verbose: int. 0: quiet, 1: update messages.
        mode: one of `{'auto', 'min', 'max'}`. In `'min'` mode,
          the temperature will be reduced when the
          quantity monitored has stopped decreasing; in `'max'` mode it will be
          reduced when the quantity monitored has stopped increasing; in
          `'auto'` mode, the direction is automatically inferred from the name
          of the monitored quantity.
        min_delta: threshold for measuring the new optimum, to only focus on
          significant changes.
        cooldown: number of epochs to wait before resuming normal operation
          after temperature has been reduced.
        min_temperature: lower bound on the temperature.
    """

    def __init__(
        self,
        monitor="val_loss",
        factor=0.1,
        patience=10,
        verbose=0,
        mode="auto",
        min_delta=1e-4,
        cooldown=0,
        min_temperature=0,
        **kwargs,
    ):
        super().__init__()

        self.monitor = monitor
        if factor >= 1.0:
            raise ValueError(
                f"ReduceTemperatureOnPlateau does not support "
                f"a factor >= 1.0. Got {factor}"
            )
        if "epsilon" in kwargs:
            min_delta = kwargs.pop("epsilon")
            logging.warning(
                "`epsilon` argument is deprecated and "
                "will be removed, use `min_delta` instead."
            )
        self.factor = factor
        self.min_temperature = min_temperature
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0
        self.best = 0
        self.mode = mode
        self.monitor_op = None
        self._reset()

    def _reset(self):
        """Resets wait counter and cooldown counter."""
        if self.mode not in ["auto", "min", "max"]:
            logging.warning(
                "temperature reduction mode %s is unknown, "
                "fallback to auto mode.",
                self.mode,
            )
            self.mode = "auto"
        if self.mode == "min" or (
            self.mode == "auto" and "acc" not in self.monitor
        ):
            self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0

    def on_train_begin(self, logs=None):
        self._reset()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs["temperature"] = tf.keras.backend.get_value(self.model.temperature)
        current = logs.get(self.monitor)
        if current is None:
            logging.warning(
                "temperature reduction is conditioned on metric `%s` "
                "which is not available. Available metrics are: %s",
                self.monitor,
                ",".join(list(logs.keys())),
            )

        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                self.wait += 1
                if self.wait >= self.patience:
                    old_temperature = tf.keras.backend.get_value(self.model.temperature)
                    if old_temperature > np.float32(self.min_temperature):
                        new_temperature = old_temperature * self.factor
                        new_temperature = max(new_temperature, self.min_temperature)
                        tf.keras.backend.set_value(self.model.temperature, new_temperature)
                        if self.verbose > 0:
                            io_utils.print_msg(
                                f"\nEpoch {epoch +1}: "
                                f"ReduceTemperatureOnPlateau reducing "
                                f"temperature to {new_temperature}."
                            )
                        self.cooldown_counter = self.cooldown
                        self.wait = 0

    def in_cooldown(self):
        return self.cooldown_counter > 0


class TemperatureScheduler(tf.keras.callbacks.Callback):
    """Temperature scheduler.
    At the beginning of every epoch, this callback gets the updated learning
    rate value from `schedule` function provided at `__init__`, with the current
    epoch and current temperature, and applies the updated temperature on
    the optimizer.
    Args:
      schedule: a function that takes an epoch index (integer, indexed from 0)
          and current temperature (float) as inputs and returns a new
          temperature as output (float).
      verbose: int. 0: quiet, 1: update messages.
    Example:
    >>> # This function keeps the initial temperature for the first ten epochs
    >>> # and decreases it exponentially after that.
    >>> def scheduler(epoch, temperature):
    ...   if epoch < 10:
    ...     return temperature
    ...   else:
    ...     return temperature * tf.math.exp(-0.1)
    >>>
    >>> model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
    >>> model.compile(tf.keras.optimizers.SGD(), loss='mse')
    >>> round(model.temperature.numpy(), 5)
    0.01
    >>> callback = tf.keras.callbacks.TemperatureScheduler(scheduler)
    >>> history = model.fit(np.arange(100).reshape(5, 20), np.zeros(5),
    ...                     epochs=15, callbacks=[callback], verbose=0)
    >>> round(model.temperature.numpy(), 5)
    0.00607
    """

    def __init__(self, schedule, verbose=0):
        super().__init__()
        self.schedule = schedule
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "temperature"):
            raise ValueError('Optimizer must have a "temperature" attribute.')
        try:  # new API
            temperature = float(backend.get_value(self.model.temperature))
            temperature = self.schedule(epoch, temperature)
        except TypeError:  # Support for old API for backward compatibility
            temperature = self.schedule(epoch)
        if not isinstance(temperature, (tf.Tensor, float, np.float32, np.float64)):
            raise ValueError(
                'The output of the "schedule" function '
                f"should be float. Got: {temperature}"
            )
        if isinstance(temperature, tf.Tensor) and not temperature.dtype.is_floating:
            raise ValueError(
                f"The dtype of `temperature` Tensor should be float. Got: {temperature.dtype}"
            )
        backend.set_value(self.model.temperature, backend.get_value(temperature))
        if self.verbose > 0:
            io_utils.print_msg(
                f"\nEpoch {epoch + 1}: TemperatureScheduler setting learning "
                f"rate to {temperature}."
            )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs["temperature"] = backend.get_value(self.model.temperature)

# idea: make this universally applicable to any attribute with getattr(object, name) so it can be applied to the beta parameter as well
