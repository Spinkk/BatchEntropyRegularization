import tensorflow as tf
import tensorflow_datasets as tfds

def get_mnist(batch_size):
    """
    Load and prepare MNIST as a tensorflow dataset.
    Returns a train and a validation dataset.

    Args:
    batch_size (int)
    """
    train_ds, val_ds = tfds.load('mnist', split=['train', 'test'], shuffle_files=True)

    one_hot = lambda x: tf.one_hot(x, 10)

    map_func = lambda x,y: (tf.cast(
        tf.expand_dims(x, -1), dtype=tf.float32)/255.,
                            tf.cast(one_hot(y),tf.float32))

    map_func_2 = lambda x: (x["image"],x["label"])

    train_ds = train_ds.map(map_func_2).map(map_func)
    val_ds   = val_ds.map(map_func_2).map(map_func)

    train_ds = train_ds.shuffle(1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds   = val_ds.shuffle(1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return (train_ds, val_ds)
