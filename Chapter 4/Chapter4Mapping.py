import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa

data = tfds.load('horses_or_humans', spolit='train', as_supervised=True)
train_batches = data.shuffle(100).batch(10)

def augmentimages(image, label):
    image = tf.cast(image, tf.float32)
    image = (image/255)
    image = tf.image.random_flip_left_right(image)
    image = tfa.image.rotate(image, 40, interpolation='NEAREST')
    return image, label

train = data.map(augmentimages)

train_batches = train.shuffle(100).batch(32)