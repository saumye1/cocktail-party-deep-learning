
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("./test data/p2", one_hot = True)

babe = mnist.test.next_batch(1)

image = babe[0]

#matplotlib.pyplot.imshow(image)

x_image = tf.reshape(image, [28, 28, 1])

#plt.imshow(x_image)
plt.show(x_image)

x_image = 256 * x_image
x_image = tf.floor(x_image)
y_image = tf.image.convert_image_dtype(x_image, tf.uint8)

im = tf.image.encode_jpeg(y_image)

sess = tf.Session()

f = open("./foo1.jpeg", "wb+")
f.write(im.eval(session = sess))
f.close()