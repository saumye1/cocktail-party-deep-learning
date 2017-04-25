import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import librosa
import pickle
import sys

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev = 0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape = shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

x = tf.placeholder(tf.float32, [None, 78000])
y_ = tf.placeholder(tf.float32, [None, 78000])

W_conv1 = weight_variable([10, 10, 1, 100])
b_conv1 = bias_variable([100])

x_image = tf.reshape(x, [-1, 300, 260, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([10, 10, 100, 200])
b_conv2 = bias_variable([200])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([75 * 65 * 200, 100])
b_fc1 = bias_variable([100])

h_pool2_flat = tf.reshape(h_pool2, [-1, 75 * 65 * 200])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([100, 78000])
b_fc2 = bias_variable([78000])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
pred = tf.nn.sigmoid(y_conv)
#alpha = tf.reduce_sum(pred, 1) / 78000.0

alpha = 0.7
one = tf.ones((300, 260))
spectrogram = tf.reshape(pred, [-1, 300, 260])
confidence = alpha * one

ideal_binary_mask = tf.greater(spectrogram, confidence)

filename = sys.argv[1]
filename2 = sys.argv[2]

y1, sr = librosa.load(filename, offset = 0, duration = 6)
y2, sr = librosa.load(filename2, offset = 0, duration = 6)
audio = y1 + y2
spectrogram = np.abs(librosa.stft(audio))
cropped_specrogram = spectrogram[0:300, :]
padded_spectrogram = np.matrix(np.zeros((300, 260)))
padded_spectrogram[:, 0:259] = cropped_specrogram
reshaped_spectrogram = np.reshape(padded_spectrogram, (1, 78000))

ans = np.zeros((300, 260))

print "toh...ho gaya yaha tak"
with tf.Session() as sess:
	saver = tf.train.import_meta_graph('guitar_separator_CNN-2000.meta')
	sess.run(tf.global_variables_initializer())
	saver.restore(sess, tf.train.latest_checkpoint('./'))
	print('Network loaded...')
	ibm = sess.run(ideal_binary_mask, feed_dict = {x: reshaped_spectrogram, keep_prob: 1.0})
	print ibm.shape
	for i in range(300):
		for j in range(260):
			if ibm[0, i, j] == True:
				ans[i, j] = padded_spectrogram[i, j]
			else:
				ans[i, j] = 0.0
	plt.imshow(ans)
	plt.savefig("whiste_guitar_out.png", bbox_inches = "tight")

