import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)    

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100

 
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))


hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),'biases':tf.Variable(tf.random_normal([n_classes]))}

l1 = tf.add(tf.matmul(x, hidden_1_layer['weights']), hidden_1_layer['biases'])
l1 = tf.nn.relu(l1)

l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
l2 = tf.nn.relu(l2)
	
l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
l3 = tf.nn.relu(l3)

#keep_prob = tf.placeholder(tf.float32)
	
output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

prediction = output
y = tf.matmul(x,W) + b
	
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y))
optimizer = tf.train.AdamOptimizer().minimize(cost);

hm_epoches = 10

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

cnt = 0

for _ in range(1000):
	batch = mnist.train.next_batch(100)
	sess.run(optimizer, feed_dict = {x: batch[0], y_: batch[1]})
	cnt = cnt + 1
	#print(cnt)
		
correct = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accurcy = tf.reduce_mean(tf.cast(correct,'float'))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#print("Accuracy: %s" %(accurcy.eval({x:mnist.test.images, y_:mnist.test.labels})))
print(sess.run(accurcy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
