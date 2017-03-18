import tensorflow as tf

x = tf.placeholder(tf.float32)

#model features
theta1 = tf.Variable([0.5], tf.float32)
theta2 = tf.Variable([5.0], tf.float32)

#linear regression model
model = theta1 * x + theta2

y = tf.placeholder(tf.float32)

#define cost function or loss function
cost = tf.reduce_sum(tf.square(model - y) / 2400)

#optimizer select, then provide the cost function to be minimized
optimizer = tf.train.GradientDescentOptimizer(0.09)
train = optimizer.minimize(cost)

#training to reduce cost
x_train = range(0, 24)
y_train = [16, 16, 16, 15, 14, 15, 14, 15, 15, 16, 18, 21, 24, 25, 26, 27, 28, 27, 26, 24, 22, 21, 20, 19]

#inititalize variables
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(20000):
	sess.run(train, {x: x_train, y: y_train})

print(sess.run([theta1, theta2]))
print("Cost = %s" % sess.run(cost, {x: x_train, y: y_train}))
