from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf

import numpy
numpy.set_printoptions(threshold=numpy.nan)

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def binarize(x, y):
	return tf.assign(y, tf.sign(x))

def binarize_all():
	bin_steps = []
	for var in tf.trainable_variables():
		bin_steps.append(binarize(var, var))
	return bin_steps

def norm_and_binarize(x):
	x1 = tf.nn.batch_normalization(x, 0, 1, 0, 1, .0001)
	x2 = tf.sign(tf.subtract(x1, .1))
	return x2
	

# Define input and output
x = tf.placeholder(tf.float32, shape=[None, 784])
x_image = tf.reshape(x, [-1, 28, 28, 1])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

conv1_size = 32
conv2_size = 64

# Create Model
W_scal1 = tf.Variable(1.0, trainable=False)
W_conv1 = weight_variable([5, 5, 1, conv1_size])
W_conv1_b = weight_variable([5, 5, 1, conv1_size])
x_image_norm_b = norm_and_binarize(x_image)

b_conv1 = bias_variable([conv1_size])

h_conv1 = tf.nn.dropout(tf.scalar_mul(W_scal1, conv2d(x_image_norm_b, W_conv1_b)), 1)# + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
h_pool1_norm = norm_and_binarize(h_pool1)

W_scal2 = tf.Variable(1.0, trainable=False)
W_conv2 = weight_variable([5, 5, conv1_size, conv2_size])
W_conv2_b = weight_variable([5, 5, conv1_size, conv2_size])
#b_conv2 = bias_variable([conv2_size])

h_conv2 = tf.nn.dropout(tf.scalar_mul(W_scal2, conv2d(h_pool1_norm, W_conv2_b)), 1)# + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
h_pool2_norm = norm_and_binarize(h_pool2)

W_fc1 = weight_variable([7 * 7 * conv2_size, 1024])
W_fc1_b = weight_variable([7 * 7 * conv2_size, 1024])
#b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2_norm, [-1, 7*7*conv2_size])
h_fc1 = tf.nn.dropout(tf.matmul(h_pool2_flat, W_fc1_b), 1)# + b_fc1)
h_fc1_norm = norm_and_binarize(h_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1_norm, keep_prob)

W_fc2 = weight_variable([1024, 10])
W_fc2_b = weight_variable([1024, 10])
#b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2_b)# + b_fc2

h_test = h_pool2_norm

# Create error function
cross_entropy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

# Define training step for simple optimization
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Definte optimization method for BNN: Use binarized weights for forward prop and backprop, but update the real weights
opt = tf.train.AdamOptimizer(1e-4)
grads_and_vars = opt.compute_gradients(cross_entropy, [W_conv1_b, W_conv2_b, W_fc1_b, W_fc2_b])

variables_b = [v for g, v in grads_and_vars if g is not None]
gradients_b = [g for g, v in grads_and_vars if g is not None]

grads_and_vars_new = [(g, v) for g, v in zip(gradients_b, [W_conv1, W_conv2, W_fc1, W_fc2])]

if not variables_b:
	raise ValueError(
		"No gradients provided for any variable, check your graph for ops"
		" that do not support gradients, between variables %s and loss %s." %
		([str(v) for _, v in grads_and_vars], cross_entropy))


step = opt.apply_gradients(grads_and_vars_new)




# Binarize and scale weights
scale_step1 = tf.assign(W_scal1, tf.scalar_mul((1.0/25.0), tf.norm(W_conv1)))
scale_step2 = tf.assign(W_scal2, tf.scalar_mul((1.0/25.0), tf.norm(W_conv2)))

bin_steps = []
bin_steps.append(binarize(W_conv1, W_conv1_b))
bin_steps.append(binarize(W_conv2, W_conv2_b))
bin_steps.append(binarize(W_fc1, W_fc1_b))
bin_steps.append(binarize(W_fc2, W_fc2_b))

# Normalize and binarize inputs
bin_inputs = norm_and_binarize(W_conv1)



#Define accuracy function
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Start training
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for qsy in range(1):
		# Start training
		for i in range(40000):
			# Get scale value and then binarize weights
			sess.run(scale_step1)
			sess.run(scale_step2)
			sess.run(bin_steps)
			
			batch = mnist.train.next_batch(50)
			if i % 100 == 0:
				train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
				print('step %d, training accuracy %g' % (i, train_accuracy))
			sess.run(step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

		print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images[0:5000], y_: mnist.test.labels[0:5000], keep_prob: 1.0}))
		
		#print("layer1: ", h_test.eval(feed_dict={x: mnist.test.images[0:1]}, session=sess))
		
		#print(sess.run(h_conv1,feed_dict={x: mnist.test.images[0], y_: mnist.test.labels[0], keep_prob: 1.0}))
	#print(sess.run(W_conv1_b))
	#print(sess.run(W_conv2_b))
	#print(sess.run(W_fc1_b))
	#print(sess.run(W_fc2_b))




































