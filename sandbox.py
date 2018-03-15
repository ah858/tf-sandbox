import tensorflow as tf

# model parameters
w = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)

# inputs and outputs
x = tf.placeholder(tf.float32)
linear_model = w * x + b

y = tf.placeholder(tf.float32)

# loss
squared_delta = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_delta)

# optimize
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for i in range (1000):
    sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})

print(sess.run([w,b]))


#print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))


# example with multiplication and FileWriter to make a graph
"""
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b

sess = tf.Session()
File_Writer = tf.summary.FileWriter('/Users/AkilHashmi/Documents/GitHub/tensorflowPractice/graph', sess.graph)
print(sess.run(c))
sess.close()
"""

# this will print all values about the node
"""
print(node1, node2)
"""

# manually open and close session
"""
sess = tf.Session()
print(sess.run([node1, node2]))
sess.close()
"""

# Alternative way to open (and auto close session)
"""
with tf.Session() as sess:
    output = sess.run([node1, node2])
    print(output)
"""
