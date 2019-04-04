import tensorflow as tf
inputs=[32,1,2,3]
h=tf.shape(inputs)[0]
input_length = tf.expand_dims(tf.subtract(h, 1), 0)
t=tf.slice(inputs,[0],input_length)
t1=tf.slice(inputs,[1],input_length)

print(tf.Session().run(h))
print(tf.Session().run(input_length))
print(tf.Session().run(t))
print(tf.Session().run(t1))


