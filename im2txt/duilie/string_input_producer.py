import tensorflow as tf
files_in = ["./data/data_batch%d.bin" % i for i in range(1, 6)]
files = tf.train.string_input_producer(files_in)


