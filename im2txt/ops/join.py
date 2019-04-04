#-*- coding:utf-8 -*-
import tensorflow as tf
filenames = ['A.csv', 'B.csv', 'C.csv']
filename_queue = tf.train.string_input_producer(filenames, shuffle=False)
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)
record_defaults = [['null'], ['null']]
#定义了多种解码器,每个解码器跟一个reader相连
example_list = [tf.decode_csv(value, record_defaults=record_defaults)
                  for _ in range(2)]  # Reader设置为2

print(example_list)
# 使用tf.train.batch_join()，可以使用多个reader，并行读取数据。每个Reader使用一个线程。
example_batch, label_batch = tf.train.batch_join(
      example_list, batch_size=32)
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(1):
        e_val,l_val = sess.run([example_batch,label_batch])
        print (e_val,l_val)
    coord.request_stop()
    coord.join(threads)
