import tensorflow as tf
# 最大长度10,最小长度2,类型float的随机队列

q =tf.RandomShuffleQueue(capacity=10,min_after_dequeue=2,dtypes='float')
sess = tf.Session()
for i in range(0,10):
    sess.run(q.enqueue(i))

for i in range(0,8): # 在输出8次后会被阻塞
    print(sess.run(q.dequeue()))

file_pattern = "/media/thunder/soft/MSCOCO-tfrecord/tfrecord/train-?????-of-00256"
data_files = []
for pattern in file_pattern.split(","):
    data_files.extend(tf.gfile.Glob(pattern))
print(data_files)
reader = tf.TFRecordReader()

filename_queue = tf.train.string_input_producer(data_files, shuffle=True, capacity=16)


values_queue = tf.RandomShuffleQueue(
        capacity=7800,
        min_after_dequeue=4600,
        dtypes=[tf.string])
enqueue_ops = []
_, value = reader.read(filename_queue)
enqueue_ops.append(values_queue.enqueue([value]))
tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(values_queue, enqueue_ops))
example = values_queue.dequeue()
context, sequence = tf.parse_single_sequence_example(
      example,
      context_features={
          "image/data": tf.FixedLenFeature([], dtype=tf.string)
      },
      sequence_features={
          "image/caption_ids": tf.FixedLenSequenceFeature([], dtype=tf.int64),
      })
encoded_image = context["image/data"]
caption = sequence["image/caption_ids"]
print(sess.run(encoded_image))









