"""Input ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
#解析tensorflow.SequenceExample为图片和caption
def parse_sequence_example(serialized, image_feature, caption_feature):
  
  context, sequence = tf.parse_single_sequence_example(
      serialized,
      context_features={
          image_feature: tf.FixedLenFeature([], dtype=tf.string)
      },
      sequence_features={
          caption_feature: tf.FixedLenSequenceFeature([], dtype=tf.int64),
      })
  encoded_image = context[image_feature]
  caption = sequence[caption_feature]
  return encoded_image, caption

#从磁盘读取字符串形式的数值，到输入队列
def prefetch_input_data(reader,
                        file_pattern,
                        is_training,
                        batch_size,
                        values_per_shard,
                        input_queue_capacity_factor=16,
                        num_reader_threads=1,
                        shard_queue_name="filename_queue",
                        value_queue_name="input_queue"):
  # 根据文件名模式，获取TFRecord文件名列表
  data_files = []
  for pattern in file_pattern.split(","):
    data_files.extend(tf.gfile.Glob(pattern))
  if not data_files:
    tf.logging.fatal("Found no input files matching %s", file_pattern)
  else:
    tf.logging.info("Prefetching values from %d files matching %s",
                    len(data_files), file_pattern)

  if is_training:
    # 文件名队列生成器
    filename_queue = tf.train.string_input_producer(
        data_files, shuffle=True, capacity=16, name=shard_queue_name)

    # 队列最小数据量
    min_queue_examples = values_per_shard * input_queue_capacity_factor
    # 队列数据容量
    capacity = min_queue_examples + 100 * batch_size

    # 随机数值队列
    values_queue = tf.RandomShuffleQueue(
        capacity=capacity,
        min_after_dequeue=min_queue_examples,
        dtypes=[tf.string],
        name="random_" + value_queue_name)
  else:
    # 文件名队列生成器
    filename_queue = tf.train.string_input_producer(
        data_files, shuffle=False, capacity=1, name=shard_queue_name)
    # 队列数据容量
    capacity = values_per_shard + 3 * batch_size
    # 先入先出队列
    values_queue = tf.FIFOQueue(
        capacity=capacity, dtypes=[tf.string], name="fifo_" + value_queue_name)


 # 多线程
  enqueue_ops = []
  for _ in range(num_reader_threads):
    # 读文件队列
    _, value = reader.read(filename_queue)
    # 数据入队操作,并将操作存入到列表对象enqueue_ops，enqueue_ops含有num_reader_threads个入队列操作
    enqueue_ops.append(values_queue.enqueue([value]))

  # 添加队列执行器，实现数据队列的多线程入队列定义
  tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(
      values_queue, enqueue_ops))
  # 添加随机数据队列的尺寸标量的总结
  tf.summary.scalar(
      "queue/%s/fraction_of_%d_full" % (values_queue.name, capacity),
      tf.cast(values_queue.size(), tf.float32) * (1. / capacity))
  return values_queue



def batch_with_dynamic_pad(images_and_captions,
                           batch_size,
                           queue_capacity,
                           add_summaries=True):
  """Batches input images and captions.
  把image-caption对 列表batch化

  “”批量输入图像和标题。

   此函数将标题拆分为输入序列和目标序列，
   其中目标序列是右移1的输入序列。输入和
   将靶序列分批并填充至最大序列长度
   在批次中。 创建掩码以区分真实单词和填充单词。
  1）caption转换成多组：输入word序列（input sequence）-->目标输出word序列（target sequence）
                       其中，目标输出word序列是输入word序列右移1个位置形成的
  2）全部补充（pad)到最长序列，capacity: 整数。在队列中的最大元素个数。
  3）生成对应的真word序列mask
  4) 生成batch数据

  Example:
    Actual captions in the batch ('-' denotes padded character补充字符):
      [
        [ 1 2 5 4 5 ],
        [ 1 2 3 4 - ],
        [ 1 2 3 - - ],
      ]

    input_seqs:
      [
        [ 1 2 3 4 ],
        [ 1 2 3 - ],
        [ 1 2 - - ],
      ]

    target_seqs:
      [
        [ 2 3 4 5 ],
        [ 2 3 4 - ],
        [ 2 3 - - ],
      ]

    mask:
      [
        [ 1 1 1 1 ],
        [ 1 1 1 0 ],
        [ 1 1 0 0 ],
      ]

  Args:
    images_and_captions: A list of pairs [image, caption], where image is a
      Tensor of shape [height, width, channels] and caption is a 1-D Tensor of
      any length. Each pair will be processed and added to the queue in a
      separate thread.
    batch_size: Batch size.
    queue_capacity: Queue capacity.
    add_summaries: If true, add caption length summaries.

  Returns:
    images: 图片数据batch --> A Tensor of shape [batch_size, height, width, channels].
    input_seqs: 输入序列batch --> An int32 Tensor of shape [batch_size, padded_length].
    target_seqs: 目标序列batch --> An int32 Tensor of shape [batch_size, padded_length].
    mask: 真word序列batch --> An int32 0/1 Tensor of shape [batch_size, padded_length].
  """
  # 生成入队列表
  enqueue_list = []
  for image, caption in images_and_captions:
    #得出每个标签的长度
    caption_length = tf.shape(caption)[0]
    #目标输出word序列是输入word序列右移1个位置形成的，故先得出每个输入的长度
    input_length = tf.expand_dims(tf.subtract(caption_length, 1), 0)
    input_seq = tf.slice(caption, [0], input_length)
    target_seq = tf.slice(caption, [1], input_length)
    indicator = tf.ones(input_length, dtype=tf.int32)
    enqueue_list.append([image, input_seq, target_seq, indicator])
  # 生成batch数据的队列
  images, input_seqs, target_seqs, mask = tf.train.batch_join(
      enqueue_list,
      batch_size=batch_size,
      capacity=queue_capacity,
      dynamic_pad=True,
      name="batch_and_pad")
  # 添加总结
  if add_summaries:
    lengths = tf.add(tf.reduce_sum(mask, 1), 1)
    tf.summary.scalar("caption_length/batch_min", tf.reduce_min(lengths))
    tf.summary.scalar("caption_length/batch_max", tf.reduce_max(lengths))
    tf.summary.scalar("caption_length/batch_mean", tf.reduce_mean(lengths))

  return images, input_seqs, target_seqs, mask
