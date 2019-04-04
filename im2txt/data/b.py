#train2017 :118287;   val2017 :5000
'''
Loaded caption metadata for 118287 images from /home/thunder/桌面/Algorithmic/Tensorflow技术解析与实战/结合/MSCOCO2017/annotationstrain/captions_train2017.json
Processing captions.
Finished processing 591753 captions for 118287 images in /home/thunder/桌面/Algorithmic/Tensorflow技术解析与实战/结合/MSCOCO2017/annotationstrain/captions_train2017.json
Loaded caption metadata for 5000 images from /home/thunder/桌面/Algorithmic/Tensorflow技术解析与实战/结合/MSCOCO2017/annotationstrain/captions_val2017.json
Processing captions.
Finished processing 25014 captions for 5000 images in /home/thunder/桌面/Algorithmic/Tensorflow技术解析与实战/结合/MSCOCO2017/annotationstrain/captions_val2017.json
122537 250 500
Creating vocabulary.
Total words: 30029
Words in vocabulary: 11730
Wrote vocabulary file: /home/thunder/桌面/Algorithmic/Tensorflow技术解析与实战/结合/MSCOCO2017/word_counts.txt
2018-12-18 19:07:09.476174: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
Launching 8 threads for spacings: [[0, 76626], [76626, 153253], [153253, 229880], [229880, 306507], [306507, 383134], [383134, 459761], [459761, 536388], [536388, 613015]]

'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import Counter
from collections import namedtuple
from datetime import datetime
import json
import os.path
import random
import sys
import threading
import nltk.tokenize
import numpy as np
import tensorflow as tf
#训练图像地址
tf.flags.DEFINE_string("train_image_dir", "/home/thunder/桌面/Algorithmic/Tensorflow技术解析与实战/结合/MSCOCO2017/train2017/",
                       "Training image directory.")
#验证图像地址
tf.flags.DEFINE_string("val_image_dir", "/home/thunder/桌面/Algorithmic/Tensorflow技术解析与实战/结合/MSCOCO2017/val2017/",
                       "Validation image directory.")
#训练标签地址
tf.flags.DEFINE_string("train_captions_file", "/home/thunder/桌面/Algorithmic/Tensorflow技术解析与实战/结合/MSCOCO2017/annotationstrain/captions_train2017.json",
                       "Training captions JSON file.")
#验证标签地址
tf.flags.DEFINE_string("val_captions_file", "/home/thunder/桌面/Algorithmic/Tensorflow技术解析与实战/结合/MSCOCO2017/annotationstrain/captions_val2017.json",
                       "Validation captions JSON file.")
#数据输出地址
tf.flags.DEFINE_string("output_dir", "/media/thunder/soft/MSCOCO-tfrecord", "Output data directory.")
#训练TFRecord files线程
tf.flags.DEFINE_integer("train_shards", 256,
                        "Number of shards in training TFRecord files.")
#验证TFRecord files线程
tf.flags.DEFINE_integer("val_shards", 4,
                        "Number of shards in validation TFRecord files.")
#测试TFRecord files线程
tf.flags.DEFINE_integer("test_shards", 8,
                        "Number of shards in testing TFRecord files.")
#文字开头
tf.flags.DEFINE_string("start_word", '<S>',
                       "Special word added to the beginning of each sentence.")
#文字结尾
tf.flags.DEFINE_string("end_word", '</S>',
                       "Special word added to the end of each sentence.")
#特殊字符
tf.flags.DEFINE_string("unknown_word", "<UNK>",
                       "Special word meaning 'unknown'.")
#包含在词汇表中的训练集中每个单词出现的最小次数。
tf.flags.DEFINE_integer("min_word_count", 4,
                        "The minimum number of occurrences of each word in the "
                        "training set for inclusion in the vocabulary.")
tf.flags.DEFINE_string("word_counts_output_file", "/home/thunder/桌面/Algorithmic/Tensorflow技术解析与实战/结合/MSCOCO2017/word_counts.txt",
                       "Output vocabulary file of word counts.")

tf.flags.DEFINE_integer("num_threads", 8,
                        "Number of threads to preprocess the images.")

FLAGS = tf.flags.FLAGS

ImageMetadata = namedtuple("ImageMetadata",
                           ["image_id", "filename", "captions"])


class Vocabulary(object):
 
  def __init__(self, vocab, unk_id):    
    self._vocab = vocab
    self._unk_id = unk_id

  def word_to_id(self, word):
    """Returns the integer id of a word string."""
    if word in self._vocab:
      return self._vocab[word]
    else:
      return self._unk_id


class ImageDecoder(object):
  """Helper class for decoding images in TensorFlow."""

  def __init__(self):
    # Create a single TensorFlow Session for all image decoding calls.
    self._sess = tf.Session()

    # TensorFlow ops for JPEG decoding.
    self._encoded_jpeg = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._encoded_jpeg, channels=3)

  def decode_jpeg(self, encoded_jpeg):
    image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._encoded_jpeg: encoded_jpeg})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def _int64_feature(value):
  """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
  #return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
'''
def _bytes_feature(value):
  """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))
  #return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
'''
def _bytes_feature(value):  
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value])) 
 
def _int64_feature_list(values):
  """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
  return tf.train.FeatureList(feature=[_int64_feature(v)  for v in values])

def _bytes_features(value): 
  value=bytes(value, encoding = "utf8") 
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value])) 

def _bytes_feature_list(values):
  """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
  return tf.train.FeatureList(feature=[_bytes_features(v)  for v in values])


def _to_sequence_example(image, decoder, vocab):
  with tf.gfile.FastGFile(image.filename, 'rb') as f:
    encoded_image = f.read()
  #print(encoded_image)
  try:
    decoder.decode_jpeg(encoded_image)
  except (tf.errors.InvalidArgumentError, AssertionError):
    print("Skipping file with invalid JPEG data: %s" % image.filename)
    return
  #print(decoder.decode_jpeg(encoded_image))
  context = tf.train.Features(feature={
      "image/image_id": _int64_feature(image.image_id),
      "image/data": _bytes_feature(encoded_image),
  })
  #print(context)
  assert len(image.captions) == 1
  caption = image.captions[0]
  #print(caption)
  caption_ids = [vocab.word_to_id(word) for word in caption]
  #print(caption_ids)
  
  feature_lists = tf.train.FeatureLists(feature_list={
      "image/caption": _bytes_feature_list(caption),
      "image/caption_ids": _int64_feature_list(caption_ids)
  })
  
  sequence_example = tf.train.SequenceExample(
      context=context, feature_lists=feature_lists)

  return sequence_example
  

def _process_image_files(thread_index, ranges, name, images, decoder, vocab, num_shards):
  num_threads = len(ranges)
  assert not num_shards % num_threads
  #每个线程将产生的文件数
  num_shards_per_batch = int(num_shards / num_threads)
  #表示线程内部的数据分块标记
  shard_ranges = np.linspace(ranges[thread_index][0], ranges[thread_index][1],
                             num_shards_per_batch + 1).astype(int)
  #表示线程负责写入的数据样本总数
  num_images_in_thread = ranges[thread_index][1] - ranges[thread_index][0]
  #print( num_shards_per_batch,shard_ranges,num_images_in_thread)
  counter = 0

  for s in range(num_shards_per_batch):
    # 生成一个线程的文件名例如'train-00002-of-00010'
    shard = thread_index * num_shards_per_batch + s
    output_filename = "%s-%.5d-of-%.5d" % (name, shard, num_shards)
    #print(output_filename)
    #向output_file中写入带有文件名的空文件
    output_file = os.path.join(FLAGS.output_dir, output_filename)
    writer = tf.python_io.TFRecordWriter(output_file)

    shard_counter = 0
    #将所有图像分成256份,例如这里共有train的标签图像613014，故每个线程大约2395个图像
    images_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
    #print(images_in_shard,images_in_shard.shape)
    
    for i in images_in_shard:
      image = images[i]      
      sequence_example = _to_sequence_example(image, decoder, vocab)
      #print(sequence_example)
      
      if sequence_example is not None:
        # 序列化，并写入TFRecord文件
        writer.write(sequence_example.SerializeToString())
        shard_counter += 1
        counter += 1

      if not counter % 1000:
        print("%s [thread %d]: Processed %d of %d items in thread batch." %
              (datetime.now(), thread_index, counter, num_images_in_thread))
        sys.stdout.flush()
    # 关闭写文件器
    writer.close()
    print("%s [thread %d]: Wrote %d image-caption pairs to %s" %
          (datetime.now(), thread_index, shard_counter, output_file))
    sys.stdout.flush()
    shard_counter = 0
  print("%s [thread %d]: Wrote %d image-caption pairs to %d shards." %
        (datetime.now(), thread_index, counter, num_shards_per_batch))
  sys.stdout.flush()


def _process_dataset(name, images, vocab, num_shards):
  ## 因为一个图片有5个caption，所以需要为每个caption存储一个图片，形成ImageMetadata列表
  images = [ImageMetadata(image.image_id, image.filename, [caption])
            for image in images for caption in image.captions]
  # #随机排序，即将顺序打乱，不然同样一幅图像对应的多个标注会连在一起
  random.seed(12345)
  random.shuffle(images)

  # 将图像分割成数字线程批。这里分成了num_threads份，批处理定义为images[ranges[i][0]:ranges[i][1]].
  num_threads = min(num_shards, FLAGS.num_threads)
  spacing = np.linspace(0, len(images), num_threads + 1).astype(np.int)
  #spacing=[0 76626 153253 229880 306507 383134 459761 536388 613015]
  ranges = []
  threads = []
  for i in range(len(spacing) - 1):
    ranges.append([spacing[i], spacing[i + 1]])
  #ranges=[[0, 76626], [76626, 153253], [153253, 229880], [229880, 306507], 
  #        [306507, 383134], [383134, 459761], [459761,  536388], [536388, 613015]]

  #创建一个监视所有线程何时完成的机制。
  coord = tf.train.Coordinator()
  #创建一个用于解码JPEG图像的实用程序来运行健全性检查。
  decoder = ImageDecoder() 
  print("Launching %d threads for spacings: %s" % (num_threads, ranges))
  
  # 为每个batch启动一个thread .
  for thread_index in range(len(ranges)):
    args = (thread_index, ranges, name, images, decoder, vocab, num_shards)
    t = threading.Thread(target=_process_image_files, args=args)
    t.start()
    threads.append(t)
  # 等待所有线程threads终止
  coord.join(threads)
  print("%s: Finished processing all %d image-caption pairs in data set '%s'." %
        (datetime.now(), len(images), name))
 

def _create_vocab(captions): 
  print("Creating vocabulary.")
  #统计单词数
  counter = Counter()
  for c in captions:
    counter.update(c)
  #print(counter)
  print("Total words:", len(counter))

  #print(counter.items())
  #counter.items() 例如: ([('completely', 82), ('frutis', 2)])
  #单词出现次数低于FLAGS.min_word_count将会被排除
  word_counts = [x for x in counter.items() if x[1] >= FLAGS.min_word_count]
  #将单词按照出现次数逆序排列，即由大到小的顺序
  word_counts.sort(key=lambda x: x[1], reverse=True)
  #len(word_counts)=11731
  print("Words in vocabulary:", len(word_counts))

  # 写出单词计数文件
  with tf.gfile.FastGFile(FLAGS.word_counts_output_file, "w") as f:
    f.write("\n".join(["%s %d" % (w, c) for w, c in word_counts]))
  print("Wrote vocabulary file:", FLAGS.word_counts_output_file)

  # 创建词汇词典。
  reverse_vocab = [x[0] for x in word_counts]
  unk_id = len(reverse_vocab)
  # 将单词进行编号，并将号码放在单词后面
  vocab_dict = dict([(x, y) for (y, x) in enumerate(reverse_vocab)])
  #单词若是在词汇中便会输出与其对应的号码，否则便会输出unk_id表示未知符号
  vocab = Vocabulary(vocab_dict, unk_id) 
  return vocab


#将标签的句子分成单个词汇，并在每一句开头加上<S>,尾部加上</S>,
#.upper()把所有字符中的小写字母转换成大写字母
#.lower()把所有字符中的大写字母转换成小写字母
def _process_caption(caption):
  tokenized_caption = [FLAGS.start_word]
  tokenized_caption.extend(nltk.tokenize.word_tokenize(caption.lower()))
  tokenized_caption.append(FLAGS.end_word)  
  return tokenized_caption


#处理captions并将数据合并到ImageMetadata列表中。
def _load_and_process_metadata(captions_file, image_dir):
  #读取文件信息，包含["images"]和["annotations"]等信息
  with tf.gfile.FastGFile(captions_file, "r") as f:
    caption_data = json.load(f)
    # caption_data["images"]
    #例如{'date_captured': '2013-11-25 14:00:10', 'file_name': '000000015335.jpg', 
    #'flickr_url': 'http://  farm6.staticflickr.com/5533/10257288534_c916fafd78_z.jpg', 'height': 480, 
    #'coco_url': 'http://images.cocodataset.org/val2017/000000015335.jpg', 'license': 2, 'width': 640, 'id': 15335}

    # caption_data["annotations"]
    #例如{'image_id': 386134, 'caption': 'Food is in a styrofoam take out container.', 'id': 826793}

  #从caption_data["images"]中提取id 和 file_name 信息
  id_to_filename = [(x["id"], x["file_name"]) for x in caption_data["images"]]
  #id_to_filename例如(15335, '000000015335.jpg')

  #提取caption_data["annotations"]信息。 每个image_id都与多个标题相关联。
  id_to_captions = {}
  for annotation in caption_data["annotations"]:
    image_id = annotation["image_id"]
    caption = annotation["caption"]
    id_to_captions.setdefault(image_id, [])#dict.setdefault(key, default=None)
    id_to_captions[image_id].append(caption)
  #print(id_to_captions)
  #id_to_captions例如 {450559: ['A teenager doing skateboard tricks at a skate park']}


  #判断(id_to_filename)长度是否等于(id_to_captions)
  assert len(id_to_filename) == len(id_to_captions)
  #判断id_to_filename["id"]是否与id_to_captions[image_id]对应
  assert set([x[0] for x in id_to_filename]) == set(id_to_captions.keys())
  print("Loaded caption metadata for %d images from %s" %
        (len(id_to_filename), captions_file))


  #处理captions并将数据合并到ImageMetadata列表中。
  print("Processing captions.")
  image_metadata = []
  num_captions = 0
  for image_id, base_filename in id_to_filename:
    filename = os.path.join(image_dir, base_filename)
    captions = [_process_caption(c) for c in id_to_captions[image_id]]
    #print(captions)
    image_metadata.append(ImageMetadata(image_id, filename, captions))
    num_captions += len(captions)#标签句子数
  #print(num_captions)
  #print(image_metadata)
  '''
  image_metadata 例如下面：
  ImageMetadata(image_id=15335, filename='/home/thunder/桌面/Tensorflow技术解析与实战/结合/val2017/000000015335.jpg',
  captions=[['<S>', 'a', 'group', 'of', 'people', 'sitting', 'at', 'a', 'table', 'with', 'food', '.', '</S>'], 
  ['<S>', 'a', 'man', ',', 'woman', ',', 'and', 'boy', 'are', 'sitting', 'at', 'a', 'table', '.', '</S>'],
  ['<S>', 'a', 'man', ',', 'woman', 'and', 'child', 'eating', 'together', 'at', 'a', 'restaurant', '.', '</S>'],
  ['<S>', 'a', 'boy', 'sitting', 'between', 'a', 'man', 'and', 'a', 'woman', '.', '</S>'],
  ['<S>', 'a', 'young', 'child', ',', 'lady', ',', 'and', 'man', 'sitting', 'in', 'a', 'booth', 'at', 'a', 'table', '</S>']])]
  '''
  print("Finished processing %d captions for %d images in %s" %
        (num_captions, len(id_to_filename), captions_file))
  return image_metadata


def main(unused_argv):
  
  #判断线程数是否小于FLAGS.num_threads或可以整除FLAGS.num_threads后取反是否为真，有一个为真就是真。
  def _is_valid_num_shards(num_shards):
    """Returns True if num_shards is compatible with FLAGS.num_threads."""
    return num_shards < FLAGS.num_threads or not num_shards % FLAGS.num_threads

  assert _is_valid_num_shards(FLAGS.train_shards), (
      "Please make the FLAGS.num_threads commensurate with FLAGS.train_shards")
  assert _is_valid_num_shards(FLAGS.val_shards), (
      "Please make the FLAGS.num_threads commensurate with FLAGS.val_shards")
  assert _is_valid_num_shards(FLAGS.test_shards), (
      "Please make the FLAGS.num_threads commensurate with FLAGS.test_shards")
  #判断是否有输出文件夹，若没有则创建一个新的
  if not tf.gfile.IsDirectory(FLAGS.output_dir):
    tf.gfile.MakeDirs(FLAGS.output_dir)


  # 处理captions并将数据合并到ImageMetadata列表中。
  mscoco_train_dataset = _load_and_process_metadata(FLAGS.train_captions_file,
                                                    FLAGS.train_image_dir)
  mscoco_val_dataset = _load_and_process_metadata(FLAGS.val_captions_file,
                                                  FLAGS.val_image_dir)

  # 设置train，val，test的数量
  train_cutoff = int(0.85 * len(mscoco_val_dataset))
  val_cutoff = int(0.90 * len(mscoco_val_dataset))
  train_dataset = mscoco_train_dataset + mscoco_val_dataset[0:train_cutoff]
  val_dataset = mscoco_val_dataset[train_cutoff:val_cutoff]
  test_dataset = mscoco_val_dataset[val_cutoff:]
  print(len(train_dataset), len(val_dataset),len(test_dataset))
  #print(val_dataset)

  #从训练的标签中创建一个词汇文本
  # Create vocabulary from the training captions.
  train_captions = [c for image in train_dataset for c in image.captions]
  #print(train_captions)
  vocab = _create_vocab(train_captions)
  #print(vocab)

  #将分好的train，val，test转化为tfrecord格式
  _process_dataset("train", train_dataset, vocab, FLAGS.train_shards) 
  _process_dataset("val", val_dataset, vocab, FLAGS.val_shards)
  _process_dataset("test", test_dataset, vocab, FLAGS.test_shards)

  
if __name__ == "__main__":
  tf.app.run()
