from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

class ModelConfig(object):
  """Wrapper class for model hyperparameters."""
  def __init__(self):
    """Sets the default model hyperparameters."""
    # sharded TFRecord文件的命名模式
    self.input_file_pattern = None
    # Image format ("jpeg" or "png").
    self.image_format = "jpeg"
    # 每个TFRecord文件的大约数量
    self.values_per_input_shard = 2300
    # 输入队列的最少shards数量
    self.input_queue_capacity_factor = 2
    # 读线程数量
    self.num_input_reader_threads = 1
    #包含图片数据的SequenceExample context feature名称
    self.image_feature_name = "image/data"
    # 包含caption word_id数据的SequenceExample feature list名称
    self.caption_feature_name = "image/caption_ids"
    ## 字典尺寸
    self.vocab_size = 12000
    # 用于图像预处理的线程数。应该是2的倍数。
    self.num_preprocess_threads = 4
    # Batch size.
    self.batch_size = 32
    #Inception v3的pre-trained模型文件，首次训练需要提供
    self.inception_checkpoint_file = None
    # Inception v3的图片输入尺寸
    self.image_height = 299
    self.image_width = 299
    # 模型变量初始化Scale
    self.initializer_scale = 0.08
    # LSTM的输入、输出维度
    self.embedding_size = 512
    self.num_lstm_units = 512
    # If < 1.0, the dropout keep probability applied to LSTM variables.
    self.lstm_dropout_keep_prob = 0.7


class TrainingConfig(object):
  """Wrapper class for training hyperparameters."""

  def __init__(self):
    """Sets the default training hyperparameters."""
    ## 每个epoch的training data.数量
    self.num_examples_per_epoch = 586363
    # 优化方法
    self.optimizer = "SGD"
    # 学习率
    self.initial_learning_rate = 2.0
    self.learning_rate_decay_factor = 0.5
    self.num_epochs_per_decay = 8.0
    # 调优Inception v3 模型参数的学习率
    self.train_inception_learning_rate = 0.0005
    # 梯度剪裁
    self.clip_gradients = 5.0
    # 可保留的最大checkpoints模型文件数量
    self.max_checkpoints_to_keep = 5
