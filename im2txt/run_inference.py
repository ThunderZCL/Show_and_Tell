from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import os
import cv2
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import configuration
import inference_wrapper
import pyttsx3
from inference_utils import caption_generator
from inference_utils import vocabulary
from PIL import Image
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("checkpoint_path", "./saver",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("vocab_file", "word_counts.txt", "Text file containing the vocabulary.")
tf.flags.DEFINE_string("input_files", "/home/thunder/桌面/Algorithmic/show_and_tell/g3doc",
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")

tf.logging.set_verbosity(tf.logging.INFO)

def main(_):
  # Build the inference graph.
  g = tf.Graph()
  with g.as_default():
    model = inference_wrapper.InferenceWrapper()
    restore_fn = model.build_graph_from_config(configuration.ModelConfig(),FLAGS.checkpoint_path)

  vocab = vocabulary.Vocabulary(FLAGS.vocab_file)

  filenames = []
  data_dir= FLAGS.input_files
  tf_flie_pattern = os.path.join(data_dir, '*.jpg' )

  for file_pattern in tf_flie_pattern.split(","):
    filenames.extend(tf.gfile.Glob(file_pattern))
  tf.logging.info("Running caption generation on %d files matching %s",
                  len(filenames), FLAGS.input_files)

  with tf.Session(graph=g) as sess: 
    restore_fn(sess)
    generator = caption_generator.CaptionGenerator(model, vocab)

    font=cv2.FONT_HERSHEY_SIMPLEX
    for filename in filenames:
      with tf.gfile.GFile(filename, "rb") as f:
        image = f.read()      
      frame = tf.image.decode_jpeg(image)
      frame = sess.run(frame) 
      print(type(frame))
      captions = generator.beam_search(sess, image)  
      print("Captions for image %s:" % os.path.basename(filename))
      for i, caption in enumerate(captions):
        sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]       
        sentence = " ".join(sentence)  
        print("%d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))

        if i==0:
          s=sentence

      plt.axis('off')
      plt.title(s)
      plt.imshow(frame)
      plt.show() 
      #eng1 = pyttsx3.init()
      #rate = eng1.getProperty('rate')
      #eng1.setProperty('rate', rate-90)
      #eng1.say(s)
      #eng1.runAndWait()
      

if __name__ == "__main__":
  tf.app.run()
