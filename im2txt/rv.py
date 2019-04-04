from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os

import cv2
import tensorflow as tf

import configuration
import inference_wrapper
from inference_utils import caption_generator
from inference_utils import vocabulary

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "./save",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("vocab_file", "word_counts.txt", "Text file containing the vocabulary.")

tf.logging.set_verbosity(tf.logging.INFO)


def main(_):
  # Build the inference graph.
  g = tf.Graph()
  with g.as_default():
    model = inference_wrapper.InferenceWrapper()
    restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                               FLAGS.checkpoint_path)
  #g.finalize()

  # Create the vocabulary.
  vocab = vocabulary.Vocabulary(FLAGS.vocab_file)
  
  with tf.Session(graph=g) as sess:
    # Load the model from checkpoint.
    restore_fn(sess)

    # Prepare the caption generator. Here we are implicitly using the default
    # beam search parameters. See caption_generator.py for a description of the
    # available beam search parameters.
    generator = caption_generator.CaptionGenerator(model, vocab)
    font=cv2.FONT_HERSHEY_SIMPLEX
    cap = cv2.VideoCapture(0)
    while(True):
      ret, frame = cap.read()
      #cap.set(cv2.CAP_PROP_FPS,1)
      #cap.set(3,1000)
      #cap.set(4,1000)
      image = tf.image.encode_jpeg(frame)
      image = sess.run(image)
      captions = generator.beam_search(sess, image)
      caption=captions[0]
      #for i, caption in enumerate(captions):
        # Ignore begin and end words.
      sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]

      sentence = " ".join(sentence)
      print(" %s (p=%f)" % ( sentence, math.exp(caption.logprob)))
      
      cv2.putText(frame,sentence,(0,20), font, 0.6,(255,255,255),2)
      cv2.imshow('frame',frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cap.release()
    cv2.destroyAllWindows()			


if __name__ == "__main__":
  tf.app.run()
