# -*- coding: utf-8 -*-

import tensorflow as tf

 

tensor_list = [[1,2,3,4], [5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20]]

tensor_list2 = [[[1,2,3,4]], [[5,6,7,8]],[[9,10,11,12]],[[13,14,15,16]],[[17,18,19,20]]]

with tf.Session() as sess:

    #x1 = tf.train.batch(tensor_list, batch_size=4, enqueue_many=False)

    #x2 = tf.train.batch(tensor_list, batch_size=4, enqueue_many=True)

 

    y1 = tf.train.batch_join(tensor_list, batch_size=4, enqueue_many=False)

    y2 = tf.train.batch_join(tensor_list2, batch_size=4, enqueue_many=False)

 

    coord = tf.train.Coordinator()

    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    '''
    print("x1 batch:"+"-"*88)

    print(sess.run(x1))

    print("x2 batch:"+"-"*88)

    print(sess.run(x2))
    '''

    print("y1 batch:"+"-"*88)

    print(sess.run(y1))
    
    print("y2 batch:"+"-"*88)

    print(sess.run(y2))

    print("-"*97)

    coord.request_stop()

    coord.join(threads)
