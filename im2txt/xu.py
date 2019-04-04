import tensorflow as tf
 
'''FIFO队列操作'''
 
# 创建队列
# 队列有两个int32的元素
q = tf.FIFOQueue(2,'int32')
# 初始化队列
init= q.enqueue_many(([0,10],))
# 出队
x = q.dequeue()
y = x + 1
# 入队
q_inc = q.enqueue([y])
 
with tf.Session() as sess:
    init.run()
    for _ in range(5):
        v,_ = sess.run([x,q_inc])
        print(v)
