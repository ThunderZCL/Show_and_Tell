import tensorflow as tf
def RenameCkpt(): # 1.0.1 : 1.2.1 
	vars_to_rename = { 
	"lstm/lstm_cell/weights": "lstm/lstm_cell/kernel", 
	"lstm/lstm_cell/biases": "lstm/lstm_cell/bias", 
	} 
	new_checkpoint_vars = {} 
	reader = tf.train.NewCheckpointReader('/home/thunder/桌面/Algorithmic/show_and_tell/im2txt/save/model.ckpt-961038') 
	for old_name in reader.get_variable_to_shape_map(): 
		if old_name in vars_to_rename: 
			new_name = vars_to_rename[old_name] 
		else: 
			new_name = old_name 
		new_checkpoint_vars[new_name] = tf.Variable(reader.get_tensor(old_name)) 
	init = tf.global_variables_initializer() 
	saver = tf.train.Saver(new_checkpoint_vars) 
	with tf.Session() as sess: 
		sess.run(init) 
		saver.save(sess, "/home/thunder/桌面/Algorithmic/show_and_tell/im2txt/saver/model.ckpt-961038") 			
	print("checkpoint file rename successful... ")
RenameCkpt()
