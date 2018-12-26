# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import math


def initial(shape):
	q = np.random.normal(0.0, 1.0, shape)
	return tf.Variable(q[:shape[0], :shape[1]], dtype=tf.float64, trainable=True)


class GRUCell(object):
	def __init__(self, inputs, num_units, initializer, l2=False, init_h=None):
		self.inputs = inputs
		self.batch_size = inputs.shape[1]  # 输入张量已经被转置过shape = [句子长度, batch, dim_emb]
		self.dim_input = inputs.shape[-1]  # 输入张量中每一个向量的维度
		self.dim_output = num_units  # 输出向量的维度
		self.initializer = initializer
		self.type = 'gru'

		if init_h is None:
			self.init_h = tf.Variable(np.zeros([self.batch_size, self.dim_output]), dtype=tf.float64)
			self.previous = self.init_h

		self.re_gate = self.gate()
		self.up_gate = self.gate()
		self.cell = self.gate()

		self.w_x = tf.concat(values=[self.re_gate[0], self.up_gate[0], self.cell[0]], axis=0)
		self.w_h = tf.concat(values=[self.re_gate[1], self.up_gate[1], self.cell[1]], axis=0)
		self.b = tf.concat(values=[self.re_gate[2], self.up_gate[2], self.cell[2]], axis=0)

		if l2:
			self.L2_loss = tf.nn.l2_loss(self.w_x) + tf.nn.l2_loss(self.w_h)

	def gate(self, bias=0.001):
		wx = self.initializer([self.dim_input, self.dim_output])
		wh = self.initializer([self.dim_output, self.dim_output])
		b = tf.Variable(tf.constant(bias, shape=[self.dim_output], dtype=tf.float64), trainable=True)
		return wx, wh, b

	def step(self, prev_h, current_x):
		self.w_x = tf.reshape(self.w_x, [-1, self.dim_input, self.dim_output])
		self.w_h = tf.reshape(self.w_h, [-1, self.dim_output, self.dim_output])
		self.b = tf.reshape(self.b, [-1, self.dim_output])

		x = tf.tensordot(current_x, self.w_x, axes=(1, 1)) + self.b
		h = tf.tensordot(prev_h, self.w_h, axes=(1, 1))

		x_t = tf.transpose(x, [1, 0, 2])
		h_t = tf.transpose(h, [1, 0, 2])

		r = tf.sigmoid(x_t[0] + h_t[0])
		u = tf.sigmoid(x_t[1] + h_t[1])

		c = tf.tanh(x_t[2] + r*h_t[2])

		current_h = (1 - u) * prev_h + u * c

		return current_h


def rnn(cell):
	hid_states = tf.scan(fn=cell.step, elems=cell.inputs, initializer=cell.previous, name='hidden_states')
	return hid_states


class Model(object):
	def __init__(self, config, sess):
		self.dim_emb = config.dim_emb
		self.num_classes = config.num_classes
		self.num_slices = config.num_slices
		self.max_len = config.max_len
		self.max_num_target = config.max_num_target
		self.num_units = config.num_units
		self.batch_size = config.batch_size
		self.num_epoch = config.num_epoch
		self.learn_rate = config.learn_rate
		self.init_std = config.init_std
		self.num_layers = config.num_layers

		self.pre_train_embedding = config.pre_train_embedding  # 词嵌入矩阵

		# tensor_g,v,c数值选用np进行初始化
		self.tensor_g = tf.Variable(
			0.2*np.random.normal(0.0, self.init_std, [self.num_slices, self.dim_emb, self.dim_emb]), name='tensor_g'
		)
		self.v = tf.Variable(0.2*np.random.normal(0.0, self.init_std, [1, self.num_units]), name='att_vec')
		self.c = tf.Variable(0.2*np.random.normal(0.0, self.init_std, [self.num_classes, self.dim_emb]), name='class_matrix')

		self.context = tf.placeholder(tf.int32, [self.batch_size, self.max_len], name='context')
		self.tar_word = tf.placeholder(tf.int32, [self.batch_size, self.max_num_target], name='target_word')
		self.target_label = tf.placeholder(tf.float64, [self.batch_size, 3], name='polarity')
		# self.target_len = tf.placeholder(tf.float64, [None], name='target_len')

		# self.pre_lab = tf.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
		self.keep_prob = 0.5
		self.layer_state = []
		self.sess = sess
		self.loss = None
		# self.step = None
		self.opt = None
		self.correct_prediction = None
		self.num_target = []

	def build_layer(self, input_tar, input_text):  # input_tar,input_text都是批量输入
		# tensor_operation批处理
		beta = tf.matmul(input_text, tf.tensordot(input_tar, self.tensor_g, axes=(1, 1)), transpose_b=True)
		beta = tf.tanh(beta)
		# beta = tf.nn.dropout(beta, keep_prob=self.keep_prob)

		# GRU网络
		cell = GRUCell(inputs=tf.transpose(beta, [1, 0, 2]), num_units=self.num_units, initializer=initial)
		r = rnn(cell)
		r_t = tf.transpose(r, [1, 0, 2])
		cell_outputs = tf.nn.dropout(r_t, keep_prob=self.keep_prob)

		# Attention机制
		# v = tf.Variable(np.random.normal(0.0, self.init_std, [1, self.num_units]), name='att_vec')
		e = tf.tensordot(cell_outputs, self.v, axes=(2, 1))
		alphas = tf.nn.softmax(e, axis=1)
		out_text = tf.multiply(alphas, input_text)
		# self.layer_state.append(out_text)
		return out_text

	def build_model(self):
		asp_emb = tf.reduce_sum(tf.nn.embedding_lookup(self.pre_train_embedding, self.tar_word), axis=1)
		# asp_emb = tf.map_fn(
		# 	lambda x: x[0] / x[1], (asp_emb_sum, self.target_len), dtype=tf.float64
		# )
		text_emb = tf.nn.embedding_lookup(self.pre_train_embedding, self.context)
		# 堆叠多个网络层
		print("堆叠多层网络")
		for layer in range(self.num_layers):
			if layer < 1:
				print("堆叠第" + str(layer+1) + "个网络")
				self.layer_state.append(self.build_layer(input_tar=asp_emb, input_text=text_emb))
			else:
				print("堆叠第" + str(layer+1) + "个网络")
				self.layer_state.append(self.build_layer(input_tar=asp_emb, input_text=self.layer_state[-1]))

		# 对极性进行分类
		print("极性分类")
		last_state = tf.reduce_sum(self.layer_state[-1], axis=1)
		prediction = tf.tensordot(last_state, self.c, axes=(1, 1))
		print(prediction.shape)
		print(self.target_label.shape)

		# 损失函数
		print("构建损失函数")
		self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.target_label, logits=prediction))
		# self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.target_label, logits=prediction)

		# 优化器
		print("构建优化器")
		# self.opt = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss + GRUCell.L2_loss)
		self.opt = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss)

		# 变量初始化
		self.correct_prediction = tf.argmax(prediction, 1)
		tf.global_variables_initializer().run()

		# self.training_summary = tf.summary.scalar("training_loss", tf.reduce_mean(self.loss))
		# self.validation_summary = tf.summary.scalar("validation_loss", tf.reduce_mean(self.loss))
		# logs_path = 'example/'
		# self.summary_writer = tf.summary.FileWriter(logs_path, self.sess.graph)

	def train(self, data):
		source_data, target_data, target_label = data[0], data[1], data[2]
		n = int(math.ceil(len(source_data) / self.batch_size))
		cost = 0

		target = np.zeros([self.batch_size, self.max_num_target], dtype=np.int32)
		text = np.zeros([self.batch_size, self.max_len], dtype=np.int32)
		true_lab = np.zeros([self.batch_size, 3], dtype=np.float64)
		rand_idx, cur = np.random.permutation(len(source_data)), 0
		print(len(rand_idx))

		for i in range(n):
			# tar_len = []
			# text.fill(self.pad_id)
			# text.fill(0)
			# target.fill(0)
			for b in range(self.batch_size):
				if cur >= len(rand_idx):
					break
				m = rand_idx[cur]
				target[b, :len(target_data[m])] = target_data[m]
				# tar_len.append(len(target_data[m]))
				text[b, :len(source_data[m])] = source_data[m]
				true_lab[b][target_label[m]] = 1.0
				cur += 1

			a, loss = self.sess.run(
				[self.opt, self.loss],
				feed_dict={
					self.tar_word: target, self.context: text, self.target_label: true_lab}
			)

			# a, loss = self.sess.run(
			# 	[self.opt, self.loss],
			# 	feed_dict={
			# 		self.tar_word: target, self.context: text, self.target_label: true_lab,
			# 		self.target_len: tar_len}
			# )
			cost += np.sum(loss)

		_, train_acc = self.test(data)
		return cost / n / self.batch_size, train_acc

	def test(self, data):
		source_data, target_data, target_label = data[0], data[1], data[2]
		n = int(math.ceil(len(source_data) / self.batch_size))
		cost = 0
		target = np.zeros([self.batch_size, self.max_num_target], dtype=np.int32)
		text = np.zeros([self.batch_size, self.max_len], dtype=np.int32)
		true_lab = np.zeros([self.batch_size, 3], dtype=np.float64)

		m, acc = 0, 0
		for i in range(n):
			# tar_len = []
			# text.fill(self.pad_id)
			# text.fill(0)
			# target.fill(0)
			raw_labels = []
			for b in range(self.batch_size):
				if m >= len(target_label):
					break
				target[b, :len(target_data[m])] = target_data[m]
				# tar_len.append(len(target_data[m]))
				text[b, :len(source_data[m])] = source_data[m]
				true_lab[b][target_label[m]] = 1
				raw_labels.append(target_label[m])
				m += 1

			loss = self.sess.run(
				[self.loss],
				feed_dict={
					self.tar_word: target, self.context: text, self.target_label: true_lab}
			)

			cost += np.sum(loss)

			predictions = self.sess.run(
				self.correct_prediction,
				feed_dict={
					self.tar_word: target, self.context: text, self.target_label: true_lab}
			)

			# loss = self.sess.run(
			# 	[self.loss],
			# 	feed_dict={
			# 		self.tar_word: target, self.context: text, self.target_label: true_lab,
			# 		self.target_len: tar_len}
			# )
			#
			# cost += np.sum(loss)
			#
			# predictions = self.sess.run(
			# 	self.correct_prediction,
			# 	feed_dict={
			# 		self.tar_word: target, self.context: text, self.target_label: true_lab,
			# 		self.target_len: tar_len}
			# )

			for b in range(self.batch_size):
				if b >= len(raw_labels):
					break
				if raw_labels[b] == predictions[b]:
					acc += 1
			print('正确数量' + str(acc))
		return cost / float(len(source_data)), acc / float(len(source_data))

	def run(self, train_data, test_data):
		print('training...')
		for epoch in range(self.num_epoch):
			print('epoch ' + str(epoch) + '...')
			train_loss, train_acc = self.train(data=train_data)
			test_loss, test_acc = self.test(data=test_data)
			print('train-loss=%.2f;train-acc=%.2f;test_loss=%.2f;test-acc=%.2f;' % (train_loss, train_acc, test_loss, test_acc))
