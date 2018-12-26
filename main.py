# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import tensor_networks
import data
# import pprint

# pp = pprint.PrettyPrinter()

flags = tf.flags

flags.DEFINE_integer("dim_emb", 200, "词向量的维度")
flags.DEFINE_integer("num_classes", 3, "情感极性")
flags.DEFINE_integer("num_slices", 20, "张量的切片数")
flags.DEFINE_integer("num_units", 120, "GRU隐藏单元数量")
flags.DEFINE_integer("batch_size", 100, "批量的大小")
flags.DEFINE_integer("num_layers", 2, "网络的层数")
flags.DEFINE_integer("num_epoch", 50, "训练次数")
flags.DEFINE_float("learn_rate", 0.01, "学习率")
flags.DEFINE_float("init_std", 3, "随机数方差")
# flags.DEFINE_string("train_data", "file_name", "训练数据")
# flags.DEFINE_string("test_data", "file_name", "测试数据")
flags.DEFINE_string("pre_train", "text_and_emb/crawl-300d.vec/glove.twitter.27B.200d.txt", "预训练词向量")
flags.DEFINE_integer("max_len", None, "最长句子的长度")
flags.DEFINE_integer("max_num_target", None, "最长目标短语的长度")
flags.DEFINE_list("pre_train_embedding", None, "词嵌入矩阵")
FLAGS = flags.FLAGS


# 生成嵌入矩阵
def data_embedding(word2idx):
	print("构建词嵌入矩阵")
	word2idx_matrix = np.random.normal(0, FLAGS.init_std, [len(word2idx), FLAGS.dim_emb])
	word2idx_matrix[word2idx['<pad>']].fill(0)
	with open(FLAGS.pre_train, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
		for line in f:
			content = line.split()
			if content[0] in word2idx:
				word2idx_matrix[word2idx[content[0]]] = np.array(list(map(float, content[1:])))
	return word2idx_matrix


def main(_):
	source_count = []
	source_word2idx = {}

	train_file_name = ["text_and_emb/Restaurants_Train.xml"]  # file_name = [FLAGS.train_data, FLAGS.test_data]
	test_file_name = ['text_and_emb/Restaurants_Test_Gold.xml']
	train_data = data.read(train_file_name, source_count, source_word2idx)
	test_data = data.read(test_file_name, source_count, source_word2idx)

	# FLAGS.pad_id = source_word2idx['<pad>']
	FLAGS.max_len = train_data[3] if train_data[3] > test_data[3] else test_data[3]
	FLAGS.max_num_target = train_data[4] if train_data[4] > test_data[4] else test_data[4]
	FLAGS.pre_train_embedding = data_embedding(source_word2idx)

	with tf.Session() as sess:
		model = tensor_networks.Model(FLAGS, sess)
		model.build_model()
		model.run(train_data, test_data)
	# sess = tf.InteractiveSession()
	# model = tensor_networks.Model(FLAGS, sess)
	# model.build_model()
	# model.run(train_data, test_data)


if __name__ == '__main__':
	tf.app.run()
