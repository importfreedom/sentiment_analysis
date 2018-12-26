# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
from collections import Counter
import spacy
en_nlp = spacy.load('en')


def get_lab(label):
	lab = None
	if label == 'negative':
		lab = 0
	elif label == 'neutral':
		lab = 1
	elif label == 'positive':
		lab = 2
	else:
		raise ValueError("Unknown label: %s" % lab)
	return lab


def parse_xml(fname):
	print("解析xml数据")
	context, target_words, lab_list = [], [], []

	if len(fname) == 1:
		fname_1 = fname[0]
		tree_1 = ET.parse(fname_1)
		root_1 = tree_1.getroot()
		for sentence in root_1:
			aspect_terms = sentence.find('aspectTerms')
			if aspect_terms is not None:
				target, label = [], []

				for aspect_term in aspect_terms.findall('aspectTerm'):
					if aspect_term.get('polarity') == 'conflict':
						continue
						# tar = aspect_term.get('term').lower()
					target.append(en_nlp(aspect_term.get('term').lower()))
					label.append((aspect_term.get('polarity')))
				if len(target) != 0:
					text = sentence.find('text').text.lower()
					context.append(text)
					target_words.append(target)
					lab_list.append(label)

	elif len(fname) == 2:
		fname_1, fname_2 = fname[0], fname[1]
		tree_1 = ET.parse(fname_1)
		root_1 = tree_1.getroot()

		tree_2 = ET.parse(fname_2)
		root_2 = tree_2.getroot()
		for sentence in root_1:
			aspect_terms = sentence.find('aspectTerms')

			if aspect_terms is not None:
				target, label = [], []
				text = sentence.find('text').text.lower()
				context.append(text)
				for aspect_term in aspect_terms.findall('aspectTerm'):
					if aspect_term.get('polarity') == 'conflict':
						continue
					target.append(en_nlp(aspect_term.get('term').lower()))
					label.append((aspect_term.get('polarity')))
				if len(target) != 0:
					text = sentence.find('text').text.lower()
					context.append(text)
					target_words.append(target)
					lab_list.append(label)

		for review in root_2:
			sentences = review.find('sentences')
			for sentence in sentences:
				if sentence.find('Opinions') is None:
					continue
				text = sentence.find('text').text.lower()
				context.append(text)
				target, label = [], []
				for opinions in sentence.iter('Opinions'):
					for opinion in opinions:
						if opinion.get('target') == 'NULL':
							word = opinion.get('category').split('#')[0]
							target.append(en_nlp(word.lower()))
						else:
							target.append(en_nlp(opinion.get('target').lower()))
						label.append(opinion.get('polarity'))
				target_words.append(target)
				lab_list.append(label)
	else:
		raise ValueError("请输入XML文件")
	return zip(context, target_words, lab_list)


def read_data(pre_data, source_count, source_word2idx):
	print("获取词表、词典，生成source_word2idx")
	data = list(zip(*pre_data))
	text = data[0]
	aspect = data[1]
	tar_lab = data[2]

	source_words, target_words, max_sent_len, max_num_tar = [], [], 0, 0
	for i, sentence in enumerate(text):
		sptoks = en_nlp(sentence)
		source_words.extend([sp.text for sp in sptoks])
		if len(sptoks) > max_sent_len:
			max_sent_len = len(sptoks)
		for tar in aspect[i]:
			if len(tar) > max_num_tar:
				max_num_tar = len(tar)
			target_words.extend([sp.text for sp in tar])

	if len(source_count) == 0:
		source_count.append(['<pad>', 0])
	source_count.extend(Counter(source_words + target_words).most_common())

	for word, _ in source_count:
		if word not in source_word2idx:
			source_word2idx[word] = len(source_word2idx)

	source_data, target_data, target_label = [], [], []
	for i, sentence in enumerate(text):
		sptoks = en_nlp(sentence)
		if len(sptoks.text.strip()) != 0:
			idx = []
			for sptok in sptoks:
				idx.append(source_word2idx[sptok.text])
			for t_sptoks in aspect[i]:
				source_data.append(idx)
				target_data.append([source_word2idx[sp.text] for sp in t_sptoks])
			for lab in tar_lab[i]:
				target_label.append(get_lab(label=lab))
	return source_data, target_data, target_label, max_sent_len, max_num_tar


def read(fname, source_count, source_word2idx):
	pre_data = parse_xml(fname)
	data = read_data(pre_data, source_count, source_word2idx)
	print("返回清洗好的数据")
	return data
