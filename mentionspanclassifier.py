"""Mention span classifier (mention/nonmention).

Usage: mentionspanclassifier.py <train> <validation> <parsesdir>
Example: mentionspanclassifier.py 'train/*.conll' 'dev/*.conll' parses/
"""
# requirements:
# - pip install 'transformers>=4.0' keras tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

import sys
from glob import glob
from itertools import groupby
import random as python_random
from lxml import etree
import numpy as np
import keras
import tensorflow as tf
from sklearn.metrics import classification_report
from coref import (readconll, readngdata, conllclusterdict, getheadidx,
		parsesentid, Mention, getmentioncandidates,
		adjustmentionspan)
import bert

DENSE_LAYER_SIZES = [500, 150, 150]
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 100
PATIENCE = 5
LAMBD = 0.1  # L2 regularization
# the minimum score in the range [0. 1] to consider a span as mention.
# the model does not have to be retrained when this value is modified.
MENTION_THRESHOLD = 0.3
MODELFILE = 'mentionspanclassif.pt'
BERTMODEL = 'GroNLP/bert-base-dutch-cased'


def extractmentionsfromconll(conlldata, trees, ngdata, gadata):
	"""Extract gold mentions from annotated data and merge features.

	:returns: mentions sorted by sentno, mention length."""
	mentions = []
	goldspansforcluster = conllclusterdict(conlldata)
	for _clusterid, spans in goldspansforcluster.items():
		# firstment = None
		for sentno, begin, end, text in sorted(spans):
			# smallest node spanning begin, end
			(parno, _sentno), tree = trees[sentno]
			node = sorted((node for node in tree.findall('.//node')
							if begin >= int(node.get('begin'))
							and end <= int(node.get('end'))),
					key=lambda x: int(x.get('end')) - int(x.get('begin')))[0]
			headidx = getheadidx(node)
			if headidx >= end:
				headidx = max(int(x.get('begin'))
						for x in node.findall('.//node')
						if int(x.get('begin')) < end)
			mention = Mention(
					len(mentions), sentno, parno, tree, node, begin, end,
					headidx, text.split(' '), ngdata, gadata)
			mention.singleton = len(spans) == 1
			# if firstment is None:
			# 	firstment = mention
			# else:
			# 	mergefeatures(firstment, mention)
			mentions.append(mention)
	# sort by sentence, then from longest to shortest span
	mentions.sort(key=lambda x: (x.sentno, x.begin - x.end))
	for n, mention in enumerate(mentions):
		mention.id = n  # fix mention IDs after sorting
	return mentions


def loadmentions(conllfile, parsesdir, ngdata, gadata):
	# assume single document
	conlldata = next(iter(readconll(conllfile).values()))
	filenames = sorted(glob(os.path.join(parsesdir, '*.xml')), key=parsesentid)
	trees = [(parsesentid(filename), etree.parse(filename))
			for filename in filenames]
	# extract gold mentions with gold clusters
	mentions = extractmentionsfromconll(conlldata, trees, ngdata, gadata)
	return trees, mentions


class MentionDetection:
	def __init__(self):
		self.result = []  # collected feature vectors for mentions
		self.labels = []  # the target labels for the mentions
		self.spans = []  # the mention metadata

	def add(self, trees, embeddings, mentions=None):
		"""When training, mentions should be the list with the correct spans.
		"""
		# global token index
		i = 0
		idx = {}  # map (sentno, tokenno) to global token index
		for sentno, (_, tree) in enumerate(trees):
			for n, _token in enumerate(sorted(
					tree.iterfind('.//node[@word]'),
					key=lambda x: int(x.get('begin')))):
				idx[sentno, n] = i
				i += 1
		result = []
		# if given, the set of correct spans
		goldspans = {
				(mention.sentno, mention.begin, mention.end):
					(mention.node,
					int(mention.head.get('begin')),
					' '.join(mention.tokens))
				for mention in mentions or ()}
		allspans = goldspans.copy()
		# collect candidate spans;
		# if we have gold spans, only add negative examples
		for sentno, (_, tree) in enumerate(trees):
			# FIXME: getmentioncandidates extracts candidates using
			# queries on the parse tree.
			# if the parse tree has errors, this could prevent mentions
			# from being found.
			for candidate in getmentioncandidates(tree, conj=True):
				begin, end, headidx, tokens = adjustmentionspan(
						candidate, tree, relpronounsplit=True)
				if (sentno, begin, end) not in allspans and end > begin:
					allspans[sentno, begin, end] = (
							candidate, headidx, ' '.join(tokens))
				if (sentno, begin + 1, end) not in allspans and end > begin + 1:
					allspans[sentno, begin + 1, end] = (
							candidate, headidx, ' '.join(tokens[1:]))
		# group candidates by sentno and headidx
		order = sorted(allspans, key=lambda x: (x[0], allspans[x][1]))
		# collect features for spans
		for n, (sentno, begin, end) in enumerate(order, len(self.spans)):
			(node, headidx, tokens) = allspans[sentno, begin, end]
			head = (node.find('..//node[@begin="%d"][@word]' % headidx)
                                if len(node) else node)
			result.append((
				sentno, begin, end,
				# additional features
				node.get('rel') == 'su',
				node.get('rel') == 'obj1',
				node.get('rel') == 'obj2',
				# does this NP contain another NP?
				node.find('.//node[@cat="np"]') is not None,
				head.get('neclass') == 'PER',
				head.get('neclass') == 'LOC',
				head.get('neclass') == 'ORG',
				head.get('neclass') == 'MISC',
				head.get('pt') == 'n',
				head.get('pt') == 'vnw',
				head.get('pt') == 'ww',
				))
			# True == mention
			self.labels.append((sentno, begin, end) in goldspans)
			self.spans.append((sentno, headidx, begin, end, n, tokens))
		# concatenate BERT embeddings with additional features
		numotherfeats = len(result[0]) - 3
		buf = np.zeros((len(result), 2 * embeddings.shape[-1] + numotherfeats))
		for n, featvec in enumerate(result):
			# first and last BERT token representations of the mentions.
			msent, mbegin, mend = featvec[:3]
			buf[n, :embeddings.shape[-1]] = embeddings[
					idx[msent, mbegin]]
			buf[n, embeddings.shape[-1]:-numotherfeats] = embeddings[
					idx[msent, mend - 1]]
			buf[n, -numotherfeats:] = featvec[-numotherfeats:]
		self.result.append(buf)

	def getvectors(self):
		return (np.vstack(self.result),
				np.array(self.labels),
				self.spans)


def getfeatures(pattern, parsesdir, tokenizer, bertmodel):
	data = MentionDetection()
	files = glob(pattern)
	if not files:
		raise ValueError('pattern did not match any files: %s' % pattern)
	ngdata, gadata = readngdata()
	for n, conllfile in enumerate(files, 1):
		parses = os.path.join(parsesdir,
				os.path.basename(conllfile.rsplit('.', 1)[0]))
		trees, mentions = loadmentions(conllfile, parses, ngdata, gadata)
		embeddings = bert.getvectors(parses, trees, tokenizer, bertmodel)
		data.add(trees, embeddings, mentions)
		print(f'encoded {n}/{len(files)}: {conllfile}', file=sys.stderr)
	X, y, spans = data.getvectors()
	return X, y, spans


def build_mlp_model(input_shape, num_labels):
	"""Define a binary classifier."""
	model = keras.Sequential([
			keras.Input(shape=input_shape),
			keras.layers.Dropout(DROPOUT_RATE),

			keras.layers.Dense(DENSE_LAYER_SIZES[0], name='dense0'),
			keras.layers.BatchNormalization(name='bn0'),
			keras.layers.Activation('relu'),
			keras.layers.Dropout(DROPOUT_RATE),

			keras.layers.Dense(DENSE_LAYER_SIZES[1], name='dense1'),
			keras.layers.BatchNormalization(name='bn1'),
			keras.layers.Activation('relu'),
			keras.layers.Dropout(DROPOUT_RATE),

			# keras.layers.Dense(DENSE_LAYER_SIZES[2], name='dense2'),
			# keras.layers.BatchNormalization(name='bn2'),
			# keras.layers.Activation('relu'),
			# keras.layers.Dropout(DROPOUT_RATE),

			keras.layers.Dense(
				num_labels, name='output',
				kernel_regularizer=keras.regularizers.l2(LAMBD)),
			keras.layers.Activation('sigmoid'),
			])
	return model


def train(trainfiles, validationfiles, parsesdir, tokenizer, bertmodel):
	np.random.seed(1)
	python_random.seed(1)
	tf.random.set_seed(1)
	X_train, y_train, _mentions = getfeatures(
			trainfiles, parsesdir, tokenizer, bertmodel)
	X_val, y_val, _mentions = getfeatures(
			validationfiles, parsesdir, tokenizer, bertmodel)
	print('training data', X_train.shape)
	print('validation data', X_val.shape)
	classif_model = build_mlp_model([X_train.shape[-1]], 1)
	classif_model.summary()
	classif_model.compile(
			optimizer=keras.optimizers.Adam(lr=LEARNING_RATE),
			loss='binary_crossentropy')
	callbacks = [
			keras.callbacks.EarlyStopping(
				monitor='val_loss', patience=PATIENCE,
				restore_best_weights=True),
			keras.callbacks.ModelCheckpoint(
				MODELFILE, monitor='val_loss', verbose=0,
				save_best_only=True, mode='min',
				save_weights_only=True),
			]
	classif_model.fit(x=X_train, y=y_train, epochs=EPOCHS,
			batch_size=BATCH_SIZE, callbacks=callbacks,
			validation_data=(X_val, y_val), verbose=1)


def evaluate(validationfiles, parsesdir, tokenizer, bertmodel):
	X_val, y_val, spans = getfeatures(
			validationfiles, parsesdir, tokenizer, bertmodel)
	model = build_mlp_model([X_val.shape[-1]], 1)
	model.load_weights(MODELFILE).expect_partial()
	probs = model.predict(X_val)
	for (sentno, headidx, begin, end, _n, text), pred, gold in zip(
			spans, probs[:, 0], y_val):
		print(f'predict/actual={int(pred > MENTION_THRESHOLD)}/{int(gold)}, '
				f'p={pred:.3f} {sentno:3} {headidx:2} {begin:2} {end:2} {text}')
	print()
	# simple evaluation: classify each span independently
	print('independent evaluation:')
	print(classification_report(
			np.array(y_val, dtype=bool),
			probs[:, 0] > 0.5,  # MENTION_THRESHOLD,
			target_names=['nonmention', 'mention']))
	# better evaluation: pick best span from candidates with same head
	result = np.zeros(len(spans), dtype=bool)
	# group candidates by (sentno, headidx)
	for _, candidates in groupby(spans, key=lambda x: (x[0], x[1])):
		candidates = list(candidates)
		a, b = candidates[0][4], candidates[-1][4] + 1
		best = a + probs[a:b, 0].argmax()
		if probs[best, 0] > MENTION_THRESHOLD:
			result[best] = True
	print('best mention for each head:')
	print(classification_report(
			np.array(y_val, dtype=bool),
			result,
			target_names=['nonmention', 'mention']))


def predict(trees, embeddings, ngdata, gadata):
	"""Load mention classfier, get candidate mentions, and return predicted
	mentions."""
	data = MentionDetection()
	data.add(trees, embeddings)
	X, _y, spans = data.getvectors()
	model = build_mlp_model([X.shape[-1]], 1)
	model.load_weights(MODELFILE).expect_partial()
	probs = model.predict(X)
	mentions = []
	for _, candidates in groupby(spans, key=lambda x: (x[0], x[1])):
		candidates = list(candidates)
		a, b = candidates[0][4], candidates[-1][4] + 1
		best = probs[a:b, 0].argmax()
		if probs[a + best, 0] <= MENTION_THRESHOLD:
			continue
		sentno, headidx, begin, end, _n, text = candidates[best]
		# smallest node spanning begin, end
		(parno, _sentno), tree = trees[sentno]
		node = min((node for node in tree.findall('.//node')
					if begin >= int(node.get('begin'))
					and end <= int(node.get('end'))),
				key=lambda x: int(x.get('end')) - int(x.get('begin')))
		mentions.append(Mention(
				len(mentions), sentno, parno, tree, node, begin, end, headidx,
				text.split(' '), ngdata, gadata))
	return mentions


def main():
	"""CLI."""
	tokenizer, bertmodel = bert.loadmodel(BERTMODEL)
	_, trainfiles, validationfiles, parsesdir = sys.argv
	train(trainfiles, validationfiles, parsesdir, tokenizer, bertmodel)
	evaluate(validationfiles, parsesdir, tokenizer, bertmodel)


if __name__ == '__main__':
	main()
