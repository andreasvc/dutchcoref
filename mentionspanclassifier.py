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
import random as python_random
from lxml import etree
import numpy as np
import keras
import tensorflow as tf
from sklearn.metrics import classification_report
from coref import (readconll, readngdata, conllclusterdict, getheadidx,
		parsesentid, Mention, mergefeatures, gettokens, getmentioncandidates,
		adjustmentionspan)
import bert

DENSE_LAYER_SIZES = [500, 150, 150]
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 100
PATIENCE = 15
LAMBD = 0.1  # L2 regularization
MODELFILE = 'mentionspanclassif.pt'
BERTMODEL = 'GroNLP/bert-base-dutch-cased'


def extractmentionsfromconll(conlldata, trees, ngdata, gadata):
	"""Extract gold mentions from annotated data and merge features.

	:returns: mentions sorted by sentno, begin; including gold clusterid
		and detected features for the cluster."""
	mentions = []
	goldspansforcluster = conllclusterdict(conlldata)
	for clusterid, spans in goldspansforcluster.items():
		firstment = None
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
			if firstment is None:
				firstment = mention
			else:
				mergefeatures(firstment, mention)
			mentions.append(mention)
	# sort by sentence, then from longest to shortest span
	mentions.sort(key=lambda x: (x.sentno, x.begin - x.end))
	for n, mention in enumerate(mentions):
		mention.id = n  # fix mention IDs after sorting
	return mentions


def loadmentions(conllfile, parsesdir):
    ngdata, gadata = readngdata()
    # assume single document
    conlldata = next(iter(readconll(conllfile).values()))
    filenames = sorted(glob(os.path.join(parsesdir, '*.xml')), key=parsesentid)
    trees = [(parsesentid(filename), etree.parse(filename))
            for filename in filenames]
    # extract gold mentions with gold clusters
    mentions = extractmentionsfromconll(conlldata, trees, ngdata, gadata)
    return trees, mentions


class MentionDetection:
	def __init__(self, tokenizer, bertmodel):
		self.result = []  # collected feature vectors for mentions
		self.labels = []  # the target labels for the mentions
		self.spans = []  # the mention metadata
		self.tokenizer = tokenizer
		self.bertmodel = bertmodel

	def add(self, trees, mentions=None, embeddings=None):
		result = []
		goldspans = {
				(mention.sentno, mention.begin, mention.end): mention
				for mention in mentions or ()}
		# add all correct mentions:
		for (sentno, begin, end), mention in goldspans.items():
			result.append((
				sentno, begin, end,
				# additional features
				mention.node.get('rel') == 'su',
				))
			self.labels.append(True)  # True = mention
			self.spans.append((sentno, begin, end, ' '.join(mention.tokens)))
		# add other mention candidates that can be extracted
		# from the parse tree
		allspans = set(goldspans)
		for sentno, (_, tree) in enumerate(trees):
			for candidate in getmentioncandidates(tree):
				begin, end, _headidx, tokens = adjustmentionspan(
						candidate, tree, True)
				if (sentno, begin, end) not in allspans and end > begin:
					allspans.add((sentno, begin, end))
					result.append((
						sentno, begin, end,
						# additional features (should be same as above)
						candidate.get('rel') == 'su',
						))
					self.labels.append(False)  # False = nonmention
					self.spans.append((sentno, begin, end,
							' '.join(gettokens(tree, begin, end))))
		if embeddings is None:
			# now use BERT to obtain vectors for the text of these spans
			sentences = [gettokens(tree, 0, 9999) for _, tree in trees]
			# NB: this encodes each sentence independently
			embeddings = bert.encode_sentences(
					sentences, self.tokenizer, self.bertmodel)
		buf = np.zeros((len(result), 2 * embeddings[0].shape[-1]))
		# concatenate BERT embeddings with additional features
		numotherfeats = len(result[0]) - 3
		buf = np.zeros((len(result), embeddings[0].shape[-1] + numotherfeats))
		for n, featvec in enumerate(result):
			# first and last BERT token representations of the mentions.
			msent, mbegin, mend = featvec[:3]
			buf[n, :embeddings[0].shape[-1]] = embeddings[
                    msent][mbegin]
			buf[n, embeddings[0].shape[-1]:-numotherfeats] = embeddings[
					msent][mend - 1].mean(axis=0)
			buf[n, -numotherfeats:] = featvec[-numotherfeats:]
		self.result.append(buf)

	def getvectors(self):
		return (np.vstack(self.result),
				np.array(self.labels),
				self.spans)


def getfeatures(files, parsesdir, cachefile, tokenizer, bertmodel):
	# NB: assumes the input files don't change;
	# otherwise, manually delete cached file!
	if os.path.exists(cachefile):
		with open(cachefile, 'rb') as inp:
			X = np.load(inp)
			y = np.load(inp)
			mentions = np.load(inp, allow_pickle=True)
	else:
		data = MentionDetection(tokenizer, bertmodel)
		files = glob(files)
		for n, conllfile in enumerate(files, 1):
			parses = os.path.join(parsesdir,
					os.path.basename(conllfile.rsplit('.', 1)[0]))
			trees, mentions = loadmentions(conllfile, parses)
			data.add(trees, mentions)
			print(f'encoded {n}/{len(files)}: {conllfile}', file=sys.stderr)
		X, y, mentions = data.getvectors()
		with open(cachefile, 'wb') as out:
			np.save(out, X)
			np.save(out, y)
			np.save(out, mentions)
	return X, y, mentions


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
			trainfiles, parsesdir, 'mentspantrain.npy', tokenizer, bertmodel)
	X_val, y_val, _mentions = getfeatures(
			validationfiles, parsesdir, 'mentspanval.npy', tokenizer, bertmodel)
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
			validationfiles, parsesdir, 'mentspanval.npy', tokenizer, bertmodel)
	model = build_mlp_model([X_val.shape[-1]], 1)
	model.load_weights(MODELFILE).expect_partial()
	probs = model.predict(X_val)
	for (sentno, begin, end, text), pred, gold in zip(
			spans, probs[:, 0], y_val):
		print(f'actual={int(gold)}, pred={int(pred > 0.5)}, p={pred:.3f} '
				f'{sentno:3} {begin:2} {end:2} {text}')
	print()
	print(classification_report(
			np.array(y_val, dtype=bool),
			probs[:, 0] > 0.5,
			target_names=['nonmention', 'mention']))


def predict(trees, embeddings, ngdata, gadata):
	"""Load mention classfier, get candidate mentions, and return predicted
	mentions."""
	tokenizer = bertmodel = None
	data = MentionDetection(tokenizer, bertmodel)
	data.add(trees, embeddings=embeddings)
	X, _y, spans = data.getvectors()
	model = build_mlp_model([X.shape[-1]], 1)
	model.load_weights(MODELFILE).expect_partial()
	probs = model.predict(X)
	mentions = []
	for (sentno, begin, end, text), pred in zip(spans, probs[:, 0] > 0.5):
		if not pred:
			continue
		# smallest node spanning begin, end
		(parno, _sentno), tree = trees[sentno]
		node = min((node for node in tree.findall('.//node')
					if begin >= int(node.get('begin'))
					and end <= int(node.get('end'))),
				key=lambda x: int(x.get('end')) - int(x.get('begin')))
		headidx = getheadidx(node)
		if headidx >= end:
			headidx = max(int(x.get('begin')) for x in node.findall('.//node')
					if int(x.get('begin')) < end)
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
