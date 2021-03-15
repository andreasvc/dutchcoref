"""Mention feature classifier.

Usage: mentionfeatureclassifier.py <train> <validation> <parsesdir>
Example: mentionfeatureclassifier.py train/*.conll dev/*.conll parses/
"""
# requirements:
# - pip install 'transformers>=4.0' keras tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

import sys
from glob import glob
from lxml import etree
import numpy as np
import keras
from sklearn.metrics import classification_report
from coref import (readconll, readngdata, conllclusterdict, getheadidx,
		parsesentid, Mention, mergefeatures, gettokens)
import bert

DENSE_LAYER_SIZES = [500, 150, 150]
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 100
PATIENCE = 15
LAMBD = 0.1  # L2 regularization
MODELFILE = 'mentionfeatclassif.pt'
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


class MentionFeatures:
	def __init__(self, tokenizer, bertmodel):
		self.result = []  # collected feature vectors for mentions
		self.labels = []  # the target labels for the mentions
		self.mentions = []  # the mention objects
		self.tokenizer = tokenizer
		self.bertmodel = bertmodel

	def add(self, trees, mentions):
		result = []
		# collect mention features
		for n, mention in enumerate(mentions):
			# feature indicators: ['nh', 'h', 'f', 'm', 'n']
			# multiple values can be True!
			label = np.zeros(5)
			label[0] = mention.features['human'] == 0
			label[1] = mention.features['human'] == 1
			label[2] = 'f' in (mention.features['gender'] or '')
			label[3] = 'm' in (mention.features['gender'] or '')
			label[4] = 'n' in (mention.features['gender'] or '')
			self.labels.append(label)
			self.mentions.append(mention)
			# collecting additional features
			# FIXME: feature: is mention part of another mention?
			result.append((
					mention.sentno, mention.begin, mention.end,
					# additional features
					mention.node.get('rel') == 'su',
					))
		# now use BERT to obtain vectors for the text of these mentions
		sentences = [gettokens(tree, 0, 9999) for _, tree in trees]
		# NB: this encodes each sentence independently
		vectors = bert.encode_sentences(
				sentences, self.tokenizer, self.bertmodel)
		buf = np.zeros((len(result), vectors[0].shape[-1]))
		# concatenate BERT embeddings with additional features
		numotherfeats = len(result[0]) - 3
		buf = np.zeros((len(result), vectors[0].shape[-1] + numotherfeats))
		for n, featvec in enumerate(result):
			# mean of BERT token representations of the tokens in the mentions.
			msent, mbegin, mend = featvec[:3]
			buf[n, :vectors[0].shape[-1]] = vectors[
					msent][mbegin:mend].mean(axis=0)
			buf[n, -numotherfeats:] = featvec[-numotherfeats:]
		self.result.append(buf)

	def getvectors(self):
		return (np.vstack(self.result),
				np.vstack(self.labels),
				self.mentions)


def getfeatures(files, parsesdir, cachefile, tokenizer, bertmodel):
	# NB: assumes the input files don't change;
	# otherwise, manually delete cached file!
	if os.path.exists(cachefile):
		with open(cachefile, 'rb') as inp:
			X = np.load(inp)
			y = np.load(inp)
			mentions = np.load(inp, allow_pickle=True)
	else:
		data = MentionFeatures(tokenizer, bertmodel)
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
			keras.layers.Dropout(DROPOUT_RATE, seed=7),

			keras.layers.Dense(DENSE_LAYER_SIZES[0], name='dense0'),
			keras.layers.BatchNormalization(name='bn0'),
			keras.layers.Activation('relu'),
			keras.layers.Dropout(DROPOUT_RATE, seed=7),

			keras.layers.Dense(DENSE_LAYER_SIZES[1], name='dense1'),
			keras.layers.BatchNormalization(name='bn1'),
			keras.layers.Activation('relu'),
			keras.layers.Dropout(DROPOUT_RATE, seed=7),

			# keras.layers.Dense(DENSE_LAYER_SIZES[2], name='dense2'),
			# keras.layers.BatchNormalization(name='bn2'),
			# keras.layers.Activation('relu'),
			# keras.layers.Dropout(DROPOUT_RATE, seed=7),

			keras.layers.Dense(
				num_labels, name='output',
				kernel_regularizer=keras.regularizers.l2(LAMBD)),
			keras.layers.Activation('sigmoid'),
			])
	return model


def train(trainfiles, validationfiles, parsesdir, tokenizer, bertmodel):
	X_train, y_train, _mentions = getfeatures(
			trainfiles, parsesdir, 'mentfeattrain.npy', tokenizer, bertmodel)
	X_val, y_val, _mentions = getfeatures(
			validationfiles, parsesdir, 'mentfeatval.npy', tokenizer, bertmodel)
	print('training data', X_train.shape)
	print('validation data', X_val.shape)
	classif_model = build_mlp_model([X_train.shape[-1]], y_val.shape[-1])
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
	X_val, y_val, mentions = getfeatures(
			validationfiles, parsesdir, 'mentfeatval.npy', tokenizer, bertmodel)
	model = build_mlp_model([X_val.shape[-1]], y_val.shape[-1])
	model.load_weights(MODELFILE).expect_partial()
	probs = model.predict(X_val)
	print(classification_report(
			np.array(y_val, dtype=bool),
			np.array([a > 0.5 for a in probs], dtype=bool),
			target_names=['nonhuman', 'human', 'female', 'male', 'neuter']))


def predict(trees, mentions):
	"""Load mentions classfier, get features for mentions, and update features
	of mentions."""
	tokenizer, bertmodel = bert.loadmodel(BERTMODEL)
	data = MentionFeatures(tokenizer, bertmodel)
	data.add(trees, mentions)
	X, y, mentions = data.getvectors()
	model = build_mlp_model([X.shape[-1]], y.shape[-1])
	model.load_weights(MODELFILE).expect_partial()
	probs = model.predict(X)
	for row, mention in zip(probs, mentions):
		if row[0] > 0.5 and row[1] < 0.5:
			mention.features['human'] = 0
		if row[0] < 0.5 and row[1] > 0.5:
			mention.features['human'] = 1
		gend = ''
		if row[2] > 0.5:
			gend += 'f'
		if row[3] > 0.5:
			gend += 'm'
		if row[4] > 0.5:
			gend += 'n'
		if gend:
			mention.features['gender'] = gend


def main():
	"""CLI."""
	tokenizer, bertmodel = bert.loadmodel(BERTMODEL)
	_, trainfiles, validationfiles, parsesdir = sys.argv
	train(trainfiles, validationfiles, parsesdir, tokenizer, bertmodel)
	evaluate(validationfiles, parsesdir, tokenizer, bertmodel)


if __name__ == '__main__':
	main()
