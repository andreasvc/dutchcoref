"""Train pronoun resolution model.

Usage: pronounresolution.py <train> <validation> <parsesdir>
Example: pronounresolution.py train/*.conll dev/*.conll parses/
"""
# requirements:
# - pip install 'transformers>=4.0' keras tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

import sys
# from collections import Counter
from glob import glob
from lxml import etree
from sklearn import metrics
import numpy as np
import keras
from coref import (readconll, parsesentid, readngdata,
		conllclusterdict, getheadidx, Mention, gettokens)
import bert

# NB: if MAXPRONDIST is changed, delete the .npy files
MAXPRONDIST = 10  # max number of mentions between pronoun and candidate
DENSE_LAYER_SIZES = [500, 150, 150]
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 100
PATIENCE = 15
LAMBD = 0.1  # L2 regularization
MENTION_SCORE_THRESHOLD = 0.01
MODELFILE = 'pronounmodel.pt'
BERTMODEL = 'GroNLP/bert-base-dutch-cased'


def extractmentionsfromconll(conlldata, trees, ngdata, gadata):
	"""Extract gold mentions from annotated data.

	:returns: mentions sorted by sentno, begin; including gold clusterid."""
	mentions = []
	goldspansforcluster = conllclusterdict(conlldata)
	goldspans = {(clusterid, ) + span
			for clusterid, spans in goldspansforcluster.items()
				for span in spans}
	for clusterid, sentno, begin, end, text in sorted(goldspans):
		# smallest node spanning begin, end
		(parno, _sentno), tree = trees[sentno]
		node = sorted((node for node in tree.findall('.//node')
					if begin >= int(node.get('begin'))
					and end <= int(node.get('end'))),
				key=lambda x: int(x.get('end')) - int(x.get('begin')))[0]
		headidx = getheadidx(node)
		if headidx >= end:
			headidx = max(int(x.get('begin')) for x in node.findall('.//node')
					if int(x.get('begin')) < end)
		mention = Mention(
				len(mentions), sentno, parno, tree, node, begin, end, headidx,
				text.split(' '), ngdata, gadata)
		mention.clusterid = clusterid
		mentions.append(mention)
	mentions.sort(key=lambda x: (x.sentno, x.begin))
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


def checkfeat(mention, other, key):
	"""True if feature `key` of mention and other are compatible."""
	return (mention.features[key] == other.features[key]
			or None in (mention.features[key], other.features[key])
			or (key == 'gender'
				and 'fm' in (mention.features[key], other.features[key])
				and 'n' not in (mention.features[key], other.features[key]))
			or (key == 'gender'
				and 'mn' in (mention.features[key], other.features[key])
				and 'f' not in (mention.features[key], other.features[key]))
			or (key == 'gender'
				and 'fn' in (mention.features[key], other.features[key])
				and 'm' not in (mention.features[key], other.features[key]))
			or (key == 'number'
				and 'both' in (mention.features[key], other.features[key])))


class CorefFeatures:
	def __init__(self, tokenizer, bertmodel):
		self.result = []  # collected features for pairs
		self.coreferent = []  # the target: pair is coreferent (1) or not (0)
		self.antecedents = []  # the candidate antecedent mention in each pair
		self.anaphordata = []  # row indices for each anaphor and its candidates
		self.tokenizer = tokenizer
		self.bertmodel = bertmodel

	def add(self, trees, mentions):
		# global token index
		i = 0
		idx = {}  # map (sentno, tokenno) to global token index
		for sentno, ((parno, psentno), tree) in enumerate(trees):
			for n, token in enumerate(sorted(
					tree.iterfind('.//node[@word]'),
					key=lambda x: int(x.get('begin')))):
				idx[sentno, n] = i
				i += 1
		result = []
		# collect mentions and candidate antecedents
		for n, mention in enumerate(mentions):
			if (mention.type == 'pronoun'
					and mention.node.get('vwtype') != 'betr'  # no rel. pronouns
					and mention.features['person'] not in ('1', '2')):

				a = len(self.coreferent)
				nn = n - 1
				# determine the candidates using a window of sentences;
				# while (nn > 0
				# 		and mention.sentno - mentions[nn].sentno < MAXPRONDIST):
				# 	nn -= 1
				# determine candidates using window of words
				# while (nn > 0
				# 		and idx[mention.sentno, mention.begin]
				# 		- idx[mentions[nn].sentno, mentions[nn].begin]
				# 		< MAXPRONDIST):
				# 	nn -= 1
				# determine candidates using number of mentions
				nn = max(n - MAXPRONDIST, 0)

				# FIXME: encode context of each mention as a single segment
				# instead of each sentence independently?
				# rng = range(mentions[nn].sentno, mention.sentno + 1)
				# sentences = [gettokens(trees[x][1], 0, 9999) for x in rng]
				# vectors = dict(zip(rng,
				# 		bert.encode_sentences(
				# 			sentences, self.tokenizer, self.bertmodel)))

				# how frequent is this mention in the context?
				# freq = Counter(other.clusterid for other in mentions[nn:n])
				for other in mentions[nn + 1:n][::-1]:
					iscoreferent = mention.clusterid == other.clusterid
					self.coreferent.append(iscoreferent)
					self.antecedents.append(other)
					# FIXME: feature: is mention part of another mention?
					# FIXME: distance features should be encoded as a histogram
					# FIXME: 'salience': how frequent is antecedent in current
					# current context; but this means previous mentions must
					# have already been resolved.
					result.append((
							mention.sentno, mention.begin, mention.end,
							other.sentno, other.begin, other.end,
							len(other.tokens),  # antecedent mention width
							mention.sentno - other.sentno,  # sent distance
							idx[mention.sentno, mention.begin]
								- idx[other.sentno, other.begin],  # word distance
							checkfeat(mention, other, 'gender'),  # compatible feature
							checkfeat(mention, other, 'human'),  # compatible feature
							checkfeat(mention, other, 'number'),  # compatible feature
							# freq[other.clusterid] / sum(freq.values()).  # salience
							))
					nn -= 1
				self.anaphordata.append((a, len(self.coreferent), mention))
		# now use BERT to obtain vectors for the text of these mentions
		sentences = [gettokens(tree, 0, 9999) for _, tree in trees]
		# NB: this encodes each sentence independently
		vectors = bert.encode_sentences(
				sentences, self.tokenizer, self.bertmodel)
		numotherfeats = len(result[0]) - 6
		buf = np.zeros((len(result), 2 * vectors[0].shape[-1] + numotherfeats))
		for n, featvec in enumerate(result):
			# mean of BERT token representations of the tokens in the mentions.
			msent, mbegin, mend = featvec[:3]
			osent, obegin, oend = featvec[3:6]
			buf[n, :vectors[0].shape[-1]] = vectors[
					msent][mbegin:mend].mean(axis=0)
			buf[n, vectors[0].shape[-1]:-numotherfeats] = vectors[
					osent][obegin:oend].mean(axis=0)
			buf[n, -numotherfeats:] = featvec[-numotherfeats:]
		self.result.append(buf)

	def getvectors(self):
		return (np.vstack(self.result),
				np.array(self.coreferent, dtype=int),
				self.antecedents,
				self.anaphordata)


def getfeatures(files, parsesdir, cachefile, tokenizer, bertmodel):
	# NB: assumes the input files don't change;
	# otherwise, manually delete cached file!
	if os.path.exists(cachefile):
		with open(cachefile, 'rb') as inp:
			X = np.load(inp)
			y = np.load(inp)
			antecedents = np.load(inp, allow_pickle=True)
			anaphordata = np.load(inp, allow_pickle=True)
	else:
		data = CorefFeatures(tokenizer, bertmodel)
		files = glob(files)
		for n, conllfile in enumerate(files, 1):
			parses = os.path.join(parsesdir,
					os.path.basename(conllfile.rsplit('.', 1)[0]))
			trees, mentions = loadmentions(conllfile, parses)
			data.add(trees, mentions)
			print(f'encoded {n}/{len(files)}: {conllfile}', file=sys.stderr)
		X, y, antecedents, anaphordata = data.getvectors()
		with open(cachefile, 'wb') as out:
			np.save(out, X)
			np.save(out, y)
			np.save(out, antecedents)
			np.save(out, anaphordata)
	return X, y, antecedents, anaphordata


def build_mlp_model(input_shape):
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
				1, name='output',
				kernel_regularizer=keras.regularizers.l2(LAMBD)),
			keras.layers.Activation('sigmoid'),
			])
	return model


def train(trainfiles, validationfiles, parsesdir, tokenizer, bertmodel):
	X_train, y_train, _clusters, _indices = getfeatures(
			trainfiles, parsesdir, 'prontrain.npy', tokenizer, bertmodel)
	X_val, y_val, _clusters, _indices = getfeatures(
			validationfiles, parsesdir, 'pronval.npy', tokenizer, bertmodel)
	print('training data', X_train.shape)
	print('validation data', X_val.shape)

	classif_model = build_mlp_model([X_train.shape[-1]])
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
	X_val, y_val, antecedents, anaphordata = getfeatures(
			validationfiles, parsesdir, 'pronval.npy', tokenizer, bertmodel)
	model = build_mlp_model([X_val.shape[-1]])
	model.load_weights(MODELFILE).expect_partial()

	probs = model.predict(X_val)
	pred = probs > MENTION_SCORE_THRESHOLD
	print('(pronoun, candidate) pair classification scores:')
	print(metrics.classification_report(y_val, pred))  # mention-pair scores

	# The above are scores for mention pairs. To get actual pronoun accuracy,
	# select a best candidate for each pronoun and evaluate on that.
	# NB: if none of the candidates is likely enough, predict -1.
	y_true = []
	pred = []
	for a, b, anaphor in anaphordata:
		# To select the best prediction for each pronoun, we use extra
		# metadata. a and b are row indices in X_train with all the candidates
		# for a single pronoun.
		# anaphor: the pronoun mention which needs to be resolved;
		#	if this mention was loaded from annotated data, it has the correct
		#   cluster for this pronoun
		# the list 'antecedents' has the corresponding antecedent mention for
		# each row in X_train.
		if a == b:  # a pronoun with no candidates ...
			continue
		# select closest predicted antecedent candidate
		# predlabel = list(probs[a:b] > MENTION_SCORE_THRESHOLD)
		# pred.append(antecedents[a:][predlabel.index(1)].clusterid
		# 		if 1 in predlabel else -1)
		# select most likely antecedent
		antecedent = antecedents[a:][probs[a:b].argmax()]
		pred.append(antecedent.clusterid
				if probs[a:b].max() > MENTION_SCORE_THRESHOLD else -1)
		y_true.append(anaphor.clusterid)
		print(' '.join(anaphor.tokens), '->',
				(' '.join(antecedent.tokens)
				if probs[a:b].max() > MENTION_SCORE_THRESHOLD else '(none)'))
	print('Pronoun resolution accuracy:',
			100 * metrics.accuracy_score(y_true, pred))


def predict(trees, mentions):
	"""Load pronoun resolver, get features for trees, and return a list of
	mention pairs (anaphor, antecedent) which are predicted to be
	coreferent."""
	tokenizer, bertmodel = bert.loadmodel(BERTMODEL)
	data = CorefFeatures(tokenizer, bertmodel)
	data.add(trees, mentions)
	X, y, antecedents, anaphordata = data.getvectors()
	model = build_mlp_model([X.shape[-1]])
	model.load_weights(MODELFILE).expect_partial()
	probs = model.predict(X)
	result = []
	for a, b, anaphor in anaphordata:
		if a == b:  # a pronoun with no candidates ...
			continue
		# select most likely antecedent
		if probs[a:b].max() > MENTION_SCORE_THRESHOLD:
			antecedent = antecedents[a:][probs[a:b].argmax()]
			result.append((anaphor, antecedent))
	return result


def main():
	tokenizer, bertmodel = bert.loadmodel(BERTMODEL)
	_, trainfiles, validationfiles, parsesdir = sys.argv
	train(trainfiles, validationfiles, parsesdir, tokenizer, bertmodel)
	evaluate(validationfiles, parsesdir, tokenizer, bertmodel)


if __name__ == '__main__':
	main()
