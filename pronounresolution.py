"""Train pronoun resolution model.

Usage: pronounresolution.py <train> <validation> <parsesdir>
Example: pronounresolution.py 'train/*.conll' 'dev/*.conll' parses/

Options:
    --restrict=N    restrict training data to the first N% of each file.
    --eval=<test>   report evaluation on this set instead of validation
"""
# requirements:
# - pip install 'transformers>=4.0' keras tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

import sys
import getopt
# from collections import Counter
from glob import glob
from lxml import etree
import random as python_random
from sklearn import metrics
import numpy as np
import keras
import tensorflow as tf
from coref import (readconll, parsesentid, readngdata, initialsegment,
		extractmentionsfromconll, sameclause, debug, VERBOSE)
import bert

PRONDISTTYPE = 'words'
MAXPRONDIST = 100  # max number of words between pronoun and candidate
DENSE_LAYER_SIZES = [500, 150, 150]
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 100
PATIENCE = 5
LAMBD = 0.1  # L2 regularization

# do not link anaphor if all scores of candidates are below this value.
# the model does not have to be re-trained if this value is changed.
MENTION_PAIR_THRESHOLD = 0.2
MODELFILE = 'pronounmodel.pt'
BERTMODEL = 'GroNLP/bert-base-dutch-cased'


def loadmentions(conllfile, parsesdir, restrict=None):
	ngdata, gadata = readngdata()
	# assume single document
	conlldata = next(iter(readconll(conllfile).values()))
	if restrict:
		n = initialsegment(conllfile, restrict)
		conlldata = conlldata[:n]
	pattern = os.path.join(parsesdir, '*.xml')
	filenames = sorted(glob(pattern), key=parsesentid)
	if not filenames:
		raise ValueError('parse trees not found: %s' % pattern)
	trees = [(parsesentid(filename), etree.parse(filename))
			for filename in filenames]
	# extract gold mentions with gold clusters
	mentions = extractmentionsfromconll(conlldata, trees, ngdata, gadata,
			goldclusters=True)
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
	def __init__(self, selectpref='data/inspect-dep35-sort.csv.gz'):
		self.result = []  # collected features for pairs
		self.coreferent = []  # the target: pair is coreferent (1) or not (0)
		self.antecedents = []  # the candidate antecedent mention in each pair
		self.anaphordata = []  # row indices for each anaphor and its candidates
		# Get selectional preferences
		# import pandas as pd
		# self.selectdf = pd.read_csv(
		# 		selectpref, index_col=['hdword-rel-depword'],
		# 		dtype={'npmi': int})

	def add(self, trees, embeddings, mentions):
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
		# globalfreq = Counter(other.clusterid for other in mentions)
		# collect mentions and candidate antecedents
		for n, mention in enumerate(mentions):
			if (mention.type == 'pronoun'
					# no relative/reciprocal/reflexive pronouns
					and mention.node.get('vwtype') not in (
						'betr', 'recip', 'refl')
					and mention.features['person'] not in ('1', '2')):

				a = len(self.coreferent)
				nn = n - 1
				# determine the candidates using a window of sentences;
				if PRONDISTTYPE == 'sents':
					while (nn > 0 and (mention.sentno - mentions[nn].sentno)
								< MAXPRONDIST):
						nn -= 1
				# determine candidates using window of words
				elif PRONDISTTYPE == 'words':
					while (nn > 0
							and idx[mention.sentno, mention.begin]
							- idx[mentions[nn].sentno, mentions[nn].begin]
							< MAXPRONDIST):
						nn -= 1
				# determine candidates using number of mentions
				elif PRONDISTTYPE == 'mentions':
					nn = max(n - MAXPRONDIST, 0)
				else:
					raise ValueError('PRONDISTTYPE should be one of %r; got %r'
							% ({'sents', 'words', 'mentions'}, PRONDISTTYPE))
				# FIXME: encode context of each mention as a single segment
				# instead of each sentence independently?
				# rng = range(mentions[nn].sentno, mention.sentno + 1)
				# sentences = [gettokens(trees[x][1], 0, 9999) for x in rng]
				# vectors = dict(zip(rng,
				# 		bert.encode_sentences(
				# 			sentences, self.tokenizer, self.bertmodel)))

				# how frequent is this mention in the context?
				# freq = Counter(other.clusterid for other in mentions[nn:n])
				# consider all candidates, but in reverse order
				# (closest antecedent first)
				for m, other in list(enumerate(mentions[nn:n],
						nn))[::-1]:
					if other.node.get('rel') in ('app', 'det'):
						continue
					# The antecedent should come before the anaphor,
					# and should not contain the anaphor.
					if (other.sentno == mention.sentno
							and (other.begin >= mention.begin
								# allow: [de raket met [haar] massa van 750 ton]
								or (mention.head.get('vwtype') != 'bez'
									and other.end >= mention.end))):
						continue
					if (mention.head.get('vwtype') != 'bez'  # co-arguments
							and sameclause(other.node, mention.node)
							and other.node.find('..//node[@id="%s"]'
							% mention.node.get('id')) is not None):
						continue
					iscoreferent = mention.clusterid == other.clusterid
					self.coreferent.append(iscoreferent)
					self.antecedents.append(other)
					# FIXME: feature: is mention part of another mention?
					# FIXME: 'salience features: how frequent is antecedent
					# entity in current context or in the whole document;
					# this means previous mentions must have already been
					# resolved.
					# selpref = self.selectprefscore(mention, other)
					feats = (
							mention.sentno, mention.begin, mention.end,
							other.sentno, other.begin, other.end,
							mention.parentheadwordidx,
							other.type == 'pronoun',
							other.type == 'noun',
							other.type == 'name',
							mention.head.get('rel') == other.head.get('rel'),
							# feature compatibility
							checkfeat(mention, other, 'gender'),
							checkfeat(mention, other, 'human'),
							checkfeat(mention, other, 'number'),
							mention.features['person'] == '3',
							other.features['person'] == '1',
							other.features['person'] == '2',
							other.features['person'] == '3',
							other.features['person'] is not None
								and (mention.features['person']
									!= other.features['person']),
							# is mention part of direct speech?
							mention.head.get('quotelabel') == 'O',
							other.head.get('quotelabel') == 'O',
							# number of times the cluster of this antecedent
							# occurs in the candidates
							# freq[other.clusterid] / sum(freq.values()),
							# number of mentions in antecedent cluster
							# in whole document
							# globalfreq[other.clusterid]
							# 	/ sum(globalfreq.values()),
							# selpref == 0, 0 < selpref < 0.2,
							# 0.2 < selpref < 0.4, 0.4 < selpref < 0.6,
							# 0.6 < selpref < 0.8, 0.8 < selpref,
							)
					sentdist = mention.sentno - other.sentno  # dist in sents
					mentdist = n - m  # distance in number of mentions
					antwidth = len(other.tokens)  # antecedent mention width
					for x in (sentdist, mentdist, antwidth):
						# bin distances into:
						# [0,1,2,3,4,5-7,8-15,16-31,32-63,64+]
						# following https://aclweb.org/anthology/P16-1061
						feats += (x == 0, x == 1, x == 2, x == 3, x == 4,
								5 <= x <= 7, 8 <= x <= 15, 16 <= x <= 31,
								32 <= x <= 63, x >= 64)
					result.append(feats)
					nn -= 1
				self.anaphordata.append((a, len(self.coreferent), mention))
		if not result:
			return
		numotherfeats = len(result[0]) - 7
		buf = np.zeros((len(result),
				3 * embeddings.shape[-1] + numotherfeats))
		for n, featvec in enumerate(result):
			# mean of BERT token representations of the tokens in the mentions.
			msent, mbegin, mend = featvec[:3]
			osent, obegin, oend = featvec[3:6]
			mhd = featvec[6]
			buf[n, :embeddings.shape[-1]] = embeddings[
					idx[msent, mbegin]:idx[msent, mend - 1] + 1].mean(axis=0)
			buf[n, embeddings.shape[-1]:2 * embeddings.shape[-1]] = embeddings[
					idx[osent, obegin]:idx[osent, oend - 1] + 1].mean(axis=0)
			if mhd is not None:
				buf[n, 2 * embeddings.shape[-1]:-numotherfeats] = embeddings[
						idx[msent, mhd]]
			buf[n, -numotherfeats:] = featvec[-numotherfeats:]
		self.result.append(buf)

	def getvectors(self):
		return (np.vstack(self.result),
				np.array(self.coreferent, dtype=int),
				self.antecedents,
				self.anaphordata)

	def selectprefscore(self, mention, other):
		"""Derive selectional preferences for pronoun."""
		selec_pref = -1000
		key = None
		if mention.parentheadword is not None:
			key = '%s\thd/%s\t%s' % (
					mention.parentheadword,
					mention.node.get('rel'),
					other.head.get('root'))
		if key in self.selectdf.index:
			selec_pref = self.selectdf.loc[key, 'npmi'].max()
		return selec_pref / 10000


def getfeatures(pattern, parsesdir, tokenizer, bertmodel, restrict=None):
	data = CorefFeatures()
	files = glob(pattern)
	if not files:
		raise ValueError('pattern did not match any files: %s' % pattern)
	for n, conllfile in enumerate(files, 1):
		parses = os.path.join(parsesdir,
				os.path.basename(conllfile.rsplit('.', 1)[0]))
		trees, mentions = loadmentions(conllfile, parses, restrict=restrict)
		embeddings = bert.getvectors(parses, trees, tokenizer, bertmodel)
		data.add(trees, embeddings, mentions)
		print(f'encoded {n}/{len(files)}: {conllfile}', file=sys.stderr)
	X, y, antecedents, anaphordata = data.getvectors()
	return X, y, antecedents, anaphordata


def build_mlp_model(input_shape):
	"""Define a binary classifier."""
	model = keras.Sequential([
			keras.layers.InputLayer(input_shape=input_shape),
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
				1, name='output',
				kernel_regularizer=keras.regularizers.l2(LAMBD)),
			keras.layers.Activation('sigmoid'),
			])
	return model


def train(trainfiles, validationfiles, parsesdir, tokenizer, bertmodel,
		restrict):
	np.random.seed(1)
	python_random.seed(1)
	tf.random.set_seed(1)
	X_train, y_train, _clusters, _indices = getfeatures(
			trainfiles, parsesdir, tokenizer, bertmodel, restrict=restrict)
	X_val, y_val, _clusters, _indices = getfeatures(
			validationfiles, parsesdir, tokenizer, bertmodel)
	print('training data', X_train.shape)
	print('validation data', X_val.shape)

	classif_model = build_mlp_model([X_train.shape[-1]])
	classif_model.summary()
	classif_model.compile(
			optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
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
	with open(MODELFILE.replace('.pt', '.txt'), 'w', encoding='utf8') as out:
		print(' '.join(sys.argv), file=out)


def evaluate(validationfiles, parsesdir, tokenizer, bertmodel):
	X_val, y_val, antecedents, anaphordata = getfeatures(
			validationfiles, parsesdir, tokenizer, bertmodel)
	model = build_mlp_model([X_val.shape[-1]])
	model.load_weights(MODELFILE).expect_partial()

	probs = model.predict(X_val)

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
		# predlabel = list(probs[a:b] > MENTION_PAIR_THRESHOLD)
		# pred.append(antecedents[a:][predlabel.index(1)].clusterid
		# 		if 1 in predlabel else -1)
		# select most likely antecedent
		antecedent = antecedents[a + probs[a:b].argmax()]
		# NB: if none of the candidates is likely enough, predict -1.
		pred.append(antecedent.clusterid
				if probs[a:b].max() > MENTION_PAIR_THRESHOLD else -1)
		y_true.append(anaphor.clusterid)
		print(f'{int(pred[-1] == y_true[-1])} {probs[a:b].max():.3f}',
				anaphor.sentno, anaphor.begin, ' '.join(anaphor.tokens), '->',
				end=' ')
		if probs[a:b].max() > MENTION_PAIR_THRESHOLD:
			print(antecedent.sentno, antecedent.begin,
					' '.join(antecedent.tokens))
		else:
			print('(none)')
	pairpred = probs > MENTION_PAIR_THRESHOLD
	print('(pronoun, candidate) pair classification scores:')
	print(metrics.classification_report(y_val, pairpred,
			digits=3, zero_division=0))
	# The above are scores for mention pairs. To get actual pronoun accuracy,
	# select a best candidate for each pronoun and evaluate on that.
	print('Pronoun resolution accuracy: %5.2f'
			% (100 * metrics.accuracy_score(y_true, pred)))


def predict(trees, embeddings, mentions):
	"""Load pronoun resolver, get features for trees, and return a list of
	mention pairs (anaphor, antecedent) which are predicted to be
	coreferent."""
	data = CorefFeatures()
	data.add(trees, embeddings, mentions)
	if not data.result:
		return []
	X, _y, antecedents, anaphordata = data.getvectors()
	model = build_mlp_model([X.shape[-1]])
	model.load_weights(MODELFILE).expect_partial()
	probs = model.predict(X)
	result = []
	for a, b, anaphor in anaphordata:
		debug(anaphor.sentno, anaphor.begin, anaphor, anaphor.featrepr(),
				'depof=%s' % anaphor.parentheadword)
		if a == b:  # a pronoun with no candidates ...
			continue
		# select most likely antecedent
		best = a + probs[a:b].argmax()
		if probs[best] > MENTION_PAIR_THRESHOLD:
			antecedent = antecedents[best]
			result.append((anaphor, antecedent))
		for n in range(a, b if VERBOSE else a):
			debug('\t%d %d %s %s p=%g%s' % (
					antecedents[n].sentno, antecedents[n].begin,
					antecedents[n].node.get('rel'), antecedents[n],
					# data.selectprefscore(anaphor, antecedents[n]),
					probs[n],
					' %s %g best' % (
						'<>'[int(probs[best] > MENTION_PAIR_THRESHOLD)],
						MENTION_PAIR_THRESHOLD)
						if n == best else ''))
	return result


def main():
	"""CLI."""
	longopts = ['restrict=', 'eval=', 'help']
	try:
		opts, args = getopt.gnu_getopt(sys.argv[1:], '', longopts)
	except getopt.GetoptError:
		print(__doc__)
		return
	opts = dict(opts)
	if '--help' in opts or len(args) != 3:
		print(__doc__)
		return
	trainfiles, validationfiles, parsesdir = args
	restrict = None
	if opts.get('--restrict'):
		restrict = int(opts.get('--restrict'))
	tokenizer, bertmodel = bert.loadmodel(BERTMODEL)
	train(trainfiles, validationfiles, parsesdir, tokenizer, bertmodel,
			restrict)
	evalfiles = opts.get('--eval', validationfiles)
	evaluate(evalfiles, parsesdir, tokenizer, bertmodel)


if __name__ == '__main__':
	main()
