"""Mention feature classifier.

Usage: mentionfeatureclassifier.py <train> <validation> <parsesdir>
Or: mentionfeatureclassifier.py <parsesdir> --import=<dir> --eval=<test>
Example: mentionfeatureclassifier.py 'train/*.conll' 'dev/*.conll' parses/

Options:
    --import=<dir>  import annotated features from TSV files;
                    the annotations for an entity will override the detected
                    features of all its mentions.
    --export=<dir>  export detected features to TSV files for annotation;
                    when this option is enabled, no training is done.
    --restrict=N    restrict training data to the first N% of each file.
    --eval=<test>   report evaluation on this set using already trained model.
                    (NB: this is only meaningful if annotated features are
                    imported for this test set).
"""
# requirements:
# - pip install 'transformers>=4.0' keras tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

import sys
import getopt
from glob import glob
import random as python_random
from lxml import etree
import numpy as np
import pandas as pd
from tensorflow import keras
import tensorflow as tf
from sklearn.metrics import classification_report
from coref import (readconll, readngdata, conllclusterdict, getheadidx,
		parsesentid, Mention, mergefeatures, gettokens, initialsegment,
		color, debug, VERBOSE)
import bert

DENSE_LAYER_SIZES = [500, 150, 150]
INPUT_DROPOUT_RATE = 0.2
DROPOUT_RATE = 0.5
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
EPOCHS = 100  # maximum number of epochs
PATIENCE = 5  # stop after this number of epochs wo/improvement
LAMBD = 0.05  # L2 regularization
MODELFILE = 'mentionfeatclassif.pt'


def extractmentionsfromconll(name, conlldata, trees, ngdata, gadata,
		annotations=None, export=None):
	"""Extract gold mentions from annotated data and merge features.

	:returns: mentions sorted by sentno, begin; including gold clusterid
		and detected features for the cluster."""
	mentions = []
	goldspansforcluster = conllclusterdict(conlldata)
	for _clusterid, spans in goldspansforcluster.items():
		firstment = annotatedfeat = None
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
			mention.origfeat = mention.features.copy()
			if (annotations is not None
					and (name, sentno, begin, end) in annotations):
				mention.features.update(annotations[name, sentno, begin, end])
				# human feature is implied by gender feature
				if mention.features['gender'] == 'n':
					mention.features['human'] = 0
				elif mention.features['gender'] in ('f', 'm', 'fm'):
					mention.features['human'] = 1
				else:
					raise ValueError(('annotated gender for %r'
							' has unrecognized value %r; '
							'should be one of f, m, n, or fm.') % (
							(name, sentno, begin, end),
							mention.features['gender']))
				annotatedfeat = mention.features
			if firstment is None:
				firstment = mention
			elif annotatedfeat is not None:
				mention.features.update(firstment.features)
			else:
				mergefeatures(firstment, mention)
			mentions.append(mention)
		if export is not None:
			export.append(
					(name, firstment.sentno, firstment.begin, firstment.end,
					firstment.features['gender'] or '',
					firstment.features['number'] or '',
					', '.join(text for _, _, _, text in sorted(spans)),
					' '.join(gettokens(firstment.node.getroottree().getroot(),
						0, 9999))))
	# sort by sentence, then from longest to shortest span
	mentions.sort(key=lambda x: (x.sentno, x.begin - x.end))
	for n, mention in enumerate(mentions):
		mention.id = n  # fix mention IDs after sorting
	return mentions


def loadmentions(conllfile, parsesdir, ngdata, gadata,
		annotations, exportpath=None, restrict=None):
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
	name = os.path.splitext(os.path.basename(conllfile))[0]
	export = [] if exportpath else None
	try:
		# extract gold mentions with gold clusters
		mentions = extractmentionsfromconll(
				name, conlldata, trees, ngdata, gadata, annotations, export)
	except Exception:
		print('issue with', conllfile)
		raise
	if exportpath:
		df = pd.DataFrame(export,
				columns=['filename', 'sentno', 'begin', 'end', 'gender',
					'number', 'mentions', 'sentence'])
		df.to_csv(os.path.join(exportpath, name) + '.tsv',
				sep='\t', index=False)
	return trees, mentions


class MentionFeatures:
	def __init__(self):
		self.result = []  # collected feature vectors for mentions
		self.labels = []  # the target labels for the mentions
		self.mentions = []  # the mention objects

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
		# collect mention features
		for mention in mentions:
			# feature indicators: ['nh', 'h', 'f', 'm', 'n', 'sg', 'pl']
			# multiple values can be True!
			# if a feature is unknown, all of its possible values will be False
			label = np.zeros(7)
			label[0] = mention.features['human'] == 0
			label[1] = mention.features['human'] == 1
			label[2] = 'f' in (mention.features['gender'] or '')
			label[3] = 'm' in (mention.features['gender'] or '')
			label[4] = 'n' in (mention.features['gender'] or '')
			label[5] = mention.features['number'] == 'sg'
			label[6] = mention.features['number'] == 'pl'
			self.labels.append(label)
			self.mentions.append(mention)
			# collecting additional features
			# FIXME: feature: is mention part of another mention?
			result.append((
					mention.sentno, mention.begin, mention.end,
					# additional features
					mention.node.get('rel') == 'su',
					mention.node.get('rel') == 'obj1',
					# does this NP contain another NP?
					mention.node.find('.//node[@cat="np"]') is not None,
					# features detected with lexical resources
					mention.origfeat['human'] == 0,
					mention.origfeat['human'] == 1,
					'f' in (mention.origfeat['gender'] or ''),
					'm' in (mention.origfeat['gender'] or ''),
					'n' in (mention.origfeat['gender'] or ''),
					mention.origfeat['number'] == 'sg',
					mention.origfeat['number'] == 'pl',
					))
		buf = np.zeros((len(result), embeddings[0].shape[-1]))
		# concatenate BERT embeddings with additional features
		numotherfeats = len(result[0]) - 3
		buf = np.zeros((len(result), embeddings[0].shape[-1] + numotherfeats))
		for n, featvec in enumerate(result):
			# mean of BERT token representations of the tokens in the mentions.
			msent, mbegin, mend = featvec[:3]
			buf[n, :embeddings[0].shape[-1]] = embeddings[
					idx[msent, mbegin]:idx[msent, mend - 1] + 1].mean(axis=0)
			buf[n, -numotherfeats:] = featvec[-numotherfeats:]
		self.result.append(buf)

	def getvectors(self):
		return (np.vstack(self.result),
				np.vstack(self.labels),
				self.mentions)


def getfeatures(pattern, parsesdir, tokenizer, bertmodel,
		annotations=None, restrict=None):
	data = MentionFeatures()
	ngdata, gadata = readngdata()
	files = glob(pattern)
	if not files:
		raise ValueError('pattern did not match any files: %s' % pattern)
	for n, conllfile in enumerate(files, 1):
		parses = os.path.join(parsesdir,
				os.path.basename(conllfile.rsplit('.', 1)[0]))
		trees, mentions = loadmentions(conllfile, parses, ngdata, gadata,
				annotations=annotations, restrict=restrict)
		embeddings = bert.getvectors(parses, trees, tokenizer, bertmodel)
		data.add(trees, embeddings, mentions)
		print(f'encoded {n}/{len(files)}: {conllfile}', file=sys.stderr)
	X, y, mentions = data.getvectors()
	return X, y, mentions


def build_mlp_model(input_shape, num_labels):
	"""Define a binary classifier."""
	model = keras.Sequential([
			keras.layers.InputLayer(input_shape=input_shape),
			keras.layers.Dropout(INPUT_DROPOUT_RATE),

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


def train(trainfiles, validationfiles, parsesdir, annotations, restrict,
		tokenizer, bertmodel):
	np.random.seed(1)
	python_random.seed(1)
	tf.random.set_seed(1)
	X_train, y_train, _mentions = getfeatures(
			trainfiles, parsesdir, tokenizer, bertmodel, annotations,
			restrict=restrict)
	X_val, y_val, _mentions = getfeatures(
			validationfiles, parsesdir, tokenizer, bertmodel, annotations)
	print('training data', X_train.shape)
	print('validation data', X_val.shape)
	classif_model = build_mlp_model([X_train.shape[-1]], y_val.shape[-1])
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


def evaluate(validationfiles, parsesdir, annotations, tokenizer, bertmodel):
	def featvals(mention):
		return [
				mention.origfeat['human'] == 0,
				mention.origfeat['human'] == 1,
				'f' in (mention.origfeat['gender'] or ''),
				'm' in (mention.origfeat['gender'] or ''),
				'n' in (mention.origfeat['gender'] or ''),
				mention.origfeat['number'] == 'sg',
				mention.origfeat['number'] == 'pl',
				]

	def featvalsfallback(mention, probs):
		names = ['human', 'human'] + 3 * ['gender'] + ['number', 'number']
		return [probs[n] > 0.5 if mention.origfeat[name] is None else val
				for n, (name, val) in enumerate(zip(names, featvals(mention)))]

	X_val, y_val, mentions = getfeatures(
			validationfiles, parsesdir, tokenizer, bertmodel, annotations)
	model = build_mlp_model([X_val.shape[-1]], y_val.shape[-1])
	model.load_weights(MODELFILE).expect_partial()
	probs = model.predict(X_val)
	print('feat=prob/gold')
	for mention, p, g in zip(mentions, probs, np.array(y_val, dtype=int)):
		print(f'nh={p[0]:.3f}/{g[0]} '
				f'h={p[1]:.3f}/{g[1]} '
				f'f={p[2]:.3f}/{g[2]} '
				f'm={p[3]:.3f}/{g[3]} '
				f'n={p[4]:.3f}/{g[4]} '
				f'sg={p[5]:.3f}/{g[5]} '
				f'pl={p[6]:.3f}/{g[6]} '
				f'{" ".join(mention.tokens)}')
	target_names = ['nonhuman', 'human', 'female', 'male', 'neuter',
				'singular', 'plural']
	print('\nperformance of features detected with ngdata/gadata:')
	print(classification_report(
			np.array(y_val, dtype=bool),
			np.array([featvals(mention) for mention in mentions], dtype=bool),
			target_names=target_names,
			zero_division=0,
			digits=3))
	print('\nperformance of ngdata/gadata with fallback to feature classifier:')
	print(classification_report(
			np.array(y_val, dtype=bool),
			np.array([featvalsfallback(mention, pr) for mention, pr
				in zip(mentions, probs)], dtype=bool),
			target_names=target_names,
			zero_division=0,
			digits=3))
	print('\nperformance of feature classifier:')
	print(classification_report(
			np.array(y_val, dtype=bool),
			np.array([a > 0.5 for a in probs], dtype=bool),
			target_names=target_names,
			zero_division=0,
			digits=3))


def predict(trees, embeddings, mentions):
	"""Load mentions classfier, get features for mentions, and update features
	of mentions."""
	debug(color('mention feature detection', 'yellow'))
	data = MentionFeatures()
	for mention in mentions:
		mention.origfeat = mention.features.copy()
	data.add(trees, embeddings, mentions)
	X, y, mentions = data.getvectors()
	model = build_mlp_model([X.shape[-1]], y.shape[-1])
	model.load_weights(MODELFILE).expect_partial()
	probs = model.predict(X)
	for row, mention in zip(probs, mentions):
		if row[0] > 0.5 and row[1] < 0.5:
			mention.features['human'] = 0
		elif row[0] < 0.5 and row[1] > 0.5:
			mention.features['human'] = 1
		else:
			mention.features['human'] = None
		gend = ''
		if row[2] > 0.5:
			gend += 'f'
		if row[3] > 0.5:
			gend += 'm'
		if row[4] > 0.5:
			gend += 'n'
		if gend != '' and gend != 'fmn':
			mention.features['gender'] = gend
		elif mention.features['human']:
			mention.features['gender'] = 'fm'
		else:
			mention.features['gender'] = None
		if row[5] > 0.5 and row[6] < 0.5:
			mention.features['number'] = 'sg'
		elif row[5] < 0.5 and row[6] > 0.5:
			mention.features['number'] = 'pl'
		else:
			mention.features['number'] = None
		if VERBOSE:
			debug('%3d %2d %s ' % (mention.sentno, mention.begin, mention),
					# mention.featrepr(extended=True)),
					f'nh={row[0]:.3f} '
					f'h={row[1]:.3f} '
					f'f={row[2]:.3f} '
					f'm={row[3]:.3f} '
					f'n={row[4]:.3f} '
					f'sg={row[5]:.3f} '
					f'pl={row[6]:.3f} ')


def main():
	"""CLI."""
	longopts = ['import=', 'export=', 'restrict=', 'eval=', 'help']
	try:
		opts, args = getopt.gnu_getopt(sys.argv[1:], '', longopts)
	except getopt.GetoptError:
		print(__doc__)
		return
	opts = dict(opts)
	annotations = restrict = None
	if opts.get('--import'):
		fnames = glob(os.path.join(opts.get('--import'), '*.tsv'))
		result = []
		for fname in fnames:
			try:
				result.append(pd.read_csv(
						fname, sep='\t',
						dtype={'gender': str}, keep_default_na=False))
			except Exception:
				print('issue with', fname)
				raise
		annotations = pd.concat(result).set_index(
				['filename', 'sentno', 'begin', 'end'])[
				['gender', 'number']].T.to_dict()
	if opts.get('--eval'):
		tokenizer, bertmodel = bert.loadmodel()
		evaluate(opts['--eval'], args[0], annotations, tokenizer, bertmodel)
		return
	elif '--help' in opts or len(args) != 3:
		print(__doc__)
		return
	trainfiles, validationfiles, parsesdir = args
	if opts.get('--restrict'):
		restrict = int(opts.get('--restrict'))
	if opts.get('--export'):
		exportpath = opts.get('--export')
		ngdata, gadata = readngdata()
		for pattern in (trainfiles, validationfiles):
			files = glob(pattern)
			if not files:
				raise ValueError('pattern did not match any files: ' + pattern)
			for conllfile in files:
				parses = os.path.join(parsesdir,
						os.path.basename(conllfile.rsplit('.', 1)[0]))
				_ = loadmentions(conllfile, parses, ngdata, gadata,
						annotations=annotations, exportpath=exportpath)
	else:
		tokenizer, bertmodel = bert.loadmodel()
		train(trainfiles, validationfiles, parsesdir,
				annotations, restrict, tokenizer, bertmodel)
		evaluate(validationfiles, parsesdir, annotations, tokenizer, bertmodel)


if __name__ == '__main__':
	main()
