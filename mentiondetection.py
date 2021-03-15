"""Mention detection for coreference.

Usage: mentiondetection.py extract <trainconllfiles> <devconllfiles> <parsesdir>
(writes output to mentflattrain.txt and mentflatdev.txt; used by commands below).
Usage: mentiondetection.py train
Usage: mentiondetection.py eval

NB: training is about 50x faster with a GPU.
"""
# requirements:
# - pip install simpletransformers
# - dutchcoref; put this file in a clone of https://github.com/andreasvc/dutchcoref/
# Google Colab version of this code:
# https://colab.research.google.com/drive/1yamjiHg87Lt0cjYuAOD3PontJMnNS9LT?usp=sharing
import os
import sys
from glob import glob
from lxml import etree
from coref import (readconll, readngdata, conllclusterdict, getheadidx,
		parsesentid, Mention, mergefeatures)


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


def getbio(conllfiles, parsesdir, outputfile, outputfilenested):
	"""Extract mention spans in BIO format.

	:param conllfiles: pattern for input, eg. '*.conll' or 'coref/*.conll'
	:param parsesdir: directory with parsetrees
	:param outputfile: filename for output (flat format, longest mention only)
	:param outputfilenested: output in nested format (4 columns)
	"""
	ngdata, gadata = readngdata()
	labelset = set()
	with open(outputfile, 'w', encoding='utf8') as out:
		with open(outputfilenested, 'w', encoding='utf8') as outnested:
			for conllfile in glob(conllfiles):
				print('processing', conllfile, file=sys.stderr)
				parses = os.path.join(parsesdir,
						os.path.basename(conllfile.rsplit('.', 1)[0]),
						'*.xml')
				filenames = sorted(glob(parses), key=parsesentid)
				trees = [(parsesentid(filename), etree.parse(filename))
						for filename in filenames]
				data = readconll(conllfile)
				assert len(data) == 1, 'expected one document per conll file'
				conlldata = next(iter(data.values()))
				mentions = extractmentionsfromconll(
						conlldata, trees, ngdata, gadata)

				# nested BIO (max 4)
				labels = [[['O'] * 4 for token in sent]
						for sent in conlldata]
				for mention in mentions:
					hum = 'h' if mention.features['human'] else 'nh'
					gen = mention.features['gender']
					if gen is None:  # FIXME: add this to dutchcoref?
						gen = 'fm' if mention.features['human'] else 'n'
					label = f'{hum}-{gen}'
					labelset.add(label)
					for n in range(4):
						if all(
								labels[mention.sentno][i][n] == 'O'
								for i in range(mention.begin, mention.end)):
							labels[mention.sentno][mention.begin][n] = (
									'B-' + label)
							for i in range(mention.begin + 1, mention.end):
								labels[mention.sentno][i][n] = 'I-' + label
							break
					else:
						print('warning: mention nested too deep (>4 levels):',
								' '.join(mention.tokens), file=sys.stderr)
				for sentno, sent in enumerate(conlldata):
					for tokno, line in enumerate(sent):
						# for flat, only print 1st column, separate by space
						print(line[4], labels[sentno][tokno][0], file=out)
						# for nested, all 4 columns, separated by tab
						print(line[4], *labels[sentno][tokno],
								sep='\t', file=outnested)
					print('', file=out)
					print('', file=outnested)

	print('wrote', outputfile, file=sys.stderr)
	print('wrote', outputfilenested, file=sys.stderr)
	print(f'{len(labelset)} unique labels: {labelset}', file=sys.stderr)


def train(trainfile, evalfile):
	"""Train and evaluate flat NER model with simpletransformers."""
	import torch
	from simpletransformers.ner import NERModel
	cuda_available = torch.cuda.is_available()  # detect GPU is available

	model = NERModel(
			'bert', 'GroNLP/bert-base-dutch-cased',  # BERTje
			use_cuda=cuda_available,
			labels=['B-MENT', 'I-MENT', 'O'],
			args={
				'overwrite_output_dir': True,
				'reprocess_input_data': True,
				'output_dir': 'mentionmodel',
				})

	model.train_model(trainfile)
	model.save_model()
	results, _model_outputs, _predictions = model.eval_model(evalfile)

	print(results)


def doeval(evalfile):
	"""Evaluate existing simpletransformers model."""
	import torch
	from simpletransformers.ner import NERModel
	cuda_available = torch.cuda.is_available()
	model = NERModel(
			'bert', 'mentionmodel',
			use_cuda=cuda_available,
			labels=['B-MENT', 'I-MENT', 'O'])
	results, _model_outputs, _predictions = model.eval_model(evalfile)
	print(results)


def predict(sentences):
	"""Return predictions from existing model."""
	import torch
	from simpletransformers.ner import NERModel
	cuda_available = torch.cuda.is_available()
	model = NERModel(
			'bert', 'mentionmodel',
			use_cuda=cuda_available,
			labels=['B-MENT', 'I-MENT', 'O'])
	predictions, _raw_outputs = model.predict(sentences)
	return predictions


def main():
	"""CLI."""
	if sys.argv[1] == 'extract':
		getbio(sys.argv[2], sys.argv[4],
				'mentflattrain.txt', 'mentnestedtrain.tsv')
		getbio(sys.argv[3], sys.argv[4],
				'mentflatdev.txt', 'mentnesteddev.tsv')
	elif sys.argv[1] == 'train':
		train('mentflattrain.txt', 'mentflatdev.txt')
	elif sys.argv[1] == 'eval':
		doeval('mentflatdev.txt')

if __name__ == '__main__':
	main()
