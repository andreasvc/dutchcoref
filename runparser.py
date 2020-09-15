"""Given one or more CoNLL files, parse sentences with Alpino.

Assumes that Alpino is installed and available in PATH."""
import os
import re
import tempfile
from glob import glob
from coref import readconll


def escapebrackets(word):
	"""Escape square brackets for Alpino."""
	return word.replace('[', r'\[').replace(']', r'\]')


def parse(conlldata, docname, tokenidx):
	"""Parse a single document in a CoNLL file."""
	with tempfile.NamedTemporaryFile(mode='wt', encoding='utf8') as out:
		for sent in conlldata:
			out.write(' '.join(
					escapebrackets(fields[tokenidx])
					for fields in sent))
			out.write('\n\n')
		out.flush()

		os.mkdir(docname)
		os.system(
				'cat %s | Alpino number_analyses=1 end_hook=xml '
				'-flag treebank %s -parse' % (out.name, docname))


def parseclindata(pattern, outdir):
	"""Parse the CLIN dataset."""
	origdir = os.getcwd()
	filenames = glob(os.path.abspath(pattern))
	os.mkdir(outdir)
	os.chdir(outdir)
	for n, conllfile in enumerate(filenames, 1):
		data = next(iter(readconll(conllfile).values()))
		fname = os.path.basename(conllfile)
		docname = fname[:fname.index('_')]
		tokenidx = 3
		print('Parsing %d/%d: %s' % (n, len(filenames), docname))
		parse(data, docname, tokenidx)
	os.chdir(origdir)


def parsesemeval(path, outdir):
	"""Parse the SemEval dataset."""
	path = os.path.abspath(path)
	origdir = os.getcwd()
	os.mkdir(outdir)
	os.chdir(outdir)
	with open(path) as inp:
		data = inp.read()
	docnames = re.findall(r'#begin document ([\w_]+)', data)
	docs = readconll(path)
	for n, docname in enumerate(docnames, 1):
		data = docs[docname, 0]
		tokenidx = 2
		print('Parsing %d/%d: %s' % (n, len(docnames), docname))
		parse(data, docname, tokenidx)
	os.chdir(origdir)


if __name__ == '__main__':
	# CLIN test:
	# - separate .coref_ne files
	# - parses in directories named after numeric prefix '_'
	# SemEval test
	# - single conll file, multiple chunks
	# - create dir for chunks
	parsesemeval('data/semeval2010/task01.posttask.v1.0/corpora/test/'
			'nl.test.txt.fixed', 'data/semeval2010NLtestparses')
	os.mkdir('data/clinTestData/')
	for subset in ('boeing', 'gm', 'stock'):
		parseclindata(
				('../groref/clin26-eval-master/eval_corpora/%s/coref_ne/'
					'*.coref_ne' % subset),
				'data/clinTestData/%s' % subset)
