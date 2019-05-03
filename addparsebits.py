"""Take CoNLL 2012 files and directories with Alpino XML parse trees,
and add POS tags and parse bits to the CoNLL 2012 files;
CoNLL 2012 files are overwritten in-place!

Single file: addparsebits.py <conllfile> <parsesdir>
All files in a directory: addparsebits.py <conllfilesdir> <parsesdir> --batch
"""
import os
import re
import sys
import glob
from lxml import etree
from discodop.tree import writebrackettree
from discodop.treebank import AlpinoCorpusReader
from discodop.treetransforms import raisediscnodes
from coref import readconll


def splitparse(parse, chunk):
	"""Split bracketed parse tree into parse bits."""
	result = re.sub(r'\([^\s()]+ [^\s()]+\)([^(]*)', r'*\1\n', parse)
	for n, parsebit in enumerate(result.replace(' ', '').strip().splitlines()):
		chunk[n][5] = parsebit


def addner(block, chunk):
	"""Produce CoNLL 2012 start & end NER tags."""
	# NB: neclass attributes occur only on tokens,
	# will not produce correctly nested NER spans
	tree = etree.fromstring(block)
	for line in chunk:
		line[10] = '*'
	nerlabels = [token.get('neclass', '-') for token
			in tree.findall('.//node[@word]')]
	for n, ner in enumerate(nerlabels):
		if ner == '-':
			continue
		elif n == 0 or nerlabels[n - 1] != ner:
			if n == len(chunk) - 1 or nerlabels[n + 1] != ner:
				chunk[n][10] = '(%s)' % ner
			else:
				chunk[n][10] = '(%s*' % ner
		elif n == len(chunk) - 1 or nerlabels[n + 1] != ner:
			chunk[n][10] = '*)'


def conv(conllfile, parsesdir):
	"""Add parse bits to a single file, overwrite in-place."""
	conlldata = readconll(conllfile)
	header = open(conllfile).readlines()[0].rstrip()
	treebank = AlpinoCorpusReader(parsesdir + '/*.xml',
			morphology='replace',
			headrules='../disco-dop/alpino.headrules')
	with open(conllfile, 'w') as out:
		print(header, file=out)
		for chunk, (_key, item) in zip(conlldata, treebank.itertrees()):
			if len(chunk) != len(item.sent):
				raise ValueError('length mismatch')
			raisediscnodes(item.tree)
			for n, (_, postag) in enumerate(item.tree.pos()):
				# NB: parens as square brackets: N[eigen,...]
				chunk[n][4] = postag
			splitparse(writebrackettree(item.tree, item.sent), chunk)
			addner(item.block, chunk)
			for line in chunk:
				print('\t'.join(line), file=out)
			print('', file=out)
		print('#end document', file=out)


def main():
	"""CLI."""
	if '--batch' in sys.argv:
		conllfiles, parsesdir = sys.argv[1:]
		for directory in glob.glob(parsesdir + '/*/'):
			conllfile = (conllfiles + '/'
					+ os.path.basename(directory.rstrip('/')) + '.conll')
			if os.path.exists(conllfile):
				conv(conllfile, directory)
	else:
		conllfile, parsesdir = sys.argv[1:]
		conv(conllfile, parsesdir)


if __name__ == '__main__':
	main()
