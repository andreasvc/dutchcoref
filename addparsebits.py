"""Add POS tags, parses, and NER to CoNLL 2012 files.

Situation: you have a CoNLL 2012 file with manually corrected coreference
information, but the rest of the columns are missing. This tool extracts that
information from other files and adds it to the manually corrected file.
CoNLL 2012 files are overwritten in-place! (Make backups; use version control).

Extract parses from Alpino XML trees:
Single file: addparsebits.py alpino <conllfile> <parsesdir>
Batch mode: addparsebits.py alpino <conllfilesdir> <parsesdir> --batch

Extract parses from CoNLL 2012 files:
addparsebits.py conll <conllgold> <conllparses>
"""
import os
import re
import sys
import glob
import getopt
from coref import readconll


def splitparse(parse, chunk):
	"""Split bracketed parse tree into parse bits."""
	result = re.sub(r'\([^\s()]+ [^\s()]+\)([^(]*)', r'*\1\n', parse)
	for n, parsebit in enumerate(result.replace(' ', '').strip().splitlines()):
		chunk[n][5] = parsebit


def addner(tree, chunk):
	"""Produce CoNLL 2012 start & end NER tags."""
	# NB: neclass attributes occur only on tokens,
	# will not produce correctly nested NER spans
	for line in chunk:
		line[10] = '*'
	nerlabels = [token.get('neclass', '-') for token
			in sorted(tree.findall('.//node[@word]'),
				key=lambda node: int(node.get('begin')))]
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


def convalpino(conllfile, parsesdir):
	"""Add parse bits to a single file, overwrite in-place."""
	from lxml import etree
	try:
		from discodop.tree import writebrackettree
		from discodop.treebank import AlpinoCorpusReader
		from discodop.treetransforms import raisediscnodes
	except ImportError:
		print('Install https://github.com/andreasvc/disco-dop')
		return
	conlldata = readconll(conllfile)
	header = open(conllfile).readlines()[0].rstrip()
	treebank = AlpinoCorpusReader(parsesdir + '/*.xml',
			morphology='replace',
			headrules='../disco-dop/alpino.headrules')
	for chunk, (_key, item) in zip(conlldata, treebank.itertrees()):
		if len(chunk) != len(item.sent):
			raise ValueError('length mismatch')
		if len(chunk[0]) < 11:
			raise ValueError('Not enough fields for gold CoNLL 2012 file')
	with open(conllfile, 'w') as out:
		print(header, file=out)
		for chunk, (_key, item) in zip(conlldata, treebank.itertrees()):
			raisediscnodes(item.tree)
			for n, (_, postag) in enumerate(item.tree.pos()):
				if len(chunk[n]) < 12:  # kludge
					chunk[n] = chunk[n][:-1] + ['-', chunk[n][-1]]
				# NB: parens as square brackets: N[eigen,...]
				chunk[n][4] = postag
			splitparse(writebrackettree(item.tree, item.sent), chunk)
			xmltree = etree.fromstring(item.block)
			addner(xmltree, chunk)
			for line in chunk:
				print('\t'.join(line), file=out)
			print('', file=out)
		print('#end document', file=out)


def convconll(goldconll, parsesconll):
	"""Copy columns from parsesconll to goldconll file; overwrite in-place."""
	goldconlldata = readconll(goldconll)
	header = open(goldconll).readlines()[0].rstrip()
	parsesconlldata = readconll(parsesconll)
	if len(goldconlldata) != len(parsesconlldata):
		raise ValueError('mismatch in number of sentences')
	for gchunk, pchunk in zip(goldconlldata, parsesconlldata):
		if len(gchunk) != len(pchunk):
			raise ValueError('Sentence length mismatch')
		if len(pchunk[0]) < 12:
			raise ValueError('Not enough fields for gold CoNLL 2012 file')
		if len(gchunk[0]) < 12:
			raise ValueError('Not enough fields for parses CoNLL 2012 file')
	with open(goldconll, 'w') as out:
		print(header, file=out)
		for gchunk, pchunk in zip(goldconlldata, parsesconlldata):
			for gline, pline in zip(gchunk, pchunk):
				gline[4] = pline[4]
				gline[5] = pline[5]
				gline[10] = pline[10]
				print('\t'.join(gline), file=out)
			print('', file=out)
		print('#end document', file=out)


def main():
	"""CLI."""
	try:
		opts, args = getopt.gnu_getopt(sys.argv[1:], '', ['batch'])
		cmd, goldconll, parses = args
	except (getopt.GetoptError, ValueError):
		print(__doc__)
		return
	opts = dict(opts)
	if cmd == 'alpino' and '--batch' in opts:
		for directory in glob.glob(os.path.join(parses, '*/')):
			docid = os.path.basename(directory.rstrip('/'))
			conllfile = os.path.join(goldconll, docid + '.conll')
			if os.path.exists(conllfile):
				convalpino(conllfile, directory)
	elif cmd == 'conll' and '--batch' in opts:
		raise NotImplementedError
	elif cmd == 'alpino':
		convalpino(goldconll, parses)
	elif cmd == 'conll':
		convconll(goldconll, parses)
	else:
		print(__doc__)
		return


if __name__ == '__main__':
	main()
