"""Convert Corea/SoNaR coreference annotations to CoNLL 2012 format.

Usage: mmaxconll.py <inputdir> [outputfile]

inputdir is searched recursively for Basedata/ and Markables/ subdirectories.
output is written on stdout if no output is specified."""
# Notes:
# The following files in Corea contain XML syntax errors:
# Med/Markables/s236_coref_level.xml
# Med/Markables/s397_coref_level.xml
import os
import re
import sys
import glob
import getopt
from collections import defaultdict
from lxml import etree


def getspan(markable, idxmap, words):
	"""Convert an MMAX span into integer indices."""
	start = end = span = markable.attrib['span']
	# if the span is discontinuous, look for the componenent
	# with the head and return that as the (minimal) span.
	if ',' in span:
		head = markable.get('head', '').lower()
		for component in span.split(','):
			start = end = component
			if '..' in component:
				start, end = component.split('..')
			for n in range(idxmap[start], idxmap[end] + 1):
				if words[n].text.lower() == head:
					return idxmap[start], idxmap[end]
		# didn't find head, return last component as default
		return idxmap[start], idxmap[end]
	if '..' in markable.attrib['span']:
		start, end = markable.attrib['span'].split('..')
	return idxmap[start], idxmap[end]


def getclusters(words, nplevel, idxmap, skiptypes=('bridge', )):
	"""Convert coreference chain links into clusters."""
	# A table of reference chains (anaphor -> antecedent)
	forwardrefs = defaultdict(list)
	for markable in nplevel:
		ref = markable.get('ref')
		if ref is not None and ref != 'empty':
			for ref1 in ref.split(';'):
				forwardrefs[ref1].append(markable.get('id'))
	# Get transitive closure
	sets = []
	for markable in sorted(nplevel,
			key=lambda m: int(m.get('id').split('_')[1])):
		if 'ref' not in markable.attrib or markable.get('ref') == 'empty':
			refset = set()
			stack = [markable.get('id')]
			while stack:
				mid = stack.pop()
				if mid not in refset:
					refset.add(mid)
					stack.extend(forwardrefs.get(mid, []))
			sets.append(refset)
	# Assign each markable to an ID of its coreference cluster
	cluster = {}
	for n, refset in enumerate(sets):
		for mid in refset:
			# if mid in cluster:
			# 	raise ValueError
			cluster[mid] = n
	for markable in nplevel:
		mid = markable.get('id')
		# if mid not in cluster:
		# 	raise ValueError
	return cluster


def addclusters(words, nplevel, idxmap, cluster, skiptypes=('bridge', )):
	"""Add start and end tags of markables to the respective tokens."""
	seen = set()  # don't add same span twice
	inspan = set()  # track which token ids are in spans (except for last idx)
	missing = len(cluster)
	for markable in nplevel:
		try:

			start, end = getspan(markable, idxmap, words)
		except KeyError:  # ignore spans referring to non-existing tokens
			continue
		if (start, end) in seen:
			continue
		seen.add((start, end))
		inspan.update(range(start, end))  # NB: do not include end!
		if markable.get('type') in skiptypes:
			continue
		elif markable.get('id') in cluster:
			cid = cluster[markable.get('id')]
		else:
			cid = missing
			missing += 1
		cur = words[start].get('coref', '')
		if start == end:
			coref = ('%s|(%s)' % (cur, cid)) if cur else ('(%s)' % cid)
			words[start].set('coref', coref)
		else:
			coref = ('%s|(%s' % (cur, cid)) if cur else ('(%s' % cid)
			words[start].set('coref', coref)
			cur = words[end].get('coref', '')
			coref = ('%s|%s)' % (cur, cid)) if cur else ('%s)' % cid)
			words[end].set('coref', coref)
	return inspan


def getsents(words, sentence, idxmap):
	"""Extract indices of sentence breaks."""
	# The SoNaR sentence annotations are a mess:
	# duplicate spans, overlapping spans, etc.
	# The approach here is to output all tokens in order, and insert
	# a sentence break if it is annotated (i.e., ignore sentence starts,
	# because it could lead to missing/repeated tokens).
	sentends = set()
	if sentence is None:  # Corea
		for n, word in enumerate(words[1:], 1):
			if word.get('pos') == '0' or word.get('alppos') == '0':
				sentends.add(n - 1)
	else:  # SoNaR
		for markable in sentence:
			try:
				sentends.add(getspan(markable, idxmap, words)[1])
			except KeyError:  # ignore spans referring to non-existing tokens
				pass
	return sentends


def writeconll(words, sentends, doc, inspan, out):
	"""Write tokens and coreference information in CoNLL format."""
	n = 0
	print('#begin document (%s); part 000' % doc, file=out)
	for m, word in enumerate(words):
		print(doc, n, word.text, word.get('coref', '-'), sep='\t', file=out)
		n += 1
		# ignore sent break if any span still open
		if m in sentends and m not in inspan:
			print(file=out)
			n = 0
	print('#end document', file=out)


def conv(fname, inputdir, out):
	"""Convert a set of files for a single MMAX document to CoNLL."""
	words = etree.parse(fname).getroot()
	doc = os.path.basename(fname).replace('_words.xml', '')
	if os.path.exists('%s/Markables/%s_np_level.xml' % (inputdir, doc)):
		nplevel = etree.parse('%s/Markables/%s_np_level.xml'
				% (inputdir, doc)).getroot()
	elif os.path.exists('%s/Markables/%s_coref_level.xml' % (inputdir, doc)):
		nplevel = etree.parse('%s/Markables/%s_coref_level.xml'
				% (inputdir, doc)).getroot()
	else:
		return  # no annotations exist for this file
	if os.path.exists('%s/Markables/%s_sentence_level.xml' % (inputdir, doc)):
		sentence = etree.parse('%s/Markables/%s_sentence_level.xml'
				% (inputdir, doc)).getroot()
	else:
		sentence = None
	# word IDs may be missing or contain decimals;
	# this maps ID labels to integer indices.
	idxmap = {word.attrib['id']: n
			for n, word in enumerate(words)}
	cluster = getclusters(words, nplevel, idxmap)
	inspan = addclusters(words, nplevel, idxmap, cluster)
	sentends = getsents(words, sentence, idxmap)
	writeconll(words, sentends, doc, inspan, out)


def main():
	"""CLI."""
	try:
		_opts, args = getopt.gnu_getopt(sys.argv[1:], '', [])
	except getopt.GetoptError as err:
		args = None
	if not args or len(args) > 2:
		print(__doc__, sep='\n')
		return
	out = None if len(args) == 1 else open(args[1], 'w', encoding='utf8')
	try:
		for dirpath, dirnames, _ in os.walk(args[0]):
			if 'Basedata' in dirnames and 'Markables' in dirnames:
				pattern = '%s/Basedata/*.xml' % dirpath
				for fname in sorted(glob.glob(pattern)):
					conv(fname, dirpath, out)
	finally:
		if out:
			out.close()


if __name__ == '__main__':
	main()
