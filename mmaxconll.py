"""Convert Corea/SoNaR coreference annotations to CoNLL 2012 format.

Usage: mmaxconll.py [options] <inputdir> [outputdir]

inputdir is searched recursively for Basedata/ and Markables/ subdirectories.
outputdir should not exist; if not specified, output is written on stdout.

Options:
  --lassy=<path>     Specify path to Lassy Small Treebank/ directory. Sentence
                     and token boundaries will be aligned to Lassy trees.
                     Unalignable tokens in either SoNaR or Lassy will be logged
                     and skipped. Will create a copy of the treebank under
                     outputdir with re-ordered and renumbered trees
                     (parno-sentno.xml) corresponding to those in the
                     coreference annotations.
  --sonarner=<path>  Specify path to SoNaR NE annotations. Named entities will
                     be added to Lassy trees as neclass attributes on tokens.
"""
# Notes:
# The following files in Corea contain XML syntax errors:
# Med/Markables/s236_coref_level.xml
# Med/Markables/s397_coref_level.xml
import os
import re
import sys
import getopt
from glob import glob
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


def getclusters(nplevel):
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
	# NB: in our conversion, a markable can only be part of a single
	# cluster.
	cluster = {}
	for n, refset in enumerate(sets):
		for mid in refset:
			cluster[mid] = n
	return cluster


def addclusters(words, nplevel, idxmap, cluster, skiptypes=('bridge', )):
	"""Add start and end tags of markables to the respective tokens."""
	seen = set()  # don't add same span twice
	inspan = set()  # track which token ids are in spans (except for last idx)
	missing = len(cluster)
	markables = []
	for markable in nplevel:
		try:
			markables.append((*getspan(markable, idxmap, words), markable))
		except KeyError:  # ignore spans referring to non-existing tokens
			continue
	for start, end, markable in sorted(markables, key=lambda m: m[1] - m[0]):
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
		else:
			coref = ('(%s|%s' % (cid, cur)) if cur else ('(%s' % cid)
			words[start].set('coref', coref)
			cur = words[end].get('coref', '')
			coref = ('%s|%s)' % (cur, cid)) if cur else ('%s)' % cid)
		words[end].set('coref', coref)
	return inspan


def parsesentid(fname):
	"""Create sort key with padding from Lassy filename."""
	return (tuple(map(int, re.findall(r'\d+', os.path.basename(fname))))
			+ (0, 0, 0, 0, 0, 0, 0))[:7]


def getsents(words, sentence, idxmap, sdocname, ldocname,
		lassypath=None, sonarnerpath=None, outpath=None, lassymap=None,
		lassynewids=None, lassyunaligned=None, sonarunaligned=None):
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
	elif lassypath:
		from coref import gettokens
		lassypath = os.path.join(lassypath, ldocname, '*.xml')
		lassytrees = {fname: etree.parse(fname) for fname in glob(lassypath)}
		if not lassytrees:
			raise ValueError('no trees found at %s' % lassypath)

        # We take the order of sentences and tokens in Sonar as canonical,
        # while we take the sentence and token boundaries from Lassy.
		offsets = []
		sdoc = ''
		for word in words:
			offsets.append(len(sdoc))
			sdoc += word.text
		offsets.append(len(sdoc))
		offsetidx = {m: n for n, m in enumerate(offsets)}

		seen = set()
		sonarmap = {}
		lassyrmap = {}
		sents = [(fname, gettokens(tree, 0, 999))
				for fname, tree in lassytrees.items()]
		queue = sorted(sents, key=lambda x: parsesentid(x[0]))
		ldoc = ''.join(''.join(sent) for _, sent in queue)
		if len(sdoc) != len(ldoc):
			print('unequal number of characters in '
					'sonar and lassy doc %s' % ldocname, file=sys.stderr)

		# first pass: align lassy sents with sonar sents, but go through
		# sonar just once, skip lassy sents that cannot be aligned
		m = 0
		while m < len(sdoc):
			for n, (fname, sent) in enumerate(queue):
				signature = ''.join(sent)
				if (sdoc[m:].startswith(signature)
						and m not in seen
						and m in offsetidx
						and m + len(signature) in offsetidx):
					break
			else:
				print('unalignable sonar tokens starting from ',
						sdoc[m:m + 100], '...', file=sys.stderr)
				break
			queue.pop(n)
			seen.update(range(m, m + len(signature)))
			# add lassy sent boundary at corresponding sonar word idx
			sentends.add(offsetidx[m + len(signature)] - 1)
			sonarmap[offsetidx[m + len(signature)] - 1] = fname
			aligntokens(words[offsetidx[m]:offsetidx[m + len(signature)]],
					sent, sdocname, fname, lassymap, lassyrmap)
			m += len(signature)

		# second pass: align any lassy sent which has not been aligned yet
		for fname, sent in queue:
			signature = ''.join(sent)
			m = -1
			while True:
				m = sdoc.find(signature, m + 1)
				if m == -1:
					print('could not align sentence: %r\n%s'
							% (signature, fname))
					lassyunaligned.append(fname)
					break
				if (m not in seen
						and m in offsetidx
						and m + len(signature) in offsetidx):
					break
			if m == -1:  # could not align sentence
				continue
			seen.update(range(m, m + len(signature)))
			# add lassy sent boundary at corresponding sonar word idx
			sentends.add(offsetidx[m + len(signature)] - 1)
			sonarmap[offsetidx[m + len(signature)] - 1] = fname
			aligntokens(words[offsetidx[m]:offsetidx[m + len(signature)]],
					sent, sdocname, fname, lassymap, lassyrmap)

		# collect unaligned sonar tokens
		for word in words:
			if word.get('id') not in lassyrmap:
				word.set('action', 'skip')
				sonarunaligned.append((sdocname, word.get('id')))
				print('unaligned sonar token:', word.text, word.get('id'),
						sep='\t', file=sys.stderr)

		# assign new parno, sentno to lassy sents
		parno = sentno = 0
		for n, word in enumerate(words):
			if n in sonarmap:
				fname = sonarmap[n]
				# try to parse lassy sent id, increment from prev parno/sentno
				match = re.search(r'.p\.(\d+)\.s.(\d+)', fname)
				if match:
					if parno != int(match.group(1)):
						parno += 1
						sentno = 1
					else:
						sentno += 1
				else:
					sentno += 1
				lassynewids[fname] = ldocname, parno, sentno

		if sonarnerpath:
			addnertolassy(sonarnerpath, outpath, sdocname, ldocname,
					words, idxmap, lassytrees, lassynewids, lassyrmap)

	else:  # SoNaR
		for markable in sentence:
			try:
				sentends.add(getspan(markable, idxmap, words)[1])
			except KeyError:  # ignore spans referring to non-existing tokens
				pass
	return sentends


def aligntokens(sonartokens, lassytokens, sdocname, fname,
		lassymap, lassyrmap):
	"""Align sonar tokens to lassy tokens; modifies sonar tokens in-place with
	an attribute "merge" if a token should merge with the next token.

	Example:
		"NWO / RU / Meertens instituut" => "NWO/RU/Meertens instituut"
	"""
	offsetidx = []
	toksignature = ''
	lassymap[fname] = []
	for n, token in enumerate(lassytokens):
		offsetidx.extend(n for _ in token + ' ')
		toksignature += token + ' '
		lassymap[fname].append((sdocname, []))
	toksignature = toksignature[:-1]

	lassyoffset = 0
	for token in sonartokens:
		if toksignature[lassyoffset:].startswith(token.text + ' '):
			lassymap[fname][offsetidx[lassyoffset]][1].append(
					token.get('id'))
			lassyrmap[token.get('id')] = (fname, offsetidx[lassyoffset])
			lassyoffset += len(token.text) + 1
		elif toksignature[lassyoffset:].startswith(token.text):
			# merge sonar token
			if not toksignature[lassyoffset:].endswith(token.text):
				token.set('action', 'merge')
			lassymap[fname][offsetidx[lassyoffset]][1].append(
					token.get('id'))
			lassyrmap[token.get('id')] = (fname, offsetidx[lassyoffset])
			lassyoffset += len(token.text)
		# split sonar token
		elif toksignature[lassyoffset:].replace(' ', '').startswith(token.text):
			orig = lassyoffset
			lassymap[fname][offsetidx[lassyoffset]][1].append(
					token.get('id'))
			lassyrmap[token.get('id')] = (fname, offsetidx[lassyoffset])
			for char in token.text:
				if toksignature[lassyoffset] == char:
					lassyoffset += 1
				elif (toksignature[lassyoffset] == ' '
						and toksignature[lassyoffset + 1] == char):
					lassyoffset += 2
					lassymap[fname][offsetidx[lassyoffset]][1].append(
							token.get('id'))
				else:
					raise ValueError
			token.set('action', 'split %s' % toksignature[orig:lassyoffset])
		else:
			raise ValueError('could not align tokens\n'
					'sonar: %s\nlassy: %s' % (
					' '.join(w.text for w in sonartokens), toksignature))


def dumplassymap(lassymap, lassynewids, lassyunaligned, sonarunaligned,
		outpath):
	"""Dump map of reordered lassy sents, and map of word/token boundaries"""
	with open(os.path.join(outpath, 'sentmap.tsv'), 'w'
			) as out1, open(os.path.join(outpath, 'tokmap.tsv'), 'w') as out2:
		print('orig', 'new', sep='\t', file=out1)
		print('lassysentid', 'lassytokenid', 'sonar_doc', 'sonar_word_id',
				sep='\t', file=out2)
		for fname, (ldocname, parno, sentno) in lassynewids.items():
			print('%s\t%s/%03d-%03d.xml' % (fname, ldocname, parno, sentno),
					file=out1)
			for tokidx, (sdocname, sonar_word_ids) in enumerate(lassymap[fname]):
				# (sentid, tokidx) => (sdocname, sonar_word_ids)
				print(os.path.basename(fname), tokidx,
						sdocname, ','.join(sonar_word_ids),
						sep='\t', file=out2)
	with open(os.path.join(outpath, 'lassy_unaligned_sents.txt'), 'w') as out:
		out.writelines(sentid + '\n' for sentid in lassyunaligned)
	with open(os.path.join(outpath, 'sonar_unaligned_tokens.tsv'), 'w') as out:
		out.writelines('%s\t%s\n' % (sdocname, wordid)
				for sdocname, wordid in sonarunaligned)


def writeconll(words, sentends, docname, inspan, out):
	"""Write tokens and coreference information in CoNLL format."""
	n = 0
	queue = []
	print('#begin document (%s); part 000' % docname, file=out)
	for m, word in enumerate(words):
		if word.get('action') == 'merge':
			queue.append(word)
			continue
		elif word.get('action') == 'skip':
			continue  # FIXME: close any coref tags...
		elif word.get('action', '').startswith('split'):
			subwords = word.get('action').split(' ')[1:]
			for subword in subwords[:-1]:
				print(docname, n, subword, '-', sep='\t', file=out)
				n += 1
			wordtext = subwords[-1]
			corefcol = word.get('coref', '-')
		elif queue:
			queue.append(word)
			wordtext = ''.join(w.text for w in queue)
			corefcol = simplify('|'.join(w.get('coref') for w in queue
					if w.get('coref'))) or '-'
			queue = []
		else:
			wordtext = word.text
			corefcol = word.get('coref', '-')
		print(docname, n, wordtext, corefcol, sep='\t', file=out)
		n += 1
		# ignore sent break if any span still open
		if m in sentends and m not in inspan:
			print(file=out)
			n = 0
	print('#end document', file=out)


def simplify(corefcol):
	"""Take the coref column for a sequence of merged tokens and
	simplify it.

	>>> simplify('(23|23)|23)')
	'(23)|23)'
	"""
	coreftags = corefcol.split('|')
	for n, a in enumerate(coreftags):
		if a.startswith('(') and not a.endswith(')'):
			if a[1:] + ')' in coreftags:
				coreftags[n] += ')'
				coreftags[coreftags.index(a[1:] + ')')] = ''
	return '|'.join(a for a in coreftags if a)


def addnertolassy(sonarnerpath, outpath, sdocname, ldocname, words,
		idxmap, lassytrees, lassynewids, lassyrmap):
	"""Reads SoNaR NER annotations and adds them to Lassy Small trees;
	writes re-numbered Lassy Small trees to <outdir>/lassy_renumbered/"""
	labelmap = {
			'eve': 'MISC',
			'pro': 'MISC',
			'misc': 'MISC',
			'loc': 'LOC',
			'org': 'ORG',
			'per': 'PER',
			}
	for label in labelmap:
		markablefile = '%s/MMAX/Markables/%s_%s_level.xml' % (
				sonarnerpath, sdocname, label)
		if not os.path.exists(markablefile):
			continue
		markables = etree.parse(markablefile).getroot()
		for markable in markables:
			start, end = getspan(markable, idxmap, words)
			for word in words[start:end + 1]:
				if word.get('id') not in lassyrmap:
					continue
				fname, tokidx = lassyrmap[word.get('id')]
				tree = lassytrees[fname]
				word = tree.find('.//node[@begin="%d"][@word]' % tokidx)
				word.set('neclass', labelmap[label])
	for fname, (ldocname1, parno, sentno) in lassynewids.items():
		if ldocname1 != ldocname:
			continue
		newdir = os.path.join(outpath, 'lassy_renumbered', ldocname)
		os.makedirs(newdir, exist_ok=True)
		lassytrees[fname].write('%s/%03d-%03d.xml' % (newdir, parno, sentno))


def conv(fname, inputdir, out, lassypath, sonarnerpath, outpath, lassymap,
		lassynewids, lassyunaligned, sonarunaligned):
	"""Convert a set of files for a single MMAX document to CoNLL."""
	words = etree.parse(fname).getroot()
	for word in words:
		# there are some instances of double escaped ampersands: &amp;amp;
		# lxml unescapes once, this takes care of the second time
		if '&amp;' in word.text:
			word.text = word.text.replace('&amp;', '&')
		if r'\[' in word.text or r'\]' in word.text:
			word.text = word.text.replace(r'\[', '[').replace(r'\]', ']')
	sdocname = os.path.basename(fname).replace('_words.xml', '')
	if os.path.exists('%s/Markables/%s_np_level.xml' % (inputdir, sdocname)):
		nplevel = etree.parse('%s/Markables/%s_np_level.xml'
				% (inputdir, sdocname)).getroot()
	elif os.path.exists(
			'%s/Markables/%s_coref_level.xml' % (inputdir, sdocname)):
		nplevel = etree.parse('%s/Markables/%s_coref_level.xml'
				% (inputdir, sdocname)).getroot()
	else:
		return  # no annotations exist for this file
	if os.path.exists(
			'%s/Markables/%s_sentence_level.xml' % (inputdir, sdocname)):
		sentence = etree.parse('%s/Markables/%s_sentence_level.xml'
				% (inputdir, sdocname)).getroot()
	else:
		sentence = None
	# some of the sonar docnames are missing a dash;
	# for consistency, we keep track of both sdocname and ldocname
	# in the output, we use the lassy docname which has the dash consistently.
	ldocname = sdocname
	if re.match(r'wiki\d+', ldocname):
		ldocname = 'wiki-' + ldocname[4:]
	# word IDs may be missing or contain decimals;
	# this maps ID labels to integer indices.
	idxmap = {word.attrib['id']: n
			for n, word in enumerate(words)}
	cluster = getclusters(nplevel)
	inspan = addclusters(words, nplevel, idxmap, cluster)
	sentends = getsents(words, sentence, idxmap, sdocname, ldocname,
			lassypath, sonarnerpath, outpath, lassymap, lassynewids,
			lassyunaligned, sonarunaligned)
	writeconll(words, sentends, ldocname, inspan, out)


def main():
	"""CLI."""
	try:
		opts, args = getopt.gnu_getopt(
				sys.argv[1:], '', ['lassy=', 'sonarner='])
		opts = dict(opts)
	except getopt.GetoptError:
		args = None
	if not args or len(args) > 2:
		print(__doc__, sep='\n')
		return
	lassypath = opts.get('--lassy')
	sonarnerpath = opts.get('--sonarner')
	if len(args) == 2:
		outpath = args[1]
		os.mkdir(outpath)
		if lassypath:
			outfile = os.path.join(outpath, 'sonar1-aligned.conll')
		else:
			outfile = os.path.join(outpath, 'sonar1.conll')
	out = None if len(args) == 1 else open(outfile, 'w', encoding='utf8')
	lassymap = lassynewids = lassyunaligned = sonarunaligned = None
	if lassypath:
		lassymap = defaultdict(list)
		lassynewids = {}
		lassyunaligned = []
		sonarunaligned = []
	try:
		for dirpath, dirnames, _ in os.walk(args[0]):
			if 'Basedata' in dirnames and 'Markables' in dirnames:
				pattern = os.path.join(dirpath, 'Basedata', '*.xml')
				for fname in sorted(glob(pattern)):
					if fname.endswith('Basedata/dummyfile_words.xml'):
						continue
					conv(fname, dirpath, out, lassypath, sonarnerpath,
							outpath, lassymap, lassynewids,
							lassyunaligned, sonarunaligned)
					# print('converted', os.path.basename(fname), file=sys.stderr)
	finally:
		if out:
			out.close()
	if '--lassy' in opts:
		dumplassymap(lassymap, lassynewids, lassyunaligned, sonarunaligned,
				outpath)


if __name__ == '__main__':
	main()
