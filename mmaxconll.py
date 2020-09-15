"""Convert Corea/SoNaR coreference annotations to CoNLL 2012 format.

Usage: mmaxconll.py [options] <inputdir> <outputdir>

inputdir is searched recursively for Basedata/ and Markables/ subdirectories.
outputdir should not exist.

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
  --split=<file>     Specify a CSV file describing a train/dev/test split.
                     Splits Lassy trees across train, dev and test directories,
                     with corresponding .conll files. For SoNaR, use:
https://gist.github.com/CorbenPoot/ee1c97209cb9c5fc50f9528c7fdcdc93"""
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


def addclusters(words, nplevel, idxmap, cluster, sentends,
		skiptypes=('bridge', )):
	"""Add start and end tags of markables to the respective tokens."""
	seen = set()  # don't add same span twice
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
		# skip spans that cross sentence boundaries
		elif not any(n in sentends for n in range(start, end)):
			coref = ('(%s|%s' % (cid, cur)) if cur else ('(%s' % cid)
			words[start].set('coref', coref)
			cur = words[end].get('coref', '')
			coref = ('%s|%s)' % (cur, cid)) if cur else ('%s)' % cid)
			words[end].set('coref', coref)


def parsesentid(fname):
	"""Create sort key with padding from Lassy filename."""
	return (tuple(map(int, re.findall(r'\d+', os.path.basename(fname))))
			+ (0, 0, 0, 0, 0, 0, 0))[:7]


def normalizedocname(docname):
	"""Add dash that is missing from some of the sonar docnames."""
	return re.sub(r'wiki(\d+)', r'wiki-\1', docname)


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

		# correct errors in Sonar
		if sdocname == 'dpc-eup-000015-nl-sen':
			words[idxmap['word_675']].text = ']'
		if sdocname == 'WS-U-E-A-0000000036':
			words[idxmap['word_26']].text = 'Spee'

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
		lassyrevmap = {}
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
				if fname.endswith('WR-P-P-C-0000000054.txt-17.10.xml'):
					continue
				signature = ''.join(sent)
				if (sdoc[m:].startswith(signature)
						and m not in seen
						and m in offsetidx
						and m + len(signature) in offsetidx):
					queue.pop(n)
					seen.update(range(m, m + len(signature)))
					# add lassy sent boundary at corresponding sonar word idx
					sentends.add(offsetidx[m + len(signature)] - 1)
					sonarmap[offsetidx[m + len(signature)] - 1] = fname
					start, end = offsetidx[m], offsetidx[m + len(signature)]
					aligntokens(words[start:end], sent, sdocname, fname,
							lassymap, lassyrevmap)
					m += len(signature)
					break
			else:  # for ends without break
				print('unalignable sonar tokens starting from ',
						sdoc[m:m + 100], '...', file=sys.stderr)
				break  # break out of while loop

		# second pass: align any lassy sent which has not been aligned yet
		for fname, sent in queue:
			signature = ''.join(sent)
			m = -1
			while True:
				m = sdoc.find(signature, m + 1)
				if fname.endswith('WR-P-P-C-0000000054.txt-17.10.xml'):
					m = -1
				if m == -1:
					print('could not align sentence: %r\n%s'
							% (signature, fname))
					lassyunaligned.append((
							os.path.basename(fname), ' '.join(sent)))
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
			start, end = offsetidx[m], offsetidx[m + len(signature)]
			aligntokens(words[start:end], sent, sdocname, fname,
					lassymap, lassyrevmap)

		# collect unaligned sonar tokens
		for word in words:
			if word.get('id') not in lassyrevmap:
				word.set('action', 'skip')
				idxmap.pop(word.get('id'))  # block spans with this token
				sonarunaligned.append((sdocname, word.get('id'), word.text))
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
			addnertolassy(sonarnerpath, sdocname, words, idxmap, lassytrees,
					lassyrevmap)

		for fname, (ldocname1, parno, sentno) in lassynewids.items():
			if ldocname1 != ldocname:
				continue
			newdir = os.path.join(outpath, 'lassy_renumbered', ldocname)
			os.makedirs(newdir, exist_ok=True)
			lassytrees[fname].write('%s/%03d-%03d.xml'
					% (newdir, parno, sentno))
	else:  # SoNaR
		for markable in sentence:
			try:
				sentends.add(getspan(markable, idxmap, words)[1])
			except KeyError:  # ignore spans referring to non-existing tokens
				pass
	return sentends


def aligntokens(sonartokens, lassytokens, sdocname, fname,
		lassymap, lassyrevmap):
	"""Align sonar tokens to lassy tokens; modifies sonar tokens in-place with
	an attribute "merge" if a token should merge with the next token.

	Examples (sonar => lassy):
		merge NWO / RU / Meertens instituut => NWO/RU/Meertens instituut
		split Matthaeus"opleidingsprogramma => Matthaeus " opleidingsprogramma
	"""
	offsetidx = []
	toksignature = ''
	lassymap[fname] = []
	for n, token in enumerate(lassytokens):
		offsetidx.extend(n for _ in token + ' ')
		toksignature += token + ' '
		lassymap[fname].append([sdocname, [], '', ''])
	toksignature = toksignature[:-1]

	lassyoffset = 0
	for token in sonartokens:
		if toksignature[lassyoffset:].startswith(token.text + ' '):
			lassymap[fname][offsetidx[lassyoffset]][1].append(
					token.get('id'))
			lassymap[fname][offsetidx[lassyoffset]][3] += token.text
			lassyrevmap[token.get('id')] = (fname, offsetidx[lassyoffset])
			lassyoffset += len(token.text) + 1
		elif toksignature[lassyoffset:].startswith(token.text):
			# last token of sentence
			if toksignature[lassyoffset:] == token.text:
				lassymap[fname][offsetidx[lassyoffset]][1].append(
						token.get('id'))
				lassymap[fname][offsetidx[lassyoffset]][3] = token.text
			else:  # merge sonar token w/next sonar token
				token.set('action', 'merge')
				lassymap[fname][offsetidx[lassyoffset]][1].append(
						token.get('id'))
				lassymap[fname][offsetidx[lassyoffset]][2] = 'merge'
				lassymap[fname][offsetidx[lassyoffset]][3] += token.text + ' '
			lassyrevmap[token.get('id')] = (fname, offsetidx[lassyoffset])
			lassyoffset += len(token.text)
		# split sonar token
		elif toksignature[lassyoffset:].replace(' ', '').startswith(
				token.text):
			origlassyoffset = prevlassyoffset = lassyoffset
			for char in token.text:
				if toksignature[lassyoffset] == char:
					lassyoffset += 1
				elif (toksignature[lassyoffset] == ' '
						and toksignature[lassyoffset + 1] == char):
					lassymap[fname][offsetidx[prevlassyoffset]][1].append(
							token.get('id'))
					lassymap[fname][offsetidx[prevlassyoffset]][2] = 'split'
					lassymap[fname][offsetidx[prevlassyoffset]][3] = (
							toksignature[prevlassyoffset:lassyoffset])
					prevlassyoffset = lassyoffset + 1
					lassyoffset += 2
				else:
					raise ValueError
			ltokenidx = offsetidx[prevlassyoffset]
			lassymap[fname][ltokenidx][1].append(token.get('id'))
			lassymap[fname][ltokenidx][2] = 'split'
			lassymap[fname][ltokenidx][3] = toksignature[
					prevlassyoffset:lassyoffset]
			lassyrevmap[token.get('id')] = (fname, ltokenidx)
			token.set('action', 'split %s'
					% toksignature[origlassyoffset:lassyoffset])
			lassyoffset += 1
		else:
			raise ValueError('could not align tokens\n'
					'sonar: %s\nlassy: %s' % (
					' '.join(w.text for w in sonartokens), toksignature))


def addnertolassy(sonarnerpath, sdocname, words, idxmap, lassytrees,
		lassyrevmap):
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
			try:
				start, end = getspan(markable, idxmap, words)
			except KeyError:  # ignore spans referring to non-existing tokens
				continue
			for word in words[start:end + 1]:
				if word.get('id') not in lassyrevmap:
					continue
				fname, tokidx = lassyrevmap[word.get('id')]
				tree = lassytrees[fname]
				word = tree.find('.//node[@begin="%d"][@word]' % tokidx)
				word.set('neclass', labelmap[label])


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


def writeconll(words, sentends, docname, out):
	"""Write tokens and coreference information in CoNLL 2012 format."""
	part = tokenid = 0
	queue = []
	print('#begin document (%s); part 000' % docname, file=out)
	for n, word in enumerate(words):
		if word.get('action') == 'skip':
			continue
		elif word.get('action') == 'merge':
			queue.append(word)
			continue
		elif word.get('action', '').startswith('split'):
			subwords = word.get('action').split(' ')[1:]
			for subword in subwords[:-1]:
				print(docname, part, tokenid, subword, *(['-'] * 6), '*',
						'-', sep='\t', file=out)
				tokenid += 1
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
		print(docname, part, tokenid, wordtext, *(['-'] * 6), '*',
				corefcol, sep='\t', file=out)
		tokenid += 1
		if n in sentends:
			print(file=out)
			tokenid = 0
	print('#end document', file=out)


def conv(fname, inputdir, lassypath, sonarnerpath, outpath, lassymap,
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
	ldocname = normalizedocname(sdocname)
	# word IDs may be missing or contain decimals;
	# this maps ID labels to integer indices.
	idxmap = {word.attrib['id']: n
			for n, word in enumerate(words)}
	cluster = getclusters(nplevel)
	sentends = getsents(words, sentence, idxmap, sdocname, ldocname,
			lassypath, sonarnerpath, outpath, lassymap, lassynewids,
			lassyunaligned, sonarunaligned)
	addclusters(words, nplevel, idxmap, cluster, sentends)
	conllfile = os.path.join(outpath, 'coref', ldocname + '.conll')
	with open(conllfile, 'w') as out:
		writeconll(words, sentends, ldocname, out)
	if lassypath:  # add columns with POS, NER, and parse tree to CoNLL file
		from addparsebits import convalpino
		parsesdir = os.path.join(outpath, 'lassy_renumbered', ldocname)
		convalpino(conllfile, parsesdir)


def dumplassymap(lassymap, lassynewids, lassyunaligned, sonarunaligned,
		outpath):
	"""Dump map of reordered lassy sents, and map of word/token boundaries"""
	with open(os.path.join(outpath, 'sentmap.tsv'), 'w') as sentmap, \
			open(os.path.join(outpath, 'tokmap.tsv'), 'w') as tokmap:
		print('orig', 'new', sep='\t', file=sentmap)
		print('lassysentid', 'lassytokenid', 'sonar_doc', 'sonar_word_id',
				'action', 'token', sep='\t', file=tokmap)
		for fname, (ldocname, parno, sentno) in lassynewids.items():
			print('%s\t%s/%03d-%03d.xml' % (fname, ldocname, parno, sentno),
					file=sentmap)
			for tokidx, (sdocname, sword_ids, action, token) in enumerate(
					lassymap[fname]):
				# (sentid, tokidx) => (sdocname, sword_ids)
				print(os.path.basename(fname), tokidx,
						sdocname, ','.join(sword_ids), action, token,
						sep='\t', file=tokmap)
	with open(os.path.join(outpath, 'lassy_unaligned_sents.tsv'), 'w') as out:
		out.writelines('%s\t%s\n' % (sentid, sent)
				for sentid, sent in lassyunaligned)
	with open(os.path.join(outpath, 'sonar_unaligned_tokens.tsv'), 'w') as out:
		out.writelines('%s\t%s\t%s\n' % (sdocname, wordid, word)
				for sdocname, wordid, word in sonarunaligned)


def makesplit(fname, outpath):
	"""Divide CoNLL file and trees in train/dev/test according to CSV file."""
	with open(fname) as inp:
		lines = [line.strip().split(',') for line in inp]
	if not all(b in {'dev', 'test', 'train'} for _, b in lines):
		raise ValueError('second column should only contain: dev, test, train')
	split = {name: {normalizedocname(a) for a, b in lines if b == name}
			for name in ('dev', 'test', 'train')}
	if (split['dev'] & split['train']) or (split['train'] & split['test']):
		raise ValueError('overlap in dev/train or train/test')

	for name, docs in split.items():
		os.mkdir(os.path.join(outpath, name))
		with open('%s/%s.conll' % (outpath, name), 'w') as out:
			for doc in docs:
				conllfile = os.path.join(outpath, 'coref', doc + '.conll')
				with open(conllfile) as inp:
					out.write(inp.read())
				os.symlink(
						os.path.join('..', 'lassy_renumbered', doc),
						os.path.join(outpath, name, doc))


def main():
	"""CLI."""
	try:
		opts, args = getopt.gnu_getopt(
				sys.argv[1:], '', ['lassy=', 'sonarner=', 'split='])
		opts = dict(opts)
	except getopt.GetoptError:
		args = None
	if not args or len(args) != 2:
		print(__doc__, sep='\n')
		return
	inpath, outpath = args
	os.makedirs(os.path.join(outpath, 'coref'), exist_ok=False)
	lassymap = lassynewids = lassyunaligned = sonarunaligned = None
	lassypath = opts.get('--lassy')
	sonarnerpath = opts.get('--sonarner')
	if lassypath:
		lassymap = defaultdict(list)
		lassynewids = {}
		lassyunaligned = []
		sonarunaligned = []
	for dirpath, dirnames, _ in os.walk(inpath):
		if 'Basedata' in dirnames and 'Markables' in dirnames:
			pattern = os.path.join(dirpath, 'Basedata', '*.xml')
			for fname in sorted(glob(pattern), key=normalizedocname):
				if fname.endswith('Basedata/dummyfile_words.xml'):
					continue
				conv(fname, dirpath, lassypath, sonarnerpath,
						outpath, lassymap, lassynewids,
						lassyunaligned, sonarunaligned)
	if '--lassy' in opts:
		dumplassymap(lassymap, lassynewids, lassyunaligned, sonarunaligned,
				outpath)
	if '--split' in opts:
		makesplit(opts.get('--split'), outpath)


if __name__ == '__main__':
	main()
