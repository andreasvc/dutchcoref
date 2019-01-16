"""Dutch coreference resolution & dialogue analysis using deterministic rules.

Usage: python3 coref.py [options] <directory>
where directory contains .xml files with sentences parsed by Alpino.
Output is sent to STDOUT.

Options:
	--help        this message
	--verbose     debug output (sent to STDOUT)
	--slice=N:M   restrict input with a Python slice of sentence numbers
	--fmt=<minimal|semeval2010|conll2012|booknlp|html>
		output format:
			:minimal: doc ID, token ID, token, and coreference columns
			:booknlp: tabular format with universal dependencies and dialogue
				information in addition to coreference.
			:html: interactive HTML visualization with coreference and dialogue
				information.

Instead of giving a directory, can use one of the following presets:
	--clindev     run on CLIN26 shared task development data
	--semeval     run on SEMEVAL 2010 development data
	--test        run tests
"""
import os
import re
import sys
import tempfile
import subprocess
from collections import defaultdict
from itertools import islice
from bisect import bisect
from getopt import gnu_getopt
from datetime import datetime
from glob import glob
from lxml import etree
import colorama


STOPWORDS = (
		# List of Dutch Stop words (http://www.ranks.nl/stopwords/dutch)
		'aan af al als bij dan dat de die dit een en er had heb hem het hij '
		'hoe hun ik in is je kan me men met mij nog nu of ons ook te tot uit '
		'van was wat we wel wij zal ze zei zij zo zou '
		# titles
		'dr. drs. ing. ir. mr. lic. prof. mevr. mw. bacc. kand. dr.h.c. ds. '
		'bc. dr drs ing ir mr lic prof mevr mw bacc kand dr.h.c ds bc '
		'mevrouw meneer heer doctor professor').split()

# Reported speech verbs as they appear in the "root" attribute.
SPEECHVERBS = frozenset((
		'begin onthul loof kondig_aan beweer breng_in deel_mee merk_op zeg_op '
		'spreek breng_uit druk_uit uit spreek_uit verklaar verkondig vermeld '
		'vertel verwoord duid_aan benoem tendeer_in meen noem oordeel stel '
		'vind beduid behels beteken bewijs beveel gebied draag_op neem_aan '
		'veronderstel merk_aan verwijt beloof zeg_toe schrijf_voor geef_aan '
		'stip_aan kondig_af maak_bekend bericht beschrijf declareer gewaag '
		'meld geef_op neem_op teken_op relateer proclameer publiceer sus '
		'rapporteer leg_vast vernoem versla zeg betoon betuig manifesteer '
		'openbaar peer_op slaak spui sla_uit stort_uit stoot_uit vertolk '
		'ventileer deponeer expliciteer getuig ontvouw pretendeer doe_uiteen '
		'zet_uiteen verzeker kleed_in lucht breng_over toon vervat geef_weer '
		'betoog claim suggereer houd_vol geef_voor wend_voor verdedig vraag '
		'spreek_aan bestempel betitel kwalificeer som_op draai_af debiteer '
		'deel_mede dis_op kraam_uit verhaal poneer postuleer leg_voor beval '
		'fluister voorspel roep antwoord voeg reageer merk benadruk herhaal '
		'vervolg verzucht klaag protesteer stotter sis grom brom brul snauw '
		'schreeuw begin opper mompel loog onderbreek interrumpeer smeek gil '
		'mopper constateer beaam besluit concludeeer vul_aan informeer zucht '
		'waarschuw verduidelijk stamel beken hijg kreun jammer bulder krijs '
		'snik prevel bevestig grinnik verontschuldig grap murmel bries '
		'piep kir ').split())
WEATHERVERBS = ('dooien gieten hagelen miezeren misten motregenen onweren '
		'plenzen regenen sneeuwen stormen stortregenen ijzelen vriezen '
		'weerlichten winteren zomeren').split()
VERBOSE = False
HTMLTEMPLATE = """<!DOCTYPE html><html>
<head><meta charset=utf-8>
<style>
body { margin: 3em; max-width: 50em; }
span { font-weight: bold; }
span.q { color: green; font-weight: normal; }
</style>
<script>
function highlight(ev) {
	var cls = ev.target.className;
	var elems = document.getElementsByClassName(cls)
	for (var n in elems)
		elems[n].style = 'background: yellow; ';
}
function unhighlight(ev) {
	var cls = ev.target.className;
	var elems = document.getElementsByClassName(cls)
	for (var n in elems)
		elems[n].style = '';
}
function addhighlighting() {
	var elems = document.getElementsByTagName('span');
	for (var n in elems) {
		if (elems[n].className.charAt(0) == 'c') {
			elems[n].onmouseover = highlight;
			elems[n].onmouseout = unhighlight;
		}
	}
}
function hl1(id) {
	document.getElementById(id).style = 'background: lightblue; ';
}
function hl2(id) {
	document.getElementById(id).style = 'background: lightcoral; ';
}
function nohl(id) {
	document.getElementById(id).style = '';
}
</script>
<title>%s coref</title>
</head>
<body onLoad="addhighlighting()">
<p>Legend:
<span style="background: yellow">[ Coreference ]</span>
<span style="background: lightblue">[ Speaker ]</span>
<span style="background: lightcoral">[ Addressee ]</span></p>\n
"""


class Mention:
	"""A span referring to an entity."""
	def __init__(self, mentionid, sentno, tree, node, begin, end, headidx,
			tokens, ngdata, gadata):
		self.sentno = sentno
		self.node = node
		self.begin = begin
		self.end = end
		self.clusterid = mentionid
		self.tokens = tokens
		self.id = mentionid
		self.prohibit = set()
		removeids = {n for rng in
				(range(int(child.get('begin')), int(child.get('end')))
				for child in node.findall('.//node[@rel="app"]')
						+ node.findall('.//node[@rel="mod"]'))
					for n in rng if n > headidx}
		# without mod/app constituents after head
		self.relaxedtokens = [token.get('word') for token
				in sorted((token for token in tree.findall('.//node[@word]')
					if begin <= int(token.get('begin')) < end
					and int(token.get('begin')) not in removeids),
				key=lambda x: int(x.get('begin')))]
		if not self.relaxedtokens:
			self.relaxedtokens = self.tokens
		self.head = (node.find('.//node[@begin="%d"]' % headidx)
				if len(node) else node)
		if node.get('pdtype') == 'pron' or node.get('vwtype') == 'bez':
			self.type = 'pronoun'
		elif (self.head.get('ntype') == 'eigen'
				or self.head.get('pt') == 'spec'):
			self.type = 'name'
		else:
			self.type = 'noun'
		self.mainmod = [a.get('word') for a
				in (node.findall('.//node[@word]') if len(node) else (node, ))
				if a.get('rel') == 'mod' or a.get('pt') in ('adj', 'n')
				and begin <= int(a.get('begin')) < end]
		self.features = {
				'human': None, 'gender': None,
				'number': None, 'person': None}
		self.features['number'] = self.head.get(
				'rnum', self.head.get('num'))
		if self.features['number'] is None and 'getal' in self.head.keys():
			self.features['number'] = {
					'ev': 'sg', 'mv': 'pl', 'getal': 'both'
					}[self.head.get('getal')]
		if self.head.get('genus') in ('masc', 'fem'):
			self.features['gender'] = self.head.get('genus')[0]
			self.features['human'] = 1
		elif (self.head.get('genus') == 'onz'
				or self.head.get('gen') == 'het'):
			self.features['gender'] = 'n'
			self.features['human'] = 0

		if self.type == 'pronoun':  # pronouns: rules
			if self.head.get('persoon')[0] in '123':
				self.features['person'] = self.head.get('persoon')[0]
			if self.features['person'] in ('1', '2'):
				self.features['gender'] = 'mf'
				self.features['human'] = 1
			elif self.head.get('persoon') == '3p':
				self.features['gender'] = 'mf'
				self.features['human'] = 1
			elif self.head.get('persoon') == '3o':
				self.features['gender'] = 'n'
				self.features['human'] = 0
			if self.head.get('lemma') == 'haar':
				self.features['gender'] = 'f'
				self.features['human'] = 1
			elif self.head.get('lemma') == 'zijn':
				self.features['gender'] = 'mn'
		# nouns: use lexical resource
		elif self.head.get('lemma', '').replace('_', '') in gadata:
			gender, animacy = gadata[self.head.get(
					'lemma', '').replace('_', '')]
			if animacy == 'human':
				self.features['human'] = 1
				self.features['gender'] = 'mf'
				if gender in ('m', 'f'):
					self.features['gender'] = gender
			else:
				self.features['human'] = 0
				self.features['gender'] = 'n'
		else:  # names: dict
			if self.head.get('neclass') == 'PER':
				self.features['human'] = 1
				self.features['gender'] = 'mf'
			elif self.head.get('neclass') is not None:
				self.features['human'] = 0
				self.features['gender'] = 'n'
			if ' '.join(tokens).lower() in ngdata:
				self._nglookup(' '.join(tokens), ngdata)
			elif (self.head.get('neclass') == 'PER'
					and tokens[0] not in STOPWORDS):
				# Assume first token is first name.
				self._nglookup(tokens[0], ngdata)

	def _nglookup(self, key, ngdata):
		genderdata = ngdata.get(key.lower())
		if not key or genderdata is None:
			return
		genderdata = [int(x) for x in genderdata.split(' ')]
		self.features['number'] = 'sg'
		if (genderdata[0] > sum(genderdata) / 3
				and genderdata[1] > sum(genderdata) / 3):
			self.features['gender'] = 'mf'
			self.features['human'] = 1
		elif genderdata[0] > sum(genderdata) / 3:
			self.features['gender'] = 'm'
			self.features['human'] = 1
		elif genderdata[1] > sum(genderdata) / 3:
			self.features['gender'] = 'f'
			self.features['human'] = 1
		elif genderdata[2] > sum(genderdata) / 3:
			self.features['gender'] = 'n'
			self.features['human'] = 0
		elif genderdata[3] > sum(genderdata) / 3:
			self.features['number'] = 'pl'
			self.features['gender'] = 'n'

	def __str__(self):
		return '\'%s\' %s inquote=%s' % (
				color(' '.join(self.tokens), 'green'),
				' '.join('%s=%s' % (a, '?' if b is None else b)
					for a, b in self.features.items()),
				int(self.head.get('quotelabel') == 'I'))


class Quotation:
	def __init__(self, start, end, sentno, parno, text, sentbounds):
		self.start, self.end = start, end  # global token indices
		self.sentno = sentno  # global sentence index (ignoring paragraphs)
		self.parno = parno  # paragraph number
		self.sentbounds = sentbounds  # quote starts+ends at sent boundaries
		self.speaker = None  # detected speaker mention
		self.addressee = None  # detected addressee mention
		self.mentions = []  # list of mentions within this quote
		self.text = text  # text of quote (including quote marks)


def debug(*args, **kwargs):
	if VERBOSE:
		print(*args, **kwargs)


def color(text, c):
	"""Returns colored text."""
	if c == 'red':
		return colorama.Fore.RED + text + colorama.Fore.RESET
	elif c == 'green':
		return colorama.Fore.GREEN + text + colorama.Fore.RESET
	elif c == 'yellow':
		return colorama.Fore.YELLOW + text + colorama.Fore.RESET
	raise ValueError


def parsesentid(path):
	"""Given a filename, return tuple with numeric components for sorting."""
	filename = os.path.basename(path)
	x = tuple(map(int, re.findall('[0-9]+', filename.rsplit('.', 1)[0])))
	if len(x) == 1:
		return 0, x[0]
	elif len(x) == 2:
		return x
	else:
		raise ValueError('expected sentence ID of the form sentno.xml '
				'or parno-sentno.xml. Got: %s' % filename)


def sortmentions(mentions):
	"""Sort mentions by start position, then from small to large span length.
	"""
	return sorted(mentions,
			key=lambda x: (x.sentno, x.begin, x.end))


def getheadidx(node):
	"""Return head word index given constituent."""
	if len(node) == 0:
		return int(node.get('begin'))
	for child in node:
		if child.get('rel') in ('hd', 'whd', 'rhd', 'crd', 'cmp'):
			return getheadidx(child)
	# default to last child as head
	return getheadidx(node[-1])


def prohibited(mention1, mention2, clusters):
	"""Check if there is a constraint against merging mention1 and mention2."""
	if (mention1.clusterid == mention2.clusterid
			or not clusters[mention1.clusterid].isdisjoint(mention2.prohibit)
			or not clusters[mention2.clusterid].isdisjoint(mention1.prohibit)):
		return True
	return False


def merge(mention1, mention2, mentions, clusters):
	"""Merge cluster1 & cluster2, delete cluster with highest ID."""
	if mention1 is mention2:
		raise ValueError
	if mention1.clusterid == mention2.clusterid:
		return
	if mention1.clusterid > mention2.clusterid:
		mention1, mention2 = mention2, mention1
	mergefeatures(mention1, mention2)
	mention1.prohibit.update(mention2.prohibit)
	cluster1 = clusters[mention1.clusterid]
	cluster2 = clusters[mention2.clusterid]
	clusters[mention2.clusterid] = None
	cluster1.update(cluster2)
	for m in cluster2:
		mentions[m].clusterid = mention1.clusterid
	debug('Linked  %d %d %s\n\t%d %d %s' % (
			mention1.sentno, mention1.begin, mention1,
			mention2.sentno, mention2.begin, mention2))


def mergefeatures(mention, other):
	"""Update the features of the first mention with those of second.
	In case one is more specific than the other, keep specific value.
	In case of conflict, keep both values."""
	for key in mention.features:
		if (key == 'person' or mention.features[key] == other.features[key]
				or other.features[key] in (None, 'both')):
			pass
		elif mention.features[key] in (None, 'both'):
			mention.features[key] = other.features[key]
		elif key == 'human':
			mention.features[key] = None
		elif key == 'number':
			mention.features[key] = 'both'
		elif key == 'gender':
			if (mention.features[key] == 'mf'
					and other.features[key] in ('m', 'f')):
				mention.features[key] = other.features[key]
			elif (mention.features[key] == 'mn'
					and other.features[key] in ('m', 'n')):
				mention.features[key] = other.features[key]
			elif (mention.features[key] in ('m', 'f')
					and other.features[key] in ('m', 'f')):
				mention.features[key] = 'mf'
			elif (mention.features[key] in ('m', 'n')
					and other.features[key] in ('m', 'n')):
				mention.features[key] = 'mn'
			elif (mention.features[key] in ('m', 'f')
					and other.features[key] == 'mf'):
				pass
			elif (mention.features[key] in ('m', 'n')
					and other.features[key] == 'mn'):
				pass
			else:
				mention.features[key] = None
	other.features.update((a, b) for a, b in mention.features.items()
			if a != 'person')


def compatible(mention, other):
	"""Return True if all features are compatible."""
	return all(
			mention.features[key] == other.features[key]
			or None in (mention.features[key], other.features[key])
			or (key == 'gender'
				and 'mf' in (mention.features[key], other.features[key])
				and 'n' not in (mention.features[key], other.features[key]))
			or (key == 'gender'
				and 'mn' in (mention.features[key], other.features[key])
				and 'f' not in (mention.features[key], other.features[key]))
			or (key == 'number'
				and 'both' in (mention.features[key], other.features[key]))
			for key in mention.features)


def iwithini(mention, other):
	"""Check whether spans overlap."""
	return (mention.sentno == other.sentno
			and (mention.begin <= other.begin <= mention.end
				or mention.begin <= other.end <= mention.end
				or other.begin <= mention.begin <= other.end
				or other.begin <= mention.end <= other.end))


def checkconstraints(mention, clusters):
	"""Block coreference for first mention of indefinite NP or bare plural."""
	if len(clusters[mention.clusterid]) > 1:
		return True
	# indefinite pronoun/article
	if (mention.node.get('cat') == 'np'
			and (mention.node[0].get('def') == 'indef'
				or mention.node[0].get('vwtype') == 'onbep')):
		return False
	# bare plural
	if (mention.node.get('ntype') == 'soort'
			and mention.features['number'] == 'pl'):
		return False
	return True


def mentionselection(mentions, clusters):
	"""Yield the first mention for each cluster."""
	for cluster in clusters:
		if cluster is not None:
			n = min(cluster)
			yield n, mentions[n]


def considermention(node, tree, sentno, mentions, covered, ngdata, gadata,
		precise):
	"""Decide whether a candidate mention should be added.

	:param precise: whether to include 'precise constructs' (reflexive,
		relative, reciprocal pronouns, appositives, nominal predicates).
	"""
	if len(node) == 0 and 'word' not in node.keys():
		return
	if not precise:
		if (node.get('vwtype') in ('betr', 'refl', 'recip')
				or node.get('rel') in ('app', 'predc')):
			return
	headidx = getheadidx(node)
	indices = sorted(int(token.get('begin')) for token
			in (node.findall('.//node[@word]') if len(node) else [node]))
	a, b = min(indices), max(indices) + 1
	# allow comma when preceded by conjunct, adjective, or location.
	for punct in tree.getroot().findall('./node/node[@pt="let"]'):
		i = int(punct.get('begin'))
		if (a <= i < b
			and (node.find('.//node[@begin="%d"][@rel="cnj"]' % (i - 1))
					is not None
				or node.find('.//node[@begin="%d"][@pt="adj"]' % (i - 1))
					is not None
				or node.find('.//node[@begin="%d"][@neclass="LOC"]' % (i - 1))
					is not None)):
			indices.append(i)
	indices.sort()
	# if span is interrupted by a discontinuity from other words or
	# punctuation, cut off mention before it; avoids weird long mentions.
	if indices != list(range(a, b)):
		b = min(n for n in range(a, b) if n not in indices)
		if headidx > b:
			headidx = max(int(a.get('begin')) for a
					in node.findall('.//node[@word]')
					if int(a.get('begin')) < b)
	relpronoun = node.find('./node[@cat="rel"]/node[@wh="rel"]')
	if relpronoun is not None and int(relpronoun.get('begin')) < b:
		b = int(relpronoun.get('begin'))
		if headidx > b:
			headidx = max(int(a.get('begin')) for a
					in node.findall('.//node[@word]')
					if int(a.get('begin')) < b)
	# NP without appositive: John in "John, the painter"
	if precise and len(node) > 1 and node[1].get('rel') == 'app':
		node = node[0]
		b = int(node.get('end'))
	tokens = [token.get('word') for token
			in sorted((token for token in tree.findall('.//node[@word]')
				if a <= int(token.get('begin')) < b),
			key=lambda x: int(x.get('begin')))]
	if tokens[0] in ',\'"()':
		tokens = tokens[1:]
		a += 1
	if tokens[-1] in ',\'"()':
		tokens = tokens[:-1]
		b -= 1
	head = (node.find('.//node[@begin="%d"]' % headidx)
			if len(node) else node)
	# various
	if head.get('lemma') in ('aantal', 'keer', 'toekomst', 'manier'):
		return
	# pleonastic it
	if node.get('rel') in ('sup', 'pobj1'):
		return
	if node.get('rel') == 'su' and node.get('lemma') == 'het':
		hd = node.find('../node[@rel="hd"]')
		# het regent. / hoe gaat het?
		if hd.get('lemma') in WEATHERVERBS or hd.get('lemma') == 'gaan':
			return
		if (hd.get('lemma') == 'ontbreken'
				and node.find('../node[@rel="pc"]'
					'/node[@rel="hd"][@lemma="aan"]') is not None):
			return
		# het kan voorkomen dat ...
		if (node.get('index') and node.get('index')
				in node.xpath('../node//node[@rel="sup"]/@index')):
			return
	if node.get('rel') == 'obj1' and node.get('lemma') == 'het':
		hd = node.find('../node[@rel="hd"]')
		hd = '' if hd is None else hd.get('lemma')
		# (60) de presidente had het warm
		if hd == 'hebben' and node.find('../node[@rel="predc"]') is not None:
			return
		# (61) samen zullen we het wel rooien.
		if hd == 'rooien':
			return
		# (62) hij zette het op een lopen
		if (hd == 'zetten' and node.find('../node[@rel="svp"]/'
				'node[@word="lopen"]') is not None):
			return
		# (63) had het op mij gemunt.
		if hd == 'munten' and node.find('..//node[@word="op"]') is not None:
			return
		# (64) het erover hebben
		if (hd == 'hebben'
				and (node.find('../node[@word="erover"]') is not None
					or (node.find('..//node[@word="er"]') is not None
						and node.find('..//node[@word="over"]') is not None))):
			return
	if (headidx not in covered
			# discard measure phrases
			and node.find('.//node[@num="meas"]') is None
			and node.get('num') != "meas"
			and node.find('./node[@pt="tw"]') is None
			# and not tokens[0].isnumeric()
			# "a few" ...
			and (node.get('cat') != 'np' or node.get('rel') != 'det')
			# which in "I won't say which restaurant"
			and node.find('.//node[@begin="%d"][@vwtype="onbep"]' % a) is None
			and node.find('.//node[@begin="%d"][@vwtype="vb"]' % a) is None
			# "something"
			and node.get('vwtype') not in ('onbep', 'vb')
			# temporal expressions
			and head.get('special') != 'tmp' and node.get('special') != 'tmp'
			# partitive / quantifier
			# ongeveer 12 dollar
			and node.find('./node[@sc="noun_prep"]') is None
			# and (node.get('cat') != 'np'
			# 	or node[0].get('pos') not in ('adj', 'noun'))
			):
		mentions.append(Mention(
				len(mentions), sentno, tree, node, a, b, headidx,
				tokens, ngdata, gadata))
		covered.add(headidx)
		# California in "San Jose, California"
		if (node.get('cat') == 'mwu' and head.get('neclass') == 'LOC'
				and ',' in tokens):
			mentions.append(Mention(
					len(mentions), sentno, tree, node,
					a + tokens.index(',') + 1, b, b - 1,
					tokens[tokens.index(',') + 1:],
					ngdata, gadata))
		elif len(node) > 1 and node[0].get('rel') == 'cnj':
			mentions[-1].features['number'] = 'pl'
			for cnj in node.findall('./node[@rel="cnj"]'):
				a = int(cnj.get('begin'))
				b = int(cnj.get('end'))
				tokens = [token.get('word') for token
						in sorted((token for token
							in tree.findall('.//node[@word]')
							if a <= int(token.get('begin')) < b),
						key=lambda x: int(x.get('begin')))]
				mentions.append(Mention(
						len(mentions), sentno, tree, cnj,
						a, b, getheadidx(cnj),
						tokens, ngdata, gadata))


def getmentions(trees, ngdata, gadata, precise):
	"""Collect mentions."""
	debug(color('mention detection', 'yellow'))
	mentions = []
	for sentno, (_, tree) in enumerate(trees):
		candidates = []
		candidates.extend(tree.findall('.//node[@cat="np"]'))
		# candidates.extend(tree.findall('.//node[@cat="conj"]'))
		candidates.extend(tree.findall(
				'.//node[@cat="mwu"]/node[@pt="spec"]/..'))
		candidates.extend(tree.findall('.//node[@pt="n"][@ntype="eigen"]'))
		candidates.extend(tree.findall('.//node[@pt="n"][@rel="su"]'))
		candidates.extend(tree.findall('.//node[@pt="n"][@rel="obj1"]'))
		candidates.extend(tree.findall('.//node[@pt="n"][@rel="body"]'))
		candidates.extend(tree.findall('.//node[@pdtype="pron"]'))
		candidates.extend(tree.findall('.//node[@vwtype="bez"]'))
		covered = set()
		for candidate in candidates:
			considermention(candidate, tree, sentno, mentions, covered,
					ngdata, gadata, precise)
	return mentions


def extractmentionsfromconll(conlldata, trees, ngdata, gadata):
	"""Extract gold mentions from annotated data."""
	mentions = []
	for (sentno, begin, end) in sorted(extractgoldmentionspans(conlldata)):
		# smallest node spanning begin, end
		tree = trees[sentno][1]
		node = sorted((node for node in tree.findall('.//node')
					if begin >= int(node.get('begin'))
					and end <= int(node.get('end'))),
				key=lambda x: int(x.get('end')) - int(x.get('begin')))[0]
		tokens = [token.get('word') for token
				in sorted((token for token in tree.findall('.//node[@word]')
					if begin <= int(token.get('begin')) < end),
				key=lambda x: int(x.get('begin')))]
		headidx = getheadidx(node)
		if headidx >= end:
			headidx = max(int(x.get('begin')) for x in node.findall('.//node')
					if int(x.get('begin')) < end)
		mentions.append(Mention(
				len(mentions), sentno, tree, node, begin, end, headidx, tokens,
				ngdata, gadata))
	return mentions


def extractgoldmentionspans(conlldata):
	"""Extract mentions from conll file."""
	gold = set()
	for sentno, chunk in enumerate(conlldata):
		scratch = {}
		for idx, fields in enumerate(chunk):
			labels = fields[-1]
			for a in labels.split('|'):
				if a.startswith('('):
					scratch.setdefault(a.strip('()'), []).append((sentno, idx))
				if a.endswith(')'):
					gold.add(scratch[a.strip('()')].pop() + (idx + 1, ))
		if any(scratch.values()):
			raise ValueError('Unclosed paren? %d %r %s'
					% (sentno, scratch, chunk))
	return gold


def isspeaker(mention):
	"""Test whether mention is subject of a reported speech verb."""
	if (mention.node.get('rel') != 'su'
			or mention.head.get('quotelabel') == 'I'):
		return False
	hd1 = mention.node.find('../node[@cat="ppart"]/node[@rel="hd"]')
	hd2 = mention.node.find('../node[@rel="hd"]')
	hd = hd2 if hd1 is None else hd1
	return hd is not None and hd.get('root') in SPEECHVERBS


def getquotations(trees):
	"""Detect quoted speech spans and speaker / addressee if possible.

	Marks tokens in quoted speech with B, I, O labels.

	- Quoted speech within other quoted speech is not marked.
	- Quoted speech ends at end of paragraph even if no marker is found.
	- Quoted speech can be introduced by ASCII single ' or double quotes "
		or by a dash '-' at the start of a paragraph.
	"""
	# dictionary mapping open quote char to closing quote char
	quotechar = {
			"'": "'",
			'"': '"',
			'`': "'",
			'``': "''",
			}
	# convert to flat list of tokens
	doc = []
	parbreak = []
	idx = {}
	sentnos = []
	parnos = []
	i = 0
	for sentno, ((parno, s), tree) in enumerate(trees):
		for n, token in enumerate(sorted(
				tree.iterfind('.//node[@word]'),
				key=lambda x: int(x.get('begin')))):
			doc.append(token)
			parbreak.append(s == 1 and n == 0)
			idx[sentno, n] = i
			sentnos.append(sentno)
			parnos.append(parno)
			i += 1
	quotations = []
	inquote = None
	start = None
	for i, token in enumerate(doc):
		n = int(token.get('begin'))
		if inquote and parbreak[i]:
			inquote = None
			end = i
			quotations.append(Quotation(start, end,
					sentnos[start], parnos[start],
					' '.join(doc[i].get('word') for i
						in range(start, end)),
					True))
		if inquote is None and parbreak[i] and token.get('word') == '-':
			# detect implied end of quote:
			# - I like cats, he said
			token.set('quotelabel', 'B')
			start = i
			_, tree = trees[sentnos[i]]
			vb = tree.getroot().find('./node/node[@cat="du"]/node[@cat="sv1"]'
					'[@rel="tag"]/node[@rel="hd"]')
			if vb is not None and vb.get('root') in SPEECHVERBS:
				node = tree.getroot().find(
						'./node/node[@cat="du"]/node[@rel="nucl"]')
				end = idx[sentnos[i], int(node.get('end'))]
				quotations.append(Quotation(start, end,
						sentnos[start], parnos[start],
						' '.join(doc[i].get('word') for i
							in range(start, end)),
						False))
				for x in range(start + 1, end):
					doc[x].set('quotelabel', 'I')
			else:
				inquote = '\n'
		elif inquote is None and token.get('word') in quotechar:
			token.set('quotelabel', 'B')
			inquote = quotechar[token.get('word')]
			start = i
		elif token.get('word') == inquote:
			token.set('quotelabel', 'I')
			inquote = None
			end = i + 1
			quotations.append(Quotation(start, end,
					sentnos[start], parnos[start],
					' '.join(doc[i].get('word') for i
						in range(start, end)),
					int(doc[start].get('begin')) == 0
					and (i + 1 == len(doc)
						or int(doc[i + 1].get('begin')) == 0
						or int(doc[i + 2].get('begin')) == 0)))
		elif token.get('quotelabel') is None:
			token.set('quotelabel', 'I' if inquote else 'O')
	return quotations, idx, doc


def speakeridentification(mentions, clusters, quotations, idx, doc):
	debug(color(
			'speaker identification (%d quotations)' % len(quotations),
			'yellow'))
	if not quotations:
		return
	tokenidx2quotation = {}
	for quotation in quotations:
		for i in range(quotation.start, quotation.end):
			tokenidx2quotation[i] = quotation
	qstarts = [q.start for q in quotations]
	qends = [q.end for q in quotations]
	# FIXME: never assign same speaker & addressee

	# for each subject of a reported speech verb, link it to closest quotation
	for mention in mentions:
		if isspeaker(mention):
			i = idx[mention.sentno, mention.begin]
			i1 = idx[mention.sentno, 0]
			i2 = idx.get((mention.sentno + 1, 0), len(doc))
			# first quote to left of mention
			q1 = quotations[bisect(qends, i) - 1]
			if i1 < q1.end <= i2 and i - q1.end <= 5 and q1.speaker is None:
				q1.speaker = mention
				debug('attributed %s\n\tto mention directly after: %s' % (
						color(q1.text, 'green'), q1.speaker))
			else:
				# first quote to right of mention
				x = bisect(qstarts, i)
				q2 = quotations[x if x < len(quotations) else x - 1]
				if (i1 <= q2.start < i2 and q2.start - i <= 5
						and q2.speaker is None):
					q2.speaker = mention
					debug('attributed %s\n\tto mention directly before: %s' % (
							color(q2.text, 'green'), q2.speaker))
	for prev, quotation in zip(quotations, quotations[1:]):
		# assume speaker is unchanged for consecutive quotations
		# in same paragraph when there is material outside quotes.
		# "I don't know", he said. "It seems a bad idea."
		if (quotation.speaker is None and prev.speaker is not None
				and quotation.parno == prev.parno
				and quotation.sentno <= prev.sentno + 1
				and not prev.sentbounds):
			quotation.speaker = prev.speaker
			quotation.addressee = prev.addressee
			assert quotation.speaker is not quotation.addressee
			debug('attributed %s\n\tto previous speaker %s' % (
					color(quotation.text, 'green'), prev.speaker))
		# consecutive quotations in different paragraphs are turn taking;
		# or when first quotation has no material outside quotation marks
		# "How are you?" "I'm fine."
		elif (quotation.speaker is None and prev.speaker is not None
				and (quotation.parno == prev.parno + 1
					or prev.sentbounds)):
			quotation.speaker = prev.addressee
			quotation.addressee = prev.speaker
			assert quotation.speaker is not quotation.addressee
			debug('attributed %s\n\tto previous addressee %s' % (
					color(quotation.text, 'green'), prev.addressee))
		# assume distinct consecutive speakers are addressing each other.
		# coreference has not yet been established, use string match
		elif (quotation.speaker is not None and prev.speaker is not None
				and quotation.start - prev.end < 10
				and quotation.speaker.tokens != prev.speaker.tokens):
			quotation.addressee = prev.speaker
			prev.addressee = quotation.speaker
			assert quotation.speaker is not quotation.addressee
			debug('%s %s\n\taddresses %s %s' % (
					prev.speaker, color(prev.text, 'green'),
					quotation.speaker, color(quotation.text, 'green')))
	# For unattributed quotes without any material before or after the quote,
	# assign closest human mention between quote and previous quote.
	# When there is material before or after the quote, it may not be a
	# dialogue turn, e.g.
	# 'Every happy family is alike,' according to Tolstoy's Anna Karenina.
	# Only consider human mentions that are not possessive pronouns.
	# mstarts = [idx[mention.sentno, mention.begin] for mention in mentions
	# 		if mention.features['human'] == 1
	# 		and mention.head.get('vwtype') != 'bez']
	mends = [idx[mention.sentno, mention.end - 1] for mention in mentions
			if mention.features['human'] == 1
			and mention.head.get('vwtype') != 'bez']
	for prev, quotation in zip([None] + quotations, quotations):
		if quotation.speaker is not None or not quotation.sentbounds:
			continue
		# first mention to left of quote
		i = bisect(mends, quotation.start) - 1
		m1 = mentions[i] if 0 <= i < len(mentions) else None
		if (m1 is not None
				and m1.sentno + 1 == quotation.sentno
				and (prev is None or prev.end <= idx[m1.sentno, m1.begin])):
			quotation.speaker = m1
			debug('attributed %s\n\tto closest mention before quote: %s' % (
					color(quotation.text, 'green'), quotation.speaker))
			assert quotation.speaker is not quotation.addressee
		else:
			# first mention to right of quote
			# m2 = mentions[bisect(mstarts, quotation.end)]
			pass
	# collect mentions within each quote
	for mention in mentions:
		if idx[mention.sentno, mention.begin] in tokenidx2quotation:
			i = idx[mention.sentno, mention.begin]
			tokenidx2quotation[i].mentions.append(mention)
	# add speaker constraints
	for prev, quotation in zip([None] + quotations, quotations):
		if quotation.speaker is None:
			debug('no speaker: %s' % color(quotation.text, 'green'))
		for i in range(quotation.start, quotation.end):
			if quotation.speaker is not None:
				doc[i].set('speaker', str(idx[quotation.speaker.sentno,
						quotation.speaker.begin] + 1))
			if quotation.addressee is not None:
				doc[i].set('addressee', str(idx[quotation.addressee.sentno,
						quotation.addressee.begin] + 1))
		nominalmentions = [mention for mention in quotation.mentions
				if mention.type != 'pronoun']
		for mention in quotation.mentions:
			if (mention.type == 'pronoun'
					and mention.features['person'] in ('1', '2')):
				# Nominal mentions cannot be coreferent with
				# I, you, or we in the same turn or quotation.
				mention.prohibit.update(
						mention.id for mention in nominalmentions)
			if (quotation.speaker is not None
					and (mention.type != 'pronoun'
					or mention.features['person'] != '1'
					or mention.features['number'] != 'sg')):
				# speaker and not-I-mentions in quote cannot be coreferent
				mention.prohibit.add(quotation.speaker.id)
			if (quotation.addressee is not None
					and (mention.type != 'pronoun'
					or mention.features['person'] != '2'
					or mention.features['number'] != 'sg')):
				# addressee and not-you-mentions in quote cannot be coreferent
				mention.prohibit.add(quotation.addressee.id)
		for person in ('1', '2'):
			for number in ('sg', 'pl'):
				pronouns = [a for a in quotation.mentions
						if a.features['person'] == person
						and a.features['number'] == number]
				# two pronouns with different person/number cannot corefer
				for a in pronouns:
					for b in quotation.mentions:
						if (b.type == 'pronoun'
								and (b.features['person'] != person
									or b.features['number'] != number)):
							a.prohibit.add(b.id)
							b.prohibit.add(a.id)
				# 1/2nd person mentions from different speaker cannot corefer
				# block selectively for adjacent quotes by different speakers
				if (prev is not None and number == 'sg'
						and prev.speaker is not quotation.speaker):
					for a in pronouns:
						for b in prev.mentions:
							if b.features['person'] == person:
								a.prohibit.add(b.id)
								b.prohibit.add(a.id)
	# breakdown of number of quotations per speaker
	counts = defaultdict(int)
	for quotation in quotations:
		if quotation.speaker is None:
			counts[-1] += 1
		else:
			counts[quotation.speaker.clusterid] += 1
	# for a, b in sorted(counts.items(), key=lambda x: -x[1]):
	# 	debug('%d %s' % (b, 'Unknown' if a == -1
	# 			else mentions[min(clusters[a])]))


def stringmatch(mentions, clusters, relaxed=False):
	"""Link mentions with matching strings;
	if relaxed, ignore modifiers/appositives."""
	debug(color('string match (relaxed=%s)' % relaxed, 'yellow'))
	foundentities = {}
	for _, mention in mentionselection(mentions, clusters):
		if mention.type != 'pronoun':
			if (len(clusters[mention.clusterid]) == 1
					and mention.node.get('ntype') == 'soort'
					and mention.features['number'] == 'pl'):
				continue
			mstr = ' '.join(mention.relaxedtokens
					if relaxed else mention.tokens).lower()
			if mstr in foundentities:
				merge(foundentities[mstr], mention, mentions, clusters)
			else:
				foundentities[mstr] = mention


def preciseconstructs(mentions, clusters):
	"""Link syntactically related mentions:
	appositives, predicatives, relative pronouns."""
	debug(color('precise constructs', 'yellow'))
	appositives = {}
	predicatives = {}
	relpronouns = {}
	reflpronouns = {}
	recippronouns = {}

	# Pass 1: collect antecedents
	for mention in mentions:
		if (mention.node.get('rel') != 'app'
				and len(mention.node.getparent()) > 1
				and mention.node.getparent()[1].get('rel') == 'app'):
			node = mention.node.getparent()[1]
			appositives[mention.sentno,
					int(node.get('begin')), int(node.get('end'))
					] = mention
		if (mention.node.get('rel') == 'su'
				and mention.node.find('../node[@rel="predc"]') is not None):
			node = mention.node.find('../node[@rel="predc"]')
			predicatives[mention.sentno,
					node.get('begin'), node.get('end')] = mention
		if mention.node.find(
				"./node[@cat='rel']/node[@vwtype='betr']") is not None:
			node = mention.node.find(
					"./node[@cat='rel']/node[@vwtype='betr']")
			relpronouns[mention.sentno,
					int(node.get('begin')), int(node.get('end'))
					] = mention
		if mention.node.get('vwtype') != 'refl':
			# Check if this mention is an antecedent to a reflexive pronoun.
			# Assume reflexive pronoun should be linked to closest
			# candidate antecedent.
			# det applies in genitive case: John in John's car.
			node = None
			if mention.node.getparent().get('rel') == 'su':
				node = mention.node.find('../..//node[@vwtype="refl"]')
			elif mention.node.get('rel') in ('su', 'det'):
				node = mention.node.find('..//node[@vwtype="refl"]')
			if node is not None:
				reflpronouns[mention.sentno,
					int(node.get('begin')), int(node.get('end'))
					] = mention
		if (mention.node.get('vwtype') != 'recip'
				and mention.head.get('num') == 'pl'):
			node = mention.node.find('..//node[@vwtype="recip"]')
			if node is not None:
				recippronouns[mention.sentno,
					int(node.get('begin')), int(node.get('end'))
					] = mention

	# Pass 2: find mentions to link to collected antecedents
	for mention in mentions:
		if (mention.node.get('rel') == 'app'
				and (mention.sentno, mention.begin, mention.end)
					in appositives):
			merge(appositives[mention.sentno, mention.begin, mention.end],
					mention, mentions, clusters)
		if (mention.node.get('rel') == 'predc'
				and (mention.sentno,
					mention.node.get('begin'),
					mention.node.get('end'))
				in predicatives):
			merge(predicatives[mention.sentno,
					mention.node.get('begin'),
					mention.node.get('end')],
					mention, mentions, clusters)
		if (mention.node.get('vwtype') == 'betr'
				and (mention.sentno, mention.begin, mention.end)
				in relpronouns):
			merge(relpronouns[mention.sentno, mention.begin, mention.end],
					mention, mentions, clusters)
		if (mention.node.get('vwtype') == 'refl'
				and (mention.sentno, mention.begin, mention.end)
				in reflpronouns):
			merge(reflpronouns[mention.sentno, mention.begin, mention.end],
					mention, mentions, clusters)
		if (mention.node.get('vwtype') == 'recip'
				and (mention.sentno, mention.begin, mention.end)
				in recippronouns):
			merge(recippronouns[mention.sentno, mention.begin, mention.end],
					mention, mentions, clusters)


def strictheadmatch(mentions, clusters, sieve):
	"""Link mentions with matching heads and modifiers.

	sieve is 5, 6, or 7, to determine strictness:
		:5: both 6 and 7 apply
		:6: the non-stop words of a mention are a subset of the set of words in
			the mentions of another cluster
		:7: all modifiers of a mention are included in another mention
		"""
	debug(color('strict head match %d' % sieve, 'yellow'))
	heads = [set() if cluster is None
			else {mentions[m].head.get('word') for m in cluster}
			for cluster in clusters]
	for n, mention in mentionselection(mentions, clusters):
		if mention.type != 'pronoun' and checkconstraints(mention, clusters):
			nonstop = {a for a in mention.tokens if a not in STOPWORDS}
			head = mention.head.get('word')
			for othercluster, otherheads in zip(clusters[:n], heads):
				# entity head match
				match = othercluster is not None and head in otherheads
				if match and (sieve == 5 or sieve == 6):
					other = mentions[min(othercluster)]
					if other.type == 'pronoun':
						continue
					# word inclusion
					othernonstop = {token
							for m in othercluster
								for token in mentions[m].tokens}
					match = nonstop and nonstop.issubset(othernonstop)
				if match and (sieve == 5 or sieve == 7):
					for m in othercluster:
						other = mentions[m]
						if other.type == 'pronoun':
							continue
						# compatible modifiers only
						if all(token in other.tokens
								for token in mention.mainmod):
							break
					else:
						match = False
				if match and iwithini(mention, other):
					match = False
				if match:
					merge(other, mention, mentions, clusters)
					heads[mention.clusterid] = heads[
							mention.clusterid] | otherheads


def properheadmatch(mentions, clusters, relaxed=False):
	"""Link mentions with same proper noun head."""
	debug(color('proper head match (relaxed=%s)' % relaxed, 'yellow'))
	othernonstop = {clusterid:
			{token for m in cluster
				for token in mentions[m].tokens}
			for clusterid, cluster in enumerate(clusters)
			if cluster is not None}
	for _, mention in mentionselection(mentions, clusters):
		if (mention.head.get('neclass') not in (None, 'LOC')
				and checkconstraints(mention, clusters)):
			nonstop = {a for a in mention.tokens if a not in STOPWORDS}
			# NB: also looks forward!
			# [John, John Smith, Mr Smith] will form a cluster
			for other in mentions:
				if (other.head.get('neclass') != mention.head.get('neclass')
						or mention.clusterid == other.clusterid
						or iwithini(mention, other)):
					continue
				if relaxed:
					# word inclusion
					if (mention.head.get('word') in other.tokens
							and nonstop and nonstop.issubset(
								othernonstop[other.clusterid])):
						if other.sentno < mention.sentno:
							merge(other, mention, mentions, clusters)
						else:
							merge(mention, other, mentions, clusters)
				elif mention.head.get('lemma') == other.head.get('lemma'):
					if other.sentno < mention.sentno:
						merge(other, mention, mentions, clusters)
					else:
						merge(mention, other, mentions, clusters)


def resolvepronouns(mentions, clusters, quotations):
	"""Find antecedents of unresolved pronouns with compatible features."""
	debug(color('pronoun resolution', 'yellow'))
	# Link all 1st/2nd person pronouns not in quotes
	for person in ('1', '2'):
		for number in ('sg', 'pl'):
			pronouns = [a for a in mentions
					if a.features['person'] == person
					and a.features['number'] == number
					and a.head.get('quotelabel') == 'O']
			for a in pronouns[1:]:
				merge(pronouns[0], a, mentions, clusters)
	# sortedmentions = sorted(mentions, key=lambda x: (x.sentno, x.begin))
	# prefer recent subjects, objects over other mentions.
	sortedmentions = sorted(mentions,
			key=lambda x: (x.sentno,
				x.node.get('rel') == 'su',
				x.node.get('rel') == 'obj1' and x.node.getparent().get(
					'cat') != 'pp',
				# x.node.get('rel') == 'obj2',
				x.begin))
	sortedmentionssentno = [mention.sentno for mention in sortedmentions]
	for _, mention in mentionselection(mentions, clusters):
		if (mention.type == 'pronoun'
				and len(clusters[mention.clusterid]) == 1
				and mention.features['person'] not in ('1', '2')):
			debug(mention.sentno, mention.begin, mention)
			i = bisect(sortedmentionssentno, mention.sentno)
			assert sortedmentionssentno[i - 1] == mention.sentno
			# consider identical pronouns first (check with string match)
			# for other in sorted(reversed(sortedmentions[:i]),
			# 		key=lambda x: len(x.tokens) == len(mention.tokens) == 1
			# 			and x.tokens[0].lower() != mention.tokens[0].lower()):
			for other in reversed(sortedmentions[:i]):
				if other.sentno < mention.sentno - 10:
					break
				# The antecedent should come before anaphor,
				# and should not contain anaphor.
				if (other.sentno == mention.sentno
						and (other.begin >= mention.begin
							or other.end >= mention.end)):
					debug('\t%d %d %s %d %s prohibited=%d i-within-i or >' % (
							other.sentno, other.begin, other.node.get('rel'),
							len(clusters[other.clusterid]),
							other,
							int(prohibited(mention, other, clusters))))
					continue
				# An anaphor (mention) cannot be a coargument of its
				# antecedent (other). Coarguments are in the same clause
				# but do not necessarily have the same parent.
				if (sameclause(other.node, mention.node)
						and other.node.find('..//node[@id="%s"]'
						% mention.node.get('id')) is not None):
					debug('\t%d %d %s %d %s prohibited=1 coargument' % (
							other.sentno, other.begin, other.node.get('rel'),
							len(clusters[other.clusterid]),
							other))
					continue
				debug('\t%d %d %s %d %s prohibited=%d' % (
						other.sentno, other.begin, other.node.get('rel'),
						len(clusters[other.clusterid]),
						other,
						int(prohibited(mention, other, clusters))))
				if (compatible(mention, other)
						and not prohibited(mention, other, clusters)):
					merge(other, mention, mentions, clusters)
					break
	debug(color('pronouns in quotations', 'yellow'))
	for quotation in quotations:
		for person in ('1', '2'):
			for number in ('sg', 'pl'):
				pronouns = [a for a in quotation.mentions
						if a.features['person'] == person
						and a.features['number'] == number]
				# pronouns with same person/number in quote corefer
				for a in pronouns[1:]:
					merge(pronouns[0], a, mentions, clusters)
				# I in quote is speaker
				if (pronouns and person == '1' and number == 'sg'
						and quotation.speaker is not None):
					merge(quotation.speaker, pronouns[0], mentions, clusters)
				# you in quote is addressee
				elif (pronouns and person == '2' and number == 'sg'
						and quotation.addressee is not None):
					merge(quotation.addressee, pronouns[0], mentions, clusters)


def resolvecoreference(trees, ngdata, gadata, precise=True, mentions=None):
	"""Get mentions and apply coreference sieves."""
	if mentions is None:
		mentions = getmentions(trees, ngdata, gadata, precise)
	clusters = [{n} for n, _ in enumerate(mentions)]
	quotations, idx, doc = getquotations(trees)
	if VERBOSE:
		for mention in mentions:
			debug(mention,
					'head=%s neclass=%s' % (
						mention.head.get('word'),
						mention.head.get('neclass')),
					# ngdata.get(' '.join(mention.tokens).lower()),
					# ngdata.get(mention.tokens[0].lower()),
					# gadata.get(mention.head.get('lemma', '').replace('_', ''))
					)
	speakeridentification(mentions, clusters, quotations, idx, doc)
	stringmatch(mentions, clusters)
	stringmatch(mentions, clusters, relaxed=True)
	preciseconstructs(mentions, clusters)
	strictheadmatch(mentions, clusters, 5)
	strictheadmatch(mentions, clusters, 6)
	strictheadmatch(mentions, clusters, 7)
	properheadmatch(mentions, clusters)
	properheadmatch(mentions, clusters, relaxed=True)
	resolvepronouns(mentions, clusters, quotations)
	return mentions, clusters, quotations


def sameclause(node1, node2):
	"""Return true if nodes are arguments in the same clause."""
	clausecats = ('smain', 'ssub', 'sv1', 'inf')
	index = node1.get('index')
	while (node1 is not None and node1.get('cat') not in clausecats):
		node1 = node1.getparent()
	while node2 is not None and node2.get('cat') not in clausecats:
		node2 = node2.getparent()
	# if there is a coindexed node referring to node1,
	# node1 and node2 are coarguments.
	if node2 is None:
		return False
	elif index and node2.find('./node[@index="%s"]' % index) is not None:
		return True
	return node1 is node2


def writetabular(trees, mentions,
		docname='-', part='-', file=sys.stdout, fmt=None, startcluster=0):
	"""Write output in tabular format."""
	sentences = [sorted(tree.iterfind('.//node[@word]'),
				key=lambda x: int(x.get('begin')))
			for _, tree in trees]
	sentids = ['%d-%d' % (parno, sentno) for (parno, sentno), _ in trees]
	labels = [[''] * len(sent) for sent in sentences]
	for mention in sortmentions(mentions):
		for n in range(mention.begin, mention.end):
			coreflabel = labels[mention.sentno][n]
			# Start of mention, add bracket
			if n == mention.begin:
				if coreflabel:
					coreflabel += '|'
				coreflabel += '('
			if coreflabel:
				if coreflabel[-1] != '(':
					coreflabel += '|'
			coreflabel += str(mention.clusterid + startcluster)
			# End of mention, add bracket
			if n + 1 == mention.end:
				coreflabel += ')'
			labels[mention.sentno][n] = coreflabel
	labels = [[a or '-' for a in coreflabels]
			for n, coreflabels in enumerate(labels, 1)]
	doctokenid = 0
	if fmt == 'semeval2010':
		print('#begin document %s' % docname, file=file)
	else:
		print('#begin document (%s);' % docname, file=file)
	for sentid, sent, sentlabels in zip(sentids, sentences, labels):
		if fmt == 'conll2012':
			print('# sent_id = %s' % sentid)
		for tokenid, (token, label) in enumerate(zip(sent, sentlabels), 1):
			doctokenid += 1
			if fmt is None or fmt == 'minimal':
				print(docname, doctokenid, token.get('word'), label,
						sep='\t', file=file)
			elif fmt == 'conll2012':
				print(docname, part, doctokenid, token.get('word'),
						token.get('postag'), *(['-'] * 5), label,
						sep='\t', file=file)
			elif fmt == 'semeval2010':
				print(tokenid, token.get('word'), label,
						sep='\t', file=file)
			elif fmt == 'booknlp':
				print(
						doctokenid,
						sentid,
						tokenid,
						token.get('word'),
						token.get('lemma'),
						token.get('postag'),
						token.get('UDparent', '-'),  # tokenid
						token.get('UDlabel', '-'),
						token.get('neclass', '-'),  # PER, ORG, LOC, ...
						token.get('speaker', '-'),  # doctokenid
						token.get('addressee', '-'),  # doctokenid
						token.get('quotelabel', '-'),  # B, I, O
						label,
						sep='\t', file=file)
		print('', file=file)
	if fmt == 'semeval2010':
		print('#end document %s' % docname, file=file)
	else:
		print('#end document', file=file)


def writehtml(trees, mentions, clusters, quotations,
		docname='-', file=sys.stdout):
	"""Visualize coreference in HTML document."""
	print(HTMLTEMPLATE % docname, file=file)
	sentences = [[a.get('word') for a
			in sorted(tree.iterfind('.//node[@word]'),
				key=lambda x: int(x.get('begin')))]
			for _, tree in trees]
	sentids = [(parno, sentno) for (parno, sentno), _ in trees]
	for mention in sortmentions(mentions):
		for n in range(mention.begin, mention.end):
			coreflabel = sentences[mention.sentno][n]
			# Start of mention, add bracket
			if n == mention.begin:
				if len(clusters[mention.clusterid]) > 1:
					coreflabel = '<span id="m%d" class="c%d">[%s' % (
							mention.id, mention.clusterid, coreflabel)
				else:
					coreflabel = '[' + coreflabel
			# End of mention, add bracket
			if n + 1 == mention.end:
				if len(clusters[mention.clusterid]) > 1:
					coreflabel = coreflabel + ']</span>'
				else:
					coreflabel = coreflabel + ']'
			sentences[mention.sentno][n] = coreflabel
	qstarts = {q.start: n for n, q in enumerate(quotations)}
	qends = {q.end - 1 for q in quotations}
	doctokenid = 0
	for ((parno, sentno), sent) in zip(sentids, sentences):
		if parno == 1 and sentno == 1:
			print('<p>', end='', file=file)
		elif sentno == 1:
			print('</p>\n<p>', end='', file=file)
		for token in sent:
			if doctokenid in qstarts:
				quotation = quotations[qstarts[doctokenid]]
				over = out = att = ''
				if quotation.speaker is not None:
					over += "hl1('m%d'); " % quotation.speaker.id
					out += "nohl('m%d'); " % quotation.speaker.id
				if quotation.addressee is not None:
					over += "hl2('m%d'); " % quotation.addressee.id
					out += "nohl('m%d'); " % quotation.addressee.id
				if over:
					att = ' onmouseover="%s" onmouseout="%s"' % (over, out)
				print('<span class=q%s>' % att, end='', file=file)
			print(' ' + token, end='', file=file)
			if doctokenid in qends:
				print('</span>', end='', file=file)
			doctokenid += 1
	print('\n</p></body></html>', file=file)


def readconll(conllfile, docname='-'):
	"""read conll data as list of lists: conlldata[sentno][tokenno]."""
	conlldata = [[]]
	with open(conllfile) as inp:
		while True:
			line = inp.readline()
			if (line.startswith('#begin document') and (docname == '-'
					or line.split()[2].strip('();') == docname)):
				while True:
					line = inp.readline()
					if line.startswith('#end document') or line == '':
						break
					if line.startswith('#'):
						pass
					elif line.strip():
						conlldata[-1].append(line.strip().split('\t'))
					else:
						conlldata.append([])
				break
			elif line == '':
				break
	if not conlldata[0]:
		raise ValueError('Could not read gold data from %r with docname %r' % (
				conllfile, docname))
	return conlldata


def comparementions(conlldata, trees, mentions, out=sys.stdout):
	"""Human-readable printing of a comparison between the output of the
	mention detection sieve and the 'gold' standard. Green brackets are
	correct, yellow brackets are mention boundaries only found in the gold
	standard, and red brackets are only found in our output."""
	sentences = [[a.get('word') for a in
			sorted(tree.iterfind('.//node[@word]'),
				key=lambda x: int(x.get('begin')))]
			for _, tree in trees]
	resp = {(mention.sentno, mention.begin, mention.end)
			for mention in mentions}
	gold = extractgoldmentionspans(conlldata)
	print('\nmentions in gold missing from response:')
	for sentno, begin, end in gold - resp:
		print(' '.join(sentences[sentno][begin:end]))
	print('\nmentions in response but not in gold:')
	for sentno, begin, end in resp - gold:
		print(' '.join(sentences[sentno][begin:end]))
	print()
	#
	mentionbegin = defaultdict(int)
	mentionend = defaultdict(int)
	for mention in mentions:
		mentionbegin[mention.sentno, mention.begin] += 1
		mentionend[mention.sentno, mention.end - 1] += 1
	for sentno, ((_, tree), sent) in enumerate(zip(trees, sentences)):
		out.write('%d: ' % sentno)
		for idx, token in enumerate(sent):
			goldopen = goldclose = respopen = respclose = 0
			respopen += mentionbegin.get((sentno, idx), 0)
			respclose += mentionend.get((sentno, idx), 0)
			goldopen = conlldata[sentno][idx][-1].count('(')
			goldclose = conlldata[sentno][idx][-1].count(')')
			if goldopen >= respopen:
				out.write((goldopen - respopen) * color('[', 'yellow'))
				out.write(respopen * color('[', 'green'))
			else:
				out.write((respopen - goldopen) * color('[', 'red'))
				out.write(goldopen * color('[', 'green'))
			out.write(token)
			if goldclose >= respclose:
				out.write((goldclose - respclose) * color(']', 'yellow'))
				out.write(respclose * color(']', 'green') + ' ')
			else:
				out.write((respclose - goldclose) * color(']', 'red'))
				out.write(goldclose * color(']', 'green') + ' ')
		out.write('\n')
	return resp, gold


def comparecoref(conlldata, trees, mentions, clusters, resp, gold,
		out=sys.stdout, docname='-'):
	"""List correct/incorrect coreference links."""
	def getcoref(mention):
		# look up span of mention in conll file
		# return coref chains X with "(X" at begin idx and "X)" at end idx
		return {int(a.strip('()')) for a in
				conlldata[mention.sentno][mention.begin][-1].split('|')
				if (mention.begin + 1 == mention.end
					and a.startswith('(') and a.endswith(')'))
				or (a.startswith('(') and a[1:] + ')'
					in conlldata[mention.sentno][mention.end - 1][-1].split(
						'|'))}

	def correctlink(mention1, mention2):
		a = getcoref(mention1)
		b = getcoref(mention2)
		return a and b and not a.isdisjoint(b)

	# take the first mention of cluster that is also a mention in gold
	for cluster in clusters:
		if cluster is None or len(cluster) == 1:
			continue
		cand = sorted(n for n in cluster if
				(mentions[n].sentno, mentions[n].begin, mentions[n].end)
				in gold)
		n = cand[0] if cand else min(cluster)
		print(mentions[n].sentno, mentions[n].begin,
				' '.join(mentions[n].tokens))
		for m in sorted(cluster - {n}):
			correct = correctlink(mentions[n], mentions[m])
			print('\t',
					color('<-->', 'green' if correct else 'red'),
					mentions[m].sentno, mentions[m].begin,
					' '.join(mentions[m].tokens))
		# FIXME: look up missed gold links and print as 'yellow'
		# (a) gold links for correctly identified mentions
		# (b) gold links for missed mentions


def process(path, output, ngdata, gadata,
		docname='-', conllfile=None, fmt=None,
		start=None, end=None, startcluster=0,
		precise=True, goldmentions=False):
	"""Process a single directory with Alpino XML parses."""
	if not path.endswith('*.xml'):
		path = os.path.join(path, '*.xml')
	debug(conllfile or path)
	filenames = sorted(glob(path), key=parsesentid)[start:end]
	trees = [(parsesentid(filename), etree.parse(filename))
			for filename in filenames]
	if conllfile is not None:
		conlldata = readconll(conllfile, docname)[start:end]
	mentions = None
	if goldmentions:
		mentions = extractmentionsfromconll(conlldata, trees, ngdata, gadata)
	mentions, clusters, quotations = resolvecoreference(trees, ngdata, gadata,
			precise, mentions)
	if fmt == 'booknlp':
		getUD(filenames, trees)
	if fmt == 'html':
		writehtml(trees, mentions, clusters, quotations,
				docname=docname, file=output)
	else:
		writetabular(trees, mentions, docname=docname,
				file=output, fmt=fmt, startcluster=startcluster)
	if conllfile is not None and VERBOSE:
		resp, gold = comparementions(conlldata, trees, mentions)
		comparecoref(conlldata, trees, mentions, clusters, resp, gold,
				docname=docname)
	return len(clusters)


def getUD(filenames, trees):
	"""Convert Alpino trees to UD trees and store head/label in attributes."""
	with tempfile.NamedTemporaryFile(mode='w') as out:
		out.write('<collection>')
		out.writelines('<doc href="%s"/>\n' % filename
				for filename in filenames)
		out.write('</collection>')
		out.flush()
		conll = subprocess.check_output(
				('xqilla -v ENHANCED yes -v DIR %s -v MODE conll '
				'universal_dependencies_2.2.xq' % out.name).split())
	conll = re.sub(r'<pre><code.*?</sentence>\n|[ \t]+!\n\s+</code></pre>',
			'', conll.decode('utf8'))
	for (_, tree), chunk in zip(trees, conll.split('\n\n')):
		tokens = sorted(tree.iterfind('.//node[@word]'),
					key=lambda x: int(x.get('begin')))
		chunk = [line.split('\t') for line in chunk.splitlines()]
		if len(tokens) != len(chunk):
			raise ValueError('sentence length mismatch.')
		for token, line in zip(tokens, chunk):
			token.set('UDparent', line[6] if len(line) > 7 else '-')
			token.set('UDlabel', line[7] if len(line) > 7 else '-')


def clindev(ngdata, gadata, goldmentions):
	"""Run on CLIN26 shared task dev data and evaluate."""
	timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
	os.system('mkdir -p results/' + timestamp)
	for path in glob('../groref/clinDevData/*.coref_ne'):
		dirname = os.path.join(
				os.path.dirname(path),
				os.path.basename(path).split('_')[0])
		docname = os.path.basename(path)
		outfilename = 'results/%s/%s' % (timestamp, docname)
		with open(outfilename, 'w') as out:
			process(dirname + '/*.xml', out, ngdata, gadata,
					docname=docname, conllfile=path, goldmentions=goldmentions,
					start=0, end=6)
			# shared task says the first 7 sentences are annotated,
			# but in many documents only the first 6 sentences are annotated.

	with open('results/%s/blanc_scores' % timestamp, 'w') as out:
		os.chdir('../groref/clin26-eval-master')
		subprocess.call(
				['bash', 'score_coref.sh',
					'coref_ne', 'dev_corpora/coref_ne',
					'../../dutchcoref/results/' + timestamp, 'blanc'],
				stdout=out)
	os.chdir('../../dutchcoref')
	print(open('results/%s/blanc_scores' % timestamp).read())


def semeval(ngdata, gadata, goldmentions):
	"""Run on semeval 2010 shared task dev data and evaluate."""
	timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
	os.system('mkdir -p resultsemeval/' + timestamp)
	outfilename = 'resultsemeval/%s/result.conll' % timestamp
	startcluster = 0
	with open(outfilename, 'w') as out:
		for dirname in sorted(glob('data/semeval2010NLdevparses/*/'),
				key=lambda x: int(x.rstrip('/').split('/')[-1].split('_')[1])):
			docname = os.path.basename(dirname.rstrip('/'))
			startcluster += process(dirname + '/*.xml', out, ngdata, gadata,
					fmt='semeval2010', docname=docname,
					conllfile='data/semeval2010/task01.posttask.v1.0/'
						'corpora/training/nl.devel.txt.fixed',
					startcluster=startcluster, precise=False,
					goldmentions=goldmentions)
	with open('resultsemeval/%s/blanc_scores' % timestamp, 'w') as out:
		subprocess.call([
				'../groref/conll_scorer/scorer.pl',
				'blanc',
				'../../data/semeval2010/task01.posttask.v1.0/'
					'corpora/training/nl.devel.txt.fixed',
				'resultsemeval/%s/result.conll'
				% timestamp],
				stdout=out)
	print(open('resultsemeval/%s/blanc_scores' % timestamp).read())


def runtests(ngdata, gadata):
	print('ref')
	trees = [(parsesentid(filename), etree.parse(filename))
			for filename in sorted(glob('tests/ref/*.xml'), key=parsesentid)]
	for n in range(len(trees)):
		mentions, clusters, _quotations = resolvecoreference(
				trees[n:n + 1], ngdata, gadata)
		print(n, ' '.join(a.get('word') for a in sorted(
				(a for a in trees[n][1].iterfind('.//node[@word]')),
				key=lambda x: int(x.get('begin')))))
		for n, mention in enumerate(mentions):
			print(n, mention)
		print(clusters)
		if not any(len(a) > 1 for a in clusters if a is not None):
			raise ValueError

	print('\nnonref')
	trees = [(parsesentid(filename), etree.parse(filename))
			for filename in sorted(glob('tests/nonref/*.xml'), key=parsesentid)]
	for n in range(len(trees)):
		mentions, clusters, quotations = resolvecoreference(
				trees[n:n + 1], ngdata, gadata)
		print(n, ' '.join(a.get('word') for a in sorted(
				(a for a in trees[n][1].iterfind('.//node[@word]')),
				key=lambda x: int(x.get('begin')))))
		for m, mention in enumerate(mentions):
			print(m, mention)
		print(clusters)
		if not all(len(a) == 1 for a in clusters if a is not None):
			raise ValueError

	print('\ntests passed')


def readngdata():
	"""Read noun phrase number-gender counts."""
	ngdata = {}  # Format: {NP: [masc, fem, neuter, plural]}
	with open('../groref/ngdata', encoding='utf8') as inp:
		for line in inp:
			n = line.index('\t')
			ngdata[line[:n]] = line[n + 1:-1]
	gadata = {}  # Format: {noun: (gender, animacy)}
	with open('data/gadata', encoding='utf8') as inp:
		for line in inp:
			a, b, c = line.rstrip('\n').split('\t')
			gadata[a] = b, c
	# https://www.meertens.knaw.nl/nvb/
	with open('data/Top_eerste_voornamen_NL_2010.csv', encoding='latin1') as inp:
		for line in islice(inp, 2, None):
			fields = line.split(';')
			if fields[1]:
				gadata[fields[1]] = ('f', 'human')
			if fields[3]:
				gadata[fields[3]] = ('m', 'human')
	return ngdata, gadata


def main():
	"""CLI"""
	global VERBOSE
	opts, args = gnu_getopt(sys.argv[1:], '', [
		'help', 'verbose', 'clindev', 'semeval', 'test',
		'goldmentions', 'fmt=', 'slice='])
	opts = dict(opts)
	if '--help' in opts:
		print(__doc__)
		return
	if '--verbose' in opts:
		VERBOSE = True
		sys.argv.remove('--verbose')
	ngdata, gadata = readngdata()
	if '--clindev' in opts:
		clindev(ngdata, gadata, '--goldmentions' in opts)
	elif '--semeval' in opts:
		semeval(ngdata, gadata, '--goldmentions' in opts)
	elif '--test' in opts:
		runtests(ngdata, gadata)
	else:
		start, end = opts.get('--slice', ':').split(':')
		start = int(start) if start else None
		end = int(end) if end else None
		path = args[0]
		if not path.endswith('/'):
			path += '/'
		process(path, sys.stdout, ngdata, gadata,
				fmt=opts.get('--fmt'), start=start, end=end,
				docname=os.path.basename(os.path.dirname(path)))


if __name__ == '__main__':
	main()
