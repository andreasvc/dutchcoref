"""Dutch coreference resolution & dialogue analysis using deterministic rules.

Usage: python3 coref.py [options] <directory>
where directory contains .xml files with sentences parsed by Alpino.
Filenames and sentence IDs are expected to be of the form `n.xml` or `m-n.xml`,
where `n` is a sentence number and `m` a paragraph number.
Output is sent to STDOUT unless --outputprefix is used.

Options:
    --help          this message
    --slice=N:M     restrict input with a Python slice of sentence numbers
    --verbose       debug output instead of coreference output
    --gold=<file>   with --verbose, show error analysis against CoNLL file
    --goldmentions  instead of predicting mentions, use mentions in --gold file
    --fmt=<minimal|semeval2010|conll2012|booknlp|html>
        output format:
            :minimal: doc ID, token ID, token, and coreference columns
            :booknlp: tabular format with universal dependencies and dialogue
                information in addition to coreference.
            :html: interactive HTML visualization with coreference and dialogue
                information.
    --outputprefix=<prefix>
        write conll/mention/cluster/link info to files
        prefix.{mentions,clusters,links,quotes}.tsv (tabular format)
        prefix.conll (--fmt), and prefix.icarus (ICARUS allocation format)
    --exclude=<item1,item2,...>
        exclude given types of mentions/links from output:
            :singletons: mentions without any coreference links
            :npsingletons: non-name mentions without any coreference links
            :relpronouns: relative pronouns
            :reflectives: reflective pronouns
            :reciprocals: reciprocal pronouns
            :appositives: appositives NPs
            :predicatives: nominal predicatives

Instead of specifying a directory and gold file, can use the following presets:
    --clindev     run on CLIN26 shared task development data
    --semeval     run on SemEval 2010 development data
    --test        run tests
"""
import io
import os
import re
import sys
import getopt
import pandas
import tempfile
import subprocess
from collections import defaultdict
from itertools import islice
from bisect import bisect
from datetime import datetime
from glob import glob
from html import escape
from lxml import etree
from jinja2 import Template
import colorama
import ansi2html

TITLES = (
		'dr. drs. ing. ir. mr. lic. prof. mevr. mw. bacc. kand. dr.h.c. ds. '
		'bc. dr drs ing ir mr lic prof mevr mw bacc kand dr.h.c ds bc '
		'mevrouw meneer heer doctor professor').split()
STOPWORDS = (
		# List of Dutch Stop words (http://www.ranks.nl/stopwords/dutch)
		'aan af al als bij dan dat de die dit een en er had heb hem het hij '
		'hoe hun ik in is je kan me men met mij nog nu of ons ook te tot uit '
		'van was wat we wel wij zal ze zei zij zo zou ').split() + TITLES

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
# Weather verbs as they appear in the "lemma" attribute;
# e.g. het dooit; het kan morgen dooien; etc.
WEATHERVERBS = ('dooien gieten hagelen miezeren misten motregenen onweren '
		'plenzen regenen sneeuwen stormen stortregenen ijzelen vriezen '
		'weerlichten winteren zomeren').split()
VERBOSE = False
DEBUGFILE = sys.stdout


class Mention:
	"""A span referring to an entity.

	:ivar clusterid: cluster (entity) ID this mention is in.
	:ivar prohibit: do not link this mention to these mention IDs.
	:ivar filter: if True, do not include this mention in output.
	:ivar relaxedtokens: list of tokens without postnominal modifiers.
	:ivar head: node corresponding to head word.
	:ivar type: one of ('name', 'noun', 'pronoun')
	:ivar mainmod: list of string tokens that modify the head noun
	:ivar features: dict with following keys and possible values:
		:number: ('sg', 'pl', both, None); None means unknown.
		:gender: ('m', 'f', 'n', 'fm', 'nm', 'fn', None)
		:human: (0, 1, None)
		:person: (1, 2, 3, None); only for pronouns.
	:ivar antecedent: mention ID of antecedent of this mention, or None.
	:ivar sieve: name of sieve responsible for linking this mention, or None.
	"""
	def __init__(self, mentionid, sentno, parno, tree, node, begin, end,
			headidx, tokens, ngdata, gadata):
		"""Create a new mention.

		:param mentionid: unique integer for this mention.
		:param sentno: global sentence index (ignoring paragraphs, 0-indexed)
			this mention occurs in.
		:param tree: lxml.ElementTree with Alpino XML parse tree of sentence
		:param node: node in tree covering this mention
		:param begin: start index in sentence of mention (0-indexed)
		:param end: end index in sentence of mention (exclusive)
		:param headidx: index in sentence of head word of mention
		:param tokens: list of tokens in mention as strings
		:param ngdata: look up table with number and gender data
		:param gadata:: look up table with gender and animacy data
		"""
		self.id = mentionid
		self.sentno = sentno
		self.parno = parno
		self.node = node
		self.begin = begin
		self.end = end
		self.tokens = tokens
		self.clusterid = mentionid
		self.prohibit = set()
		self.filter = False
		self.antecedent = self.sieve = None
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
		self.head = (tree.find('.//node[@begin="%d"][@word]' % headidx)
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
		self._detectfeatures(ngdata, gadata)

	def _detectfeatures(self, ngdata, gadata):
		"""Set features for this mention based on linguistic features or
		external dataset."""
		firsttoken = self.node.find('.//node[@word][@begin="%d"]' % self.begin)
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
				self.features['gender'] = 'fm'
				self.features['human'] = 1
			elif self.head.get('persoon') == '3p':
				self.features['gender'] = 'fm'
				self.features['human'] = 1
			elif self.head.get('persoon') == '3o':
				self.features['gender'] = 'n'
				self.features['human'] = 0
			if self.head.get('lemma') == 'haar':
				self.features['gender'] = 'fn'
			elif self.head.get('lemma') == 'zijn':
				self.features['gender'] = 'nm'
			elif (self.head.get('lemma') in ('hun', 'hen')
					and self.head.get('vwtype') == 'pers'):
				self.features['human'] = 1
		# nouns: use lexical resource
		elif (self.head.get('neclass') is None
				and self.head.get('lemma', '').replace('_', '') in gadata):
			self.features.update(galookup(self.head.get('lemma', ''), gadata))
		elif (len(self.tokens) > 1 and firsttoken is not None
				and firsttoken.get('neclass') is None
				and self.tokens[0].lower() in gadata):  # e.g. "meneer Grey"
			self.features.update(galookup(self.tokens[0], gadata))
		else:  # names: dict
			if self.head.get('neclass') == 'PER':
				self.features['human'] = 1
				self.features['gender'] = 'fm'
			elif self.head.get('neclass') is not None:
				self.features['human'] = 0
				self.features['gender'] = 'n'
			result = nglookup(' '.join(self.tokens), ngdata)
			if result:
				self.features.update(result)
			elif (self.head.get('neclass') == 'PER'
					and self.tokens[0] not in STOPWORDS):
				# Assume first token is first name.
				self.features.update(nglookup(self.tokens[0], ngdata))

	def featrepr(self, extended=False):
		"""String representations of features."""
		result = ' '.join('%s=%s' % (a, '?' if b is None else b)
					for a, b in self.features.items())
		result += ' inquote=%d' % int(self.head.get('quotelabel') == 'I')
		if extended:
			result += ' neclass=%s head=%s' % (
					self.head.get('neclass'), self.head.get('word'))
		return result

	def __str__(self):
		return "'%s'" % color(' '.join(self.tokens), 'green')

	def __repr__(self):
		return "Mention('%s', ...)" % ' '.join(self.tokens)


class Quotation:
	"""A span of direct speech.

	:ivar speaker: detected speaker Mention object.
	:ivar addressee: detected addressee Mention object.
	:ivar mentions: list of Mention objects occurring in this quote.
	"""
	def __init__(self, start, end, sentno, endsentno, parno, text, sentbounds):
		"""
		:param start: global token start index.
		:param end: global token end index.
		:param sentno: global sentence index (ignoring paragraphs).
		:param endsentno: relevant for multi-sentence quotes.
		:param parno: paragraph number.
		:param text: text of quote as string (including quote marks).
		:param sentbounds: bool, whether quote starts+ends at sent boundaries.
		"""
		self.start, self.end = start, end
		self.sentno = sentno
		self.endsentno = endsentno
		self.parno = parno
		self.sentbounds = sentbounds
		self.speaker = None
		self.addressee = None
		self.mentions = []
		self.text = text

	def __repr__(self):
		return '%d %d %r' % (self.parno, self.sentno, self.text)


def getmentions(trees, ngdata, gadata):
	"""Collect mentions."""
	debug(color('mention detection', 'yellow'))
	mentions = []
	for sentno, ((parno, _), tree) in enumerate(trees):
		candidates = []
		# Order is significant.
		candidates.extend(tree.xpath(
				'.//node[@pdtype="pron" or @vwtype="bez"]'))
		candidates.extend(tree.xpath('.//node[@cat="np"]'))
		# candidates.extend(tree.xpath(
		# 		'.//node[@cat="conj"]/node[@cat="np" or @pt="n"]/..'))
		candidates.extend(tree.xpath(
				'.//node[@cat="mwu"]/node[@pt="spec"]/..'))
		candidates.extend(tree.xpath('.//node[@pt="n"]'
				'[@ntype="eigen" or @rel="su" or @rel="obj1" or @rel="body" '
				'or @special="er_loc"]'))
		candidates.extend(tree.xpath(
				'.//node[@pt="num" and @rel!="det" and @rel!="mod"]'))
		candidates.extend(tree.xpath('.//node[@pt="det" and @rel!="det"]'))
		covered = set()
		for candidate in candidates:
			considermention(candidate, tree, sentno, parno, mentions, covered,
					ngdata, gadata)
	return mentions


def considermention(node, tree, sentno, parno, mentions, covered,
		ngdata, gadata):
	"""Decide whether a candidate mention should be added."""
	if len(node) == 0 and 'word' not in node.keys():
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
	# remove "vc" clause from NP:
	# [Behrmans voorstel] om samen te eten
	# [het feit] dat ...
	vc = node.find('./node[@rel="vc"]')
	if vc is not None and int(vc.get('begin')) < b:
		b = int(vc.get('begin'))
		if headidx > b:
			headidx = max(int(a.get('begin')) for a
					in node.findall('.//node[@word]')
					if int(a.get('begin')) < b)
	# Relative clauses: [de man] [die] ik eerder had gezien.
	# [een buurt] [waar] volgens hem ondanks de Wende niets veranderd was.
	relpronoun = node.find('./node[@cat="rel"]/node[@rel="rhd"]')
	if relpronoun is not None and int(relpronoun.get('begin')) < b:
		b = int(relpronoun.get('begin'))
		if headidx > b:
			headidx = max(int(a.get('begin')) for a
					in node.findall('.//node[@word]')
					if int(a.get('begin')) < b)
	# Appositives: "[Jan], [de schilder]"
	if len(node) > 1 and node[1].get('rel') == 'app':
		if (node[1].get('ntype') != 'eigen'
				and node[1].get('pt') != 'spec'
				and (node[1].get('cat') != 'mwu'
					or node[1][0].get('pt') != 'spec')):
			node = node[0]
			b = int(node.get('end'))
		else:  # but: "[acteur John Cleese]"; head: Cleese.
			h1 = getheadidx(node[1])
			if h1 < b:
				headidx = h1
	# mevrouw [Steele] => [mevrouw Steele]
	if (node.get('rel') == 'nucl' and a
			and gettokens(tree, a - 1, a)[0].lower() in TITLES):
		a -= 1
		headidx = a
	tokens = gettokens(tree, a, b)
	# Trim punctuation
	if tokens[0] in ',\'"()':
		tokens = tokens[1:]
		a += 1
	if tokens[-1] in ',\'"()':
		tokens = tokens[:-1]
		b -= 1
	head = (tree.find('.//node[@begin="%d"]' % headidx)
			if len(node) else node)
	# various
	if head.get('lemma') in ('aantal', 'keer', 'toekomst', 'manier'):
		return
	if pleonasticpronoun(node):
		return
	if (headidx not in covered
			# discard measure phrases
			# and node.find('.//node[@num="meas"]') is None
			and node.get('num') != "meas"
			and node.find('./node[@pt="tw"]') is None
			# and not tokens[0].isnumeric()
			# "a few" ...
			and (node.get('cat') != 'np' or node.get('rel') != 'det')
			# "welk" in "Ik ga niet zeggen welk restaurant"
			and node.find('.//node[@begin="%d"][@vwtype="onbep"]' % a) is None
			and node.find('.//node[@begin="%d"][@vwtype="vb"]' % a) is None
			# "iets"
			and node.get('vwtype') not in ('onbep', 'vb')
			# temporal expressions
			and head.get('special') != 'tmp' and node.get('special') != 'tmp'
			# partitive / quantifier
			# ongeveer 12 dollar
			and node.find('./node[@sc="noun_prep"]') is None
			# and (node.get('cat') != 'np'
			# 	or node[0].get('pos') not in ('adj', 'noun'))
			# het fietsen
			and head.get('pt') != 'ww'
			):
		mentions.append(Mention(
				len(mentions), sentno, parno, tree, node, a, b, headidx,
				tokens, ngdata, gadata))
		covered.add(headidx)
		# California in "San Jose, California"
		if (node.get('cat') == 'mwu' and head.get('neclass') == 'LOC'
				and ',' in tokens):
			mentions.append(Mention(
					len(mentions), sentno, parno, tree, node,
					a + tokens.index(',') + 1, b, b - 1,
					tokens[tokens.index(',') + 1:],
					ngdata, gadata))
		elif len(node) > 1 and node[0].get('rel') == 'cnj':
			mentions[-1].features['number'] = 'pl'
			for cnj in node.findall('./node[@rel="cnj"]'):
				a = int(cnj.get('begin'))
				b = int(cnj.get('end'))
				mentions.append(Mention(
						len(mentions), sentno, parno, tree, cnj,
						a, b, getheadidx(cnj),
						gettokens(tree, a, b), ngdata, gadata))


def pleonasticpronoun(node):
	"""Return True if node is a pleonastic (non-referential) pronoun."""
	# Examples from Lassy syntactic annotation manual.
	if node.get('rel') in ('sup', 'pobj1'):
		return True
	if node.get('lemma') == 'dat' and node.get('vwtype') == 'aanw':
		return True
	if node.get('lemma') == 'het' and node.get('rel') == 'su':
		head = node.find('../node[@rel="hd"]')
		# het regent. / hoe gaat het?
		if head.get('lemma') in WEATHERVERBS or head.get('lemma') == 'gaan':
			return True
		if (head.get('lemma') == 'ontbreken'
				and node.find('../node[@rel="pc"]'
					'/node[@rel="hd"][@lemma="aan"]') is not None):
			return True
		# het kan voorkomen dat ...
		if (node.get('index') and node.get('index')
				in node.xpath('../node//node[@rel="sup"]/@index')):
			return True
		# FIXME: add rules to detect non-pleonastic use
		return True  # assume pleonastic ...
	if node.get('lemma') == 'het' and node.get('rel') == 'obj1':
		head = node.find('../node[@rel="hd"]')
		head = '' if head is None else head.get('lemma')
		# (60) de presidente had het warm
		if head == 'hebben' and node.find('../node[@rel="predc"]') is not None:
			return True
		# (61) samen zullen we het wel rooien.
		if head == 'rooien':
			return True
		# (62) hij zette het op een lopen
		if (head == 'zetten' and node.find('../node[@rel="svp"]/'
				'node[@word="lopen"]') is not None):
			return True
		# (63) had het op mij gemunt.
		if head == 'munten' and node.find('..//node[@word="op"]') is not None:
			return True
		# (64) het erover hebben
		if (head == 'hebben'
				and (node.find('../node[@word="erover"]') is not None
					or (node.find('..//node[@word="er"]') is not None
						and node.find('..//node[@word="over"]') is not None))):
			return True
		# FIXME: add rules to detect non-pleonastic use
		return True  # assume pleonastic ...
	return False


def getquotations(trees):
	"""Detect quoted speech spans and speaker / addressee if possible.

	Marks tokens in quoted speech with B, I, O labels.

	- Quoted speech within other quoted speech is not marked.
	- Quoted speech ends at end of paragraph even if no marker is found.
	- Quoted speech can be introduced by ASCII single '/` or double quotes "/``
		or by a dash '-' at the start of a paragraph. Unicode quotes should be
		normalized to ASCII quotes in preprocessing.
	"""
	# dictionary mapping open quote char to closing quote char
	quotechar = {
			"'": "'",
			'"': '"',
			'`': "'",
			'``': "''",
			}
	# convert to flat list of tokens
	doc = []  # list of tokens as XML nodes by global token index
	parbreak = []  # True if new paragraph starts at token index
	idx = {}  # map (sentno, tokenno) to global token index
	sentnos = []  # map global token index to sentno
	parnos = []  # map global token index to parno
	i = 0
	for sentno, ((parno, psentno), tree) in enumerate(trees):
		for n, token in enumerate(sorted(
				tree.iterfind('.//node[@word]'),
				key=lambda x: int(x.get('begin')))):
			doc.append(token)
			parbreak.append(psentno == 1 and n == 0)
			idx[sentno, n] = i
			sentnos.append(sentno)
			parnos.append(parno)
			i += 1
	quotations = []
	inquote = None
	for i, token in enumerate(doc):
		n = int(token.get('begin'))
		if inquote and parbreak[i]:
			inquote = None
			end = i
			quotations.append(Quotation(start, end,
					sentnos[start], sentnos[end - 1], parnos[start],
					' '.join(doc[i].get('word') for i
						in range(start, end)),
					True))
		if inquote is None and parbreak[i] and token.get('word') == '-':
			# detect implied end of quote:
			# - I like cats, he said
			token.set('quotelabel', 'B')
			start = i
			_, tree = trees[sentnos[i]]
			verb = tree.getroot().find('./node/node[@cat="du"]/node[@cat="sv1"]'
					'[@rel="tag"]/node[@rel="hd"]')
			if verb is not None and verb.get('root') in SPEECHVERBS:
				node = tree.getroot().find(
						'./node/node[@cat="du"]/node[@rel="nucl"]')
				end = idx[sentnos[i], int(node.get('end'))]
				quotations.append(Quotation(start, end,
						sentnos[start], sentnos[end - 1], parnos[start],
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
					sentnos[start], sentnos[end - 1], parnos[start],
					' '.join(doc[i].get('word') for i
						in range(start, end)),
					int(doc[start].get('begin')) == 0
						and (i + 2 >= len(doc)
							or int(doc[i + 1].get('begin')) == 0
							or int(doc[i + 2].get('begin')) == 0)))
		elif token.get('quotelabel') is None:
			token.set('quotelabel', 'I' if inquote else 'O')
	return quotations, idx, doc


def isspeaker(mention):
	"""Test whether mention is subject of a reported speech verb."""
	if (mention.node.get('rel') != 'su'
			or mention.head.get('quotelabel') == 'I'):
		return False
	head1 = mention.node.find('../node[@cat="ppart"]/node[@rel="hd"]')
	head2 = mention.node.find('../node[@rel="hd"]')
	head = head2 if head1 is None else head1
	return head is not None and head.get('root') in SPEECHVERBS


def speakeridentification(mentions, quotations, idx, doc):
	"""Identify speakers and addressees for quotations."""
	debug(color(
			'speaker identification (%d quotations)' % len(quotations),
			'yellow'))
	if not quotations:
		return
	tokenidx2quotation = {i: quotation
			for quotation in quotations
				for i in range(quotation.start, quotation.end)}
	# collect mentions within each quote
	for mention in mentions:
		if idx[mention.sentno, mention.begin] in tokenidx2quotation:
			i = idx[mention.sentno, mention.begin]
			tokenidx2quotation[i].mentions.append(mention)
	# collect mentions within each paragraph
	par2mention = defaultdict(list)
	for mention in mentions:
		# Only consider human mentions that are not possessive pronouns.
		if (mention.features['human']
				and mention.head.get('quotelabel') != 'I'
				and mention.head.get('vwtype') != 'bez'):
			par2mention[mention.parno].append(mention)

	# quote attribution sieves; cf. https://aclweb.org/anthology/E17-1044
	vocative(quotations, doc, idx, tokenidx2quotation)
	reportedspeech(mentions, quotations, doc, idx)
	singularmentionspeaker(quotations, par2mention)
	splitquotes(quotations)
	closestmention(quotations, par2mention, idx)
	turntaking(quotations)
	turntaking(quotations, strict=False)

	speakerconstraints(quotations, doc)


def vocative(quotations, doc, idx, tokenidx2quotation):
	"""Vocative => addressee; e.g. 'John, do the dishes.'"""
	# vocative patterns from https://aclweb.org/anthology/E17-1044
	for quotation in quotations:
		for mention in quotation.mentions:
			if mention.features['human']:
				try:
					w1 = doc[idx[mention.sentno, mention.begin] - 1].get('word')
					w2 = doc[idx[mention.sentno, mention.end]].get('word')
				except (KeyError, IndexError):
					continue
				if ((w1 == ',' and w2 in '!?.,;')
						or (w1 in '\'"' and w2 == ',')
						or (w1 == ',' and w2 in '\'"')
						or (w1.lower() == 'o' and w2 == '!')
						or (w1.lower() in ('beste', 'lieve'))):
					quotation.addressee = mention
					debug('detected vocative %s\n\taddressee %s' % (
							color(quotation.text, 'green'), mention))
					break


def reportedspeech(mentions, quotations, doc, idx):
	"""Link each subject of a reported speech verb to the closest quotation."""
	qstarts = [q.start for q in quotations]
	qends = [q.end for q in quotations]
	for mention in mentions:
		if isspeaker(mention):
			i = idx[mention.sentno, mention.begin]
			i1 = idx[mention.sentno, 0]
			i2 = idx.get((mention.sentno + 1, 0), len(doc))
			# first quote to left of mention
			q1 = quotations[bisect(qends, i) - 1]
			if i1 < q1.end <= i2 and i - q1.end <= 5 and q1.speaker is None:
				q1.speaker = mention
				debug('%s\n\tis before reported speech verb; speaker: %s'
						% (color(q1.text, 'green'), q1.speaker))
			else:
				# first quote to right of mention
				x = bisect(qstarts, i)
				q2 = quotations[x if x < len(quotations) else x - 1]
				if (i1 <= q2.start < i2 and q2.start - i <= 5
						and q2.speaker is None):
					q2.speaker = mention
					debug('%s\n\tis after reported speech verb; speaker: %s'
							% (color(q2.text, 'green'), q2.speaker))


def singularmentionspeaker(quotations, par2mention):
	"""Attribute quote if there is a single human mention in paragraph."""
	for quotation in quotations:
		if (quotation.speaker is None
				and len(par2mention[quotation.parno]) == 1):
			quotation.speaker = par2mention[quotation.parno][0]
			debug('attributed %s\n\tto singular non-quoted human mention '
					'in paragraph: %s' % (color(quotation.text, 'green'),
						quotation.speaker))


def splitquotes(quotations):
	"""Assume same speaker for consecutive quotations in same paragraph.

	Example: "I don't know", he said. "It seems a bad idea."
	Only applies when there is material outside quotes."""
	for prev, quotation in zip(quotations, quotations[1:]):
		if (quotation.parno == prev.parno
				and quotation.sentno <= prev.sentno + 1
				and not prev.sentbounds):
			if quotation.speaker is None and prev.speaker is not None:
				quotation.speaker = prev.speaker
				debug('%s\n\tis directly after previous quote; '
						'assuming same speaker %s'
						% (color(quotation.text, 'green'), prev.speaker))
			if quotation.addressee is None and prev.addressee is not None:
				quotation.addressee = prev.addressee
				debug('%s\n\tis directly after previous quote; '
						'assuming same addressee %s'
						% (color(quotation.text, 'green'), prev.speaker))


def closestmention(quotations, par2mention, idx):
	# For unattributed quotes without material before or after quote in the
	# sentence, assign closest human mention in same paragraph.
	# When there is material before or after the quote, it may not be a
	# dialogue turn, e.g.
	# 'Every happy family is alike,' according to Tolstoy's Anna Karenina.
	for prev, quotation in zip([None] + quotations, quotations):
		if quotation.speaker is None and quotation.sentbounds:
			# Find closest mention in sentence before or after quote
			candidates = [mention for mention in par2mention[quotation.parno]
					if ((quotation.sentno - mention.sentno == 1
							or mention.sentno - quotation.endsentno == 1)
						and (prev is None
							or prev.end <= idx[mention.sentno, mention.begin])
						and quotation.addressee != mention)]
			for mention in sorted(
					candidates,
					key=lambda m: quotation.start - m.end
						if quotation.start >= idx[m.sentno, m.end]
						else idx[m.sentno, m.begin] - quotation.end):
				quotation.speaker = mention
				debug('assuming bare quotation %s\n'
						'\tspoken by closest non-quoted human mention: %s' % (
						color(quotation.text, 'green'), quotation.speaker))
				break


def turntaking(quotations, strict=True):
	"""Heuristics for consecutive quotations.

	:param strict: if True, only consider quotations without intervening
		sentences; if False, also consider consecutive paragraphs with
		non-quoted sentences."""
	# Consecutive quotations in different paragraphs are turn taking;
	# or when first quotation has no material outside quotation marks
	# "How are you?" "I'm fine."
	# By propagating speakers and addressees,
	# we capture 2n and 2n+1 turntaking patterns (ABABAB...)
	# FIXME: how to detect multiple consecutive turns (AAAB)
	for prev, quotation in zip(quotations, quotations[1:]):
		if ((strict and quotation.sentno == prev.endsentno + 1
					and quotation.parno == prev.parno + 1)
				or (not strict and quotation.parno == prev.parno + 1)
				or (quotation.parno == prev.parno
					and quotation.sentno == prev.sentno + 1
					and quotation.sentbounds and prev.sentbounds)):
			if (quotation.speaker is None
					and prev.addressee is not None
					and prev.addressee != quotation.addressee
					and (quotation.addressee is None
						or prev.addressee.tokens != quotation.addressee.tokens)
					):
				quotation.speaker = prev.addressee
				debug('assuming %s\n\tis spoken by previous addressee %s' % (
						color(quotation.text, 'green'), prev.addressee))
			if (prev.speaker is None
					and quotation.addressee is not None
					and quotation.addressee != prev.addressee
					and (prev.addressee is None
						or quotation.addressee.tokens != prev.addressee.tokens)
					):
				prev.speaker = quotation.addressee
				debug('assuming %s\n\tis spoken by addressee %s of next quote'
						% (color(prev.text, 'green'), quotation.addressee))
			if (quotation.addressee is None
					and prev.speaker is not None
					and prev.speaker != quotation.speaker
					and (quotation.speaker is None
						or prev.speaker.tokens != quotation.speaker.tokens)
					):
				quotation.addressee = prev.speaker
				debug('assuming %s\n\tis addressed to previous speaker %s' % (
						color(quotation.text, 'green'), prev.speaker))
			if (prev.addressee is None
					and quotation.speaker is not None
					and quotation.speaker != prev.speaker
					and (prev.speaker is None
						or quotation.speaker.tokens != prev.speaker.tokens)
					):
				prev.addressee = quotation.speaker
				debug('assuming %s\n\tis addressed to speaker %s of next quote'
						% (color(prev.text, 'green'), quotation.speaker))


def speakerconstraints(quotations, doc):
	"""Add speaker constraints."""
	for prev, quotation in zip([None] + quotations, quotations):
		if quotation.speaker is None:
			debug('no speaker: %s' % color(quotation.text, 'green'))
		for i in range(quotation.start, quotation.end):
			if quotation.speaker is not None:
				doc[i].set('speaker', str(quotation.speaker.clusterid))
			if quotation.addressee is not None:
				doc[i].set('addressee', str(quotation.addressee.clusterid))
		nominalmentions = [mention for mention in quotation.mentions
				if mention.type == 'noun']
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


# end of dialogue related functions

def stringmatch(mentions, clusters, relaxed=False):
	"""Link mentions with matching strings;
	if relaxed, ignore modifiers/appositives."""
	debug(color('string match (relaxed=%s)' % relaxed, 'yellow'))
	sieve = 'stringmatch:relaxed' if relaxed else 'stringmatch'
	foundentities = {}
	for _, mention in representativementions(mentions, clusters):
		if mention.type != 'pronoun':
			if (len(clusters[mention.clusterid]) == 1
					and mention.node.get('ntype') == 'soort'
					and mention.features['number'] == 'pl'):
				continue
			mstr = ' '.join(mention.relaxedtokens
					if relaxed else mention.tokens).lower()
			if mstr in foundentities:
				merge(foundentities[mstr], mention, sieve, mentions, clusters)
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
	acronyms = {}

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
		if (mention.head.get('neclass') is not None
				and len(mention.tokens) > 1):
			# FIXME: detect and avoid ambiguous acronyms
			# FIXME: prefer explicit cases: "Full NP (acronym)"
			# De Partij van de Arbeid => PvdA
			acronyms[''.join(token[0] for n, token in enumerate(mention.tokens)
					if token.lower() not in STOPWORDS or n > 0)] = mention
			# De Koninklijke Nederlandse Akademie van Wetenschappen => KNAW
			acronyms[''.join(token[0] for token in mention.tokens
					if token.lower() not in STOPWORDS)] = mention

	# Pass 2: find mentions to link to collected antecedents
	for mention in mentions:
		if (mention.node.get('rel') == 'app'
				and (mention.sentno, mention.begin, mention.end)
					in appositives):
			merge(appositives[mention.sentno, mention.begin, mention.end],
					mention, 'precise:appositive', mentions, clusters)
		if (mention.node.get('rel') == 'predc'
				and (mention.sentno,
					mention.node.get('begin'),
					mention.node.get('end'))
				in predicatives):
			merge(predicatives[mention.sentno,
					mention.node.get('begin'),
					mention.node.get('end')],
					mention, 'precise:predicative', mentions, clusters)
		if (mention.node.get('vwtype') == 'betr'
				and (mention.sentno, mention.begin, mention.end)
				in relpronouns):
			merge(relpronouns[mention.sentno, mention.begin, mention.end],
					mention, 'precise:relpronoun', mentions, clusters)
		if (mention.node.get('vwtype') == 'refl'
				and (mention.sentno, mention.begin, mention.end)
				in reflpronouns):
			merge(reflpronouns[mention.sentno, mention.begin, mention.end],
					mention, 'precise:reflective', mentions, clusters)
		if (mention.node.get('vwtype') == 'recip'
				and (mention.sentno, mention.begin, mention.end)
				in recippronouns):
			merge(recippronouns[mention.sentno, mention.begin, mention.end],
					mention, 'precise:reciprocal', mentions, clusters)
		if (mention.head.get('neclass') is not None
				and len(mention.tokens) == 1
				and sum(a.isupper() for a in mention.tokens[0]) > 1):
			# an acronym is a token with two or more upper case characters.
			acr = ''.join(a for a in mention.tokens[0] if a.isalnum())
			if acr in acronyms:
				merge(acronyms[acr], mention, 'precise:acronym',
						mentions, clusters)


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
	for n, mention in representativementions(mentions, clusters):
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
					merge(other, mention, 'strictheadmatch:%d' % sieve,
							mentions, clusters)
					heads[mention.clusterid] = heads[
							mention.clusterid] | otherheads


def properheadmatch(mentions, clusters, relaxed=False):
	"""Link mentions with same proper noun head."""
	debug(color('proper head match (relaxed=%s)' % relaxed, 'yellow'))
	sieve = 'properheadmatch:relaxed' if relaxed else 'properheadmatch'
	othernonstop = {clusterid:
			{token for m in cluster
				for token in mentions[m].tokens}
			for clusterid, cluster in enumerate(clusters)
			if cluster is not None}
	for _, mention in representativementions(mentions, clusters):
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
							merge(other, mention, sieve, mentions, clusters)
						else:
							merge(mention, other, sieve, mentions, clusters)
				elif mention.head.get('lemma') == other.head.get('lemma'):
					if other.sentno < mention.sentno:
						merge(other, mention, sieve, mentions, clusters)
					else:
						merge(mention, other, sieve, mentions, clusters)


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
				merge(pronouns[0], a, 'resolvepronouns', mentions, clusters)
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
	for _, mention in representativementions(mentions, clusters):
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
				# The antecedent should come before the anaphor,
				# and should not contain the anaphor.
				if (other.sentno == mention.sentno
						and (other.begin >= mention.begin
							# allow: [de raket met [haar] massa van 750 ton]
							or (mention.head.get('vwtype') != 'bez'
								and other.end >= mention.end))):
					debug('\t%d %d %s %d %s prohibited=%d i-within-i or >' % (
							other.sentno, other.begin, other.node.get('rel'),
							len(clusters[other.clusterid]),
							other,
							int(prohibited(mention, other, clusters))))
					continue
				# An anaphor (mention) cannot be a coargument of its
				# antecedent (other). Coarguments are in the same clause
				# but do not necessarily have the same parent.
				# Do not apply restriction to possessives; e.g.
				# [de raket met [haar] massa van 750 ton]
				if (mention.head.get('vwtype') != 'bez'
						and sameclause(other.node, mention.node)
						and other.node.find('..//node[@id="%s"]'
						% mention.node.get('id')) is not None):
					mention.prohibit.add(other.id)
					other.prohibit.add(mention.id)
					debug('\t%d %d %s %d %s prohibited=1 coargument' % (
							other.sentno, other.begin, other.node.get('rel'),
							len(clusters[other.clusterid]),
							other))
					continue
				iscompatible = compatible(mention, other)
				isprohibited = prohibited(mention, other, clusters)
				debug('\t%d %d %s %d %s prohibited=%d compatible=%d' % (
						other.sentno, other.begin, other.node.get('rel'),
						len(clusters[other.clusterid]), other,
						int(isprohibited), int(iscompatible)))
				if (iscompatible and not isprohibited):
					merge(other, mention, 'resolvepronouns',
							mentions, clusters)
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
					merge(pronouns[0], a, 'resolvepronouns', mentions, clusters)
				# I in quote is speaker
				if (pronouns and person == '1' and number == 'sg'
						and quotation.speaker is not None):
					merge(quotation.speaker, pronouns[0], 'resolvepronouns',
							mentions, clusters)
				# you in quote is addressee
				elif (pronouns and person == '2' and number == 'sg'
						and quotation.addressee is not None):
					merge(quotation.addressee, pronouns[0], 'resolvepronouns',
							mentions, clusters)


def resolvecoreference(trees, ngdata, gadata, mentions=None):
	"""Get mentions and apply coreference sieves."""
	if mentions is None:
		mentions = getmentions(trees, ngdata, gadata)
	clusters = [{n} for n, _ in enumerate(mentions)]
	quotations, idx, doc = getquotations(trees)
	if VERBOSE:
		for mention in mentions:
			debug(mention, mention.featrepr(extended=True),
					# nglookup(' '.join(mention.tokens).lower(), ngdata),
					# nglookup(mention.tokens[0].lower(), ngdata),
					# gadata.get(mention.head.get('lemma', '').replace('_', '')),
					)
	speakeridentification(mentions, quotations, idx, doc)
	stringmatch(mentions, clusters)
	stringmatch(mentions, clusters, relaxed=True)
	preciseconstructs(mentions, clusters)
	strictheadmatch(mentions, clusters, 5)
	strictheadmatch(mentions, clusters, 6)
	strictheadmatch(mentions, clusters, 7)
	properheadmatch(mentions, clusters)
	properheadmatch(mentions, clusters, relaxed=True)
	resolvepronouns(mentions, clusters, quotations)
	return mentions, clusters, quotations, idx


def parsesentid(path):
	"""Given a filename, return tuple with numeric components for sorting.

	Accepts three formats: 1.xml, 1-2.xml, abc.p.1.s.2.xml """
	filename = os.path.basename(path)
	x = tuple(map(int, re.findall(r'\d+', filename.rsplit('.', 1)[0])))
	if len(x) == 1:
		return 0, x[0]
	elif re.match(r'\d+-\d+.xml', filename):
		return x
	elif re.match(r'.*p\.[0-9]+\.s\.[0-9]+\.xml', filename):
		return x[-2:]
	else:
		raise ValueError('expected sentence ID of the form sentno.xml, '
				'parno-sentno.xml, p.parno.s.sentno.xml. Got: %s' % filename)


def gettokens(tree, begin, end):
	"""Return tokens of span in tree as list of strings."""
	return [token.get('word') for token
			in sorted((token for token
				in tree.findall('.//node[@word]')
				if begin <= int(token.get('begin')) < end),
			key=lambda x: int(x.get('begin')))]


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


def merge(mention1, mention2, sieve, mentions, clusters):
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
	mention2.antecedent = mention1.id
	mention2.sieve = sieve
	debug('Linked  %d %d %s %s\n\t%d %d %s %s' % (
			mention1.sentno, mention1.begin, mention1, mention1.featrepr(),
			mention2.sentno, mention2.begin, mention2, mention2.featrepr()))


def mergefeatures(mention, other):
	"""Update the features of the first mention with those of second.
	In case one is more specific than the other, keep specific value.
	In case of conflict, keep both values."""
	for key in mention.features:
		if (key == 'person' or mention.features[key] == other.features[key]
				or other.features[key] in (None, 'both')):
			pass  # only pronouns should have a 'person' attribute
		elif mention.features[key] in (None, 'both'):
			mention.features[key] = other.features[key]
		elif key == 'human':
			mention.features[key] = None
		elif key == 'number':
			mention.features[key] = 'both'
		elif key == 'gender':
			if other.features[key] in mention.features[key]:  # (fm, m) => m
				mention.features[key] = other.features[key]
			elif mention.features[key] in other.features[key]:  # (m, fm) => m
				pass
			elif (len(other.features[key]) == len(mention.features[key])
					== 1):  # (f, m) => fm
				mention.features[key] = ''.join(sorted((
						other.features[key], mention.features[key])))
			else:  # e.g. (fm, n) => unknown
				mention.features[key] = None
	other.features.update((a, b) for a, b in mention.features.items()
			if a != 'person')


def compatible(mention, other):
	"""Return True if all features are compatible."""
	return all(
			mention.features[key] == other.features[key]
			or None in (mention.features[key], other.features[key])
			or (key == 'gender'
				and 'fm' in (mention.features[key], other.features[key])
				and 'n' not in (mention.features[key], other.features[key]))
			or (key == 'gender'
				and 'nm' in (mention.features[key], other.features[key])
				and 'f' not in (mention.features[key], other.features[key]))
			or (key == 'gender'
				and 'fn' in (mention.features[key], other.features[key])
				and 'm' not in (mention.features[key], other.features[key]))
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


def sortmentions(mentions):
	"""Sort mentions by start position, then from small to large span length.
	"""
	return sorted(mentions,
			key=lambda x: (x.sentno, x.begin, x.end))


def representativementions(mentions, clusters):
	"""Yield the representative mention (here the first) for each cluster."""
	for cluster in clusters:
		if cluster is not None:
			n = min(cluster)
			yield n, mentions[n]


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


def createngdatadf():
	"""Create pickled version of ngdata DataFrame."""
	with open('../groref/ngdata', 'rb') as inp:
		df = pandas.DataFrame.from_dict(
				{line[:line.index(b'\t')]:
					[int(a) for a in line[line.index(b'\t') + 1:].split(b' ')]
				for line in inp},
				orient='index')
	df.columns = ['male', 'female', 'neuter', 'plural']
	df.to_pickle('data/ngdata.pkl')
	return df


def readngdata():
	"""Read noun phrase number-gender counts."""
	if (os.path.exists('data/ngdata.pkl')
			and os.stat('data/ngdata.pkl').st_mtime
			> os.stat('../groref/ngdata').st_mtime):
		ngdata = pandas.read_pickle('data/ngdata.pkl')
	else:
		ngdata = createngdatadf()
	gadata = {}  # Format: {noun: (gender, animacy)}
	with open('data/gadata', encoding='utf8') as inp:
		for line in inp:
			a, b, c = line.rstrip('\n').split('\t')
			gadata[a] = b, c
	# https://www.meertens.knaw.nl/nvb/
	with open('data/Top_eerste_voornamen_NL_2010.csv',
			encoding='latin1') as inp:
		for line in islice(inp, 2, None):
			fields = line.split(';')
			if fields[1]:
				gadata[fields[1]] = ('f', 'human')
			if fields[3]:
				gadata[fields[3]] = ('m', 'human')
	return ngdata, gadata


def nglookup(key, ngdata):
	"""Look up key in ngdata DataFrame.

	:returns: a dictionary with features."""
	key = key.lower().encode('utf8')
	if not key or key not in ngdata.index:
		return {}
	genderdata = list(ngdata.loc[key, :])
	if (genderdata[0] > sum(genderdata) / 3
			and genderdata[1] > sum(genderdata) / 3):
		return {'number': 'sg', 'gender': 'fm', 'human': 1}
	elif genderdata[0] > sum(genderdata) / 3:
		return {'number': 'sg', 'gender': 'm', 'human': 1}
	elif genderdata[1] > sum(genderdata) / 3:
		return {'number': 'sg', 'gender': 'f', 'human': 1}
	elif genderdata[2] > sum(genderdata) / 3:
		return {'number': 'sg', 'gender': 'n', 'human': 0}
	elif genderdata[3] > sum(genderdata) / 3:
		return {'number': 'pl', 'gender': 'n'}
	return {}


def galookup(key, gadata):
	"""Look up word in gadata."""
	result = {}
	key = key.lower().replace('_', '')
	if key and key in gadata:
		gender, animacy = gadata[key]
		if animacy == 'human':
			result['human'] = 1
			if gender in ('m', 'f'):
				result['gender'] = gender
			else:
				result['gender'] = 'fm'
		else:
			result['human'] = 0
			result['gender'] = 'n'
	return result


def writetabular(trees, mentions,
		docname='-', part=0, file=sys.stdout, fmt=None, startcluster=0):
	"""Write output in tabular format."""
	sentences = [sorted(tree.iterfind('.//node[@word]'),
				key=lambda x: int(x.get('begin')))
			for _, tree in trees]
	sentids = ['%d-%d' % (parno, sentno) for (parno, sentno), _ in trees]
	labels = [[''] * len(sent) for sent in sentences]
	for mention in sortmentions(mentions):
		if mention.filter:
			continue
		labels[mention.sentno][mention.begin] = '|(%d%s%s' % (
				mention.clusterid + startcluster,
				')' if mention.begin == mention.end - 1 else '',
				labels[mention.sentno][mention.begin])
		if mention.begin != mention.end - 1:
			labels[mention.sentno][mention.end - 1] = '|%d)%s' % (
					mention.clusterid + startcluster,
					labels[mention.sentno][mention.end - 1])
	labels = [[a.lstrip('|') or '-' for a in coreflabels]
			for n, coreflabels in enumerate(labels, 1)]
	doctokenid = 0
	if fmt == 'semeval2010':
		print('#begin document %s' % docname, file=file)
	elif part is None:  # CLIN evaluation scripts don't support part numbers
		print('#begin document (%s);' % docname, file=file)
	else:
		print('#begin document (%s); part %03d' % (docname, part), file=file)
	for sentid, sent, sentlabels in zip(sentids, sentences, labels):
		for tokenid, (token, label) in enumerate(zip(sent, sentlabels), 1):
			doctokenid += 1
			if fmt is None or fmt == 'minimal':
				print(docname, doctokenid, token.get('word'), label,
						sep='\t', file=file)
			elif fmt == 'conll2012':
				print(docname, part, tokenid - 1, token.get('word'),
						*(['-'] * 5), '*', label,
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
						token.get('speaker', '-'),  # clusterid
						token.get('addressee', '-'),  # clusterid
						token.get('quotelabel', '-'),  # B, I, O
						label,
						sep='\t', file=file)
		print('', file=file)
	if fmt == 'semeval2010':
		print('#end document %s' % docname, file=file)
	else:
		print('#end document', file=file)


def writeinfo(mentions, clusters, quotations, idx, prefix,
		docname='-', part=0):
	"""Write extra information to several files."""
	# spans are 1-indexed document token IDs, end is inclusive.
	with open(prefix + '.clusters.tsv', 'w') as out:
		print('id\tgender\thuman\tnumber\tsize\tfirstmention\tmentions\tlabel',
				file=out)
		for n, cluster in enumerate(clusters):
			if cluster is None:
				continue
			mention = mentions[min(cluster)]
			print('\t'.join((
					str(n), '\t'.join(
						'-' if mention.features[key] is None
						else str(mention.features[key])
						for key in ('gender', 'human', 'number')),
					str(len(cluster)), str(mention.id),
					','.join(str(m) for m in cluster),
					' '.join(mention.tokens).replace('\t', ' '))), file=out)
	with open(prefix + '.mentions.tsv', 'w') as out:
		print('id\tstart\tend\ttype\thead\tneclass\tperson\tquote'
				'\tgender\thuman\tnumber\tcluster\ttext', file=out)
		for mention in mentions:
			print('\t'.join((
					str(mention.id),
					str(idx[mention.sentno, mention.begin] + 1),
					str(idx[mention.sentno, mention.end - 1] + 1),
					mention.type,
					str(idx[mention.sentno,
						int(mention.head.get('begin'))] + 1),
					mention.head.get('neclass', '-'),
					mention.features['person'] or '-',
					mention.head.get('quotelabel'),  # is inside a quotation?
					'\t'.join(
						'-' if mention.features[key] is None
						else str(mention.features[key])
						for key in ('gender', 'human', 'number')),
					str(mention.clusterid),
					' '.join(mention.tokens).replace('\t', ' '),
					)), file=out)
	with open(prefix + '.links.tsv', 'w') as out:
		print('mention1\tmention2\tsieve', file=out)
		for mention in mentions:
			if mention.antecedent is not None:
				print('%d\t%d\t%s' % (mentions[mention.antecedent].id,
						mention.id, mention.sieve), file=out)
	with open(prefix + '.quotes.tsv', 'w') as out:
		print('id\tstart\tend\tsentno\tparno\tsentbounds\tmentions'
				'\tspeakermention\taddresseemention'
				'\tspeakercluster\taddresseecluster\ttext', file=out)
		for n, quotation in enumerate(quotations):
			print('\t'.join(str(a) for a in (
					n, quotation.start + 1, quotation.end,
					quotation.sentno, quotation.parno,
					int(quotation.sentbounds),
					','.join(str(m.id) for m in quotation.mentions),
					'-' if quotation.speaker is None
						else quotation.speaker.id,
					'-' if quotation.addressee is None
						else quotation.addressee.id,
					'-' if quotation.speaker is None
						else quotation.speaker.clusterid,
					'-' if quotation.addressee is None
						else quotation.addressee.clusterid,
					quotation.text.replace('\t', ' '))), file=out)
	with open(prefix + '.icarus', 'w') as out:
		icarusallocation(mentions, clusters, docname, part, file=out)


def icarusallocation(mentions, clusters, docname='-', part=0, file=sys.stdout):
	"""Write mention and link info in ICARUS allocation format.

	Cf. https://wiki.ims.uni-stuttgart.de/extern/ICARUS-Coreference-Perspective
	In ICARUS, load the gold conll file as a document set;
	choose "Add allocation", enter this .icarus file and pick "Default" Reader
	(not "CoNLL 2012 allocation"!);
	add gold conll file as allocation; pick CoNLL 2012 allocation reader."""
	# NB: in this file, spans are represented in the format expected by ICARUS.
	file.write('#begin document (%s); part %03d\n' % (docname, part))
	print('#begin nodes', file=file)
	print('ROOT', file=file)
	for mention in mentions:
		# ICARUS uses this format for spans: sentno (0-indexed),
		# start token index (1-indexed), end token index (1-indexed, inclusive)
		print('%d-%d-%d\t%s;%s' % (
				mention.sentno, mention.begin + 1, mention.end,
				'type:%s;head:%s;neclass:%s;quote:%s' % (
					mention.type,
					int(mention.head.get('begin')) + 1,
					mention.head.get('neclass'),
					mention.head.get('quotelabel')),
				';'.join('%s:%s' % (name, val)
					for name, val in mention.features.items())),
				file=file)
	print('#end nodes', file=file)
	print('#begin edges', file=file)
	for mention in mentions:
		if mention.antecedent is None:
			print('ROOT>>%d-%d-%d\ttype:IDENT;sieve:%s' % (
					mention.sentno, mention.begin + 1, mention.end,
					mention.sieve), file=file)
	for mention in mentions:
		if mention.antecedent is not None:
			n = min(clusters[mention.clusterid])
			print('%d-%d-%d>>%d-%d-%d\ttype:IDENT;sieve:%s' % (
					mentions[n].sentno, mentions[n].begin + 1, mentions[n].end,
					mention.sentno, mention.begin + 1, mention.end,
					mention.sieve), file=file)
	print('#end edges', file=file)
	print('#end document', file=file)


def htmlvis(trees, mentions, clusters, quotations):
	"""Visualize coreference in HTML document."""
	output = []
	sentences = [[a.get('word') for a
			in sorted(tree.iterfind('.//node[@word]'),
				key=lambda x: int(x.get('begin')))]
			for _, tree in trees]
	sentids = [(parno, sentno) for (parno, sentno), _ in trees]
	for mention in sortmentions(mentions):
		if mention.filter:
			continue
		cls = ('c%d' % mention.clusterid
				if len(clusters[mention.clusterid]) > 1 else 'n')
		sentences[mention.sentno][mention.begin] = (
				'<span id="m%d" class="%s" title="%d %d %s">[%s' % (
					mention.id, cls, mention.sentno, mention.begin,
					mention.featrepr(extended=True),
					sentences[mention.sentno][mention.begin]))
		sentences[mention.sentno][mention.end - 1] += ']</span>'
	qstarts = {q.start: n for n, q in enumerate(quotations)}
	qends = {q.end - 1 for q in quotations}
	try:
		from discodop.tree import DrawTree
		from discodop.treebank import alpinotree
		from discodop.punctuation import applypunct
		import xml.etree.ElementTree as ElementTree
		drawtrees = True
	except ImportError:
		drawtrees = False
	dt = ''
	for (parno, sentno), tree in trees:
		xml = etree.tostring(tree, encoding='utf8', pretty_print=True)
		if drawtrees:
			# discodop expects ElementTree instead of lxml tree
			item = alpinotree(
					ElementTree.fromstring(xml),
					functions='add', morphology='no')
			applypunct('move', item.tree, item.sent)
			dt = DrawTree(item.tree, item.sent).text(
						unicodelines=True, html=True, funcsep='-')
		output.append('<div id=t%d-%d style="display: none; ">'
				'<pre style="white-space: pre-wrap;">%s</pre>'
				'<pre>%s</pre></div>' % (
				parno, sentno,
				escape(xml.decode('utf8')),  # FIXME: highlight syntax?
				dt))
	doctokenid = 0
	quotation = None
	att = ''
	output.append('<div class=main>\n')
	for ((parno, sentno), sent) in zip(sentids, sentences):
		if parno == 1 and sentno == 1:
			output.append('<p>')
		elif sentno == 1:
			output.append('</p>\n<p>')
		output.append('<span class=n onClick="toggle(\'t%d-%d\')">' % (
				parno, sentno))
		if quotation is not None:  # quotation spanning multiple sents
			output.append('<span class=q%s>' % att)
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
				output.append('<span class=q%s>' % att)
			output.append(' ' + token)
			if doctokenid in qends:
				output.append('</span>')
				quotation = None
			doctokenid += 1
		if quotation is not None:  # quotation spanning multiple sents
			output.append('</span>')
		output.append('</span>\n')
	output.append('\n</p></div>\n')
	debugoutput = ''
	if VERBOSE:
		conv = ansi2html.Ansi2HTMLConverter(scheme='xterm', dark_bg=True)
		debugoutput = conv.convert(DEBUGFILE.getvalue(), full=False)
	return ''.join(output), debugoutput


def getunivdeps(filenames, trees):
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


def readconll(conllfile, docname='-'):
	"""Read conll data as list of lists: conlldata[sentno][tokenno][col].

	If multiple "#begin document docname" lines are found,
	only return chunks with matching docname; otherwise, return all chunks.
	"""
	conlldata = [[]]
	with open(conllfile) as inp:
		if inp.read().count('#begin document') == 1:
			docname = '-'
		inp.seek(0)
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
						conlldata[-1].append(line.strip().split())
					else:
						conlldata.append([])
				break
			elif line == '':
				break
	if not conlldata[-1]:  # remove empty sentence if applicable
		conlldata.pop()
	if not conlldata[0]:
		raise ValueError('Could not read gold data from %r with docname %r' % (
				conllfile, docname))
	return conlldata


def compare(conlldata, trees, mentions, clusters, out=sys.stdout):
	"""Visualize mentions and links wrt conll file."""
	goldspansforcluster = conllclusterdict(conlldata)
	respspansforcluster = respclusterdict(mentions, clusters)
	goldspans = {span for spans in goldspansforcluster.values()
			for span in spans}
	respspans = {(mention.sentno, mention.begin, mention.end,
			' '.join(mention.tokens))
			for mention in mentions if not mention.filter}
	comparementions(conlldata, trees, mentions,
			goldspans, respspans, out=out)
	comparecoref(conlldata, mentions, clusters, goldspans, respspans,
			goldspansforcluster, respspansforcluster, out=out)


def comparementions(conlldata, trees, mentions, goldspans, respspans,
		out=sys.stdout):
	"""Human-readable printing of a comparison between the output of the
	mention detection sieve and the 'gold' standard. Green brackets are
	correct, yellow brackets are mention boundaries only found in the gold
	standard, and red brackets are only found in our output."""
	sentences = [[a.get('word') for a in
			sorted(tree.iterfind('.//node[@word]'),
				key=lambda x: int(x.get('begin')))]
			for _, tree in trees]
	print(color('mentions in gold missing from response:', 'yellow'), file=out)
	for sentno, begin, end, text in sorted(goldspans - respspans):
		print('%3d %2d %2d %s' % (sentno, begin, end, text), file=out)
	if len(goldspans - respspans) == 0:
		print('(none)')
	print('\n' + color('mentions in response but not in gold:', 'yellow'),
			file=out)
	for sentno, begin, end, text in sorted(respspans - goldspans):
		print('%3d %2d %2d %s' % (sentno, begin, end, text), file=out)
	if len(respspans - goldspans) == 0:
		print('(none)')
	print('', file=out)
	#
	mentionbegin = defaultdict(int)
	mentionend = defaultdict(int)
	for mention in mentions:
		if not mention.filter:
			mentionbegin[mention.sentno, mention.begin] += 1
			mentionend[mention.sentno, mention.end - 1] += 1
	for sentno, sent in enumerate(sentences):
		out.write('%d: ' % sentno)
		# FIXME: ensure parentheses are well-nested
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


def comparecoref(conlldata, mentions, clusters, goldspans, respspans,
		goldspansforcluster, respspansforcluster, out=sys.stdout):
	"""List correct/incorrect coreference links.

	Assuming perfect mentions, gold and system coreference are partitions
	of a set of mentions into clusters. For each cluster identified by the
	system, there are three possibilities:
		- exact match with gold
		- completely disjoint with gold
		- partial match: links that should and should not be there.
	Imperfect mentions add an additional error type: incorrect link because
	one or two of the mentions should not have been a mention, or correctly
	identified mentions but should not be linked.
	"""
	def goldclustersforspan(sentno, begin, end):
		"""Look up span in conll file. Return set of cluster IDs X coreferent
		with the span; i.e., look for "(X)" or "(X" at begin index and "X)"
		at end index."""
		return {int(a.strip('()')) for a in
				conlldata[sentno][begin][-1].split('|')
				if (begin + 1 == end
					and a.startswith('(') and a.endswith(')'))
				or (a.startswith('(') and a[1:] + ')'
					in conlldata[sentno][end - 1][-1].split(
						'|'))}

	def correctlink(mention1, mention2):
		"""Return True if mention1 and mention2 are coreferent in gold data."""
		a = goldclustersforspan(mention1.sentno, mention1.begin, mention1.end)
		b = goldclustersforspan(mention2.sentno, mention2.begin, mention2.end)
		return a and b and not a.isdisjoint(b)

	print('\n' + color('coreference clusters:', 'yellow'), file=out)
	# take the first mention of cluster that is also a mention in gold
	for cluster in clusters:
		# skip clusters that are singleton in both response and gold
		if cluster is None or (
				len(cluster) == 1 and all(
					len(goldspansforcluster[cid]) == 1
					for m in cluster
						for cid in goldclustersforspan(mentions[m].sentno,
							mentions[m].begin, mentions[m].end))):
			continue
		cand = sorted(n for n in cluster if
				(mentions[n].sentno, mentions[n].begin, mentions[n].end,
					' '.join(mentions[n].tokens))
				in goldspans and not mentions[n].filter)
		n = cand[0] if cand else min(cluster)
		correctmention = ((mentions[n].sentno, mentions[n].begin,
				mentions[n].end, ' '.join(mentions[n].tokens)) in goldspans)
		if correctmention:
			c = 'yellow' if mentions[n].filter else 'green'
		elif mentions[n].filter:
			continue
		else:
			c = 'red'
		print(mentions[n].sentno, mentions[n].begin,
				color('[', c) + ' '.join(mentions[n].tokens) + color(']', c),
				file=out)
		for m in sorted(cluster - {n}):
			correct = correctlink(mentions[n], mentions[m])
			correctmention = ((mentions[m].sentno, mentions[m].begin,
					mentions[m].end) in goldspans)
			print('\t',
					color('<-->', 'green' if correct else 'red'),
					mentions[m].sentno, mentions[m].begin,
					color('[', 'green' if correctmention else 'red')
					+ ' '.join(mentions[m].tokens)
					+ color(']', 'green' if correctmention else 'red'),
					file=out)
		# look up missed gold links and print as 'yellow'
		for cid in goldclustersforspan(mentions[n].sentno, mentions[n].begin,
				mentions[n].end):
			for span in goldspansforcluster[cid]:
				if (span != (mentions[n].sentno, mentions[n].begin,
						mentions[n].end, ' '.join(mentions[n].tokens))
						and span not in respspansforcluster[
							mentions[n].clusterid]):
					sentno, begin, _end, text = span
					print('\t',
							color('<-->', 'yellow'),
							sentno, begin,
							color('[', 'green'
								if span in respspans else 'yellow')
							+ text
							+ color(']', 'green'
								if span in respspans else 'yellow'),
							file=out)


def extractmentionsfromconll(conlldata, trees, ngdata, gadata):
	"""Extract gold mentions from annotated data."""
	mentions = []
	goldspansforcluster = conllclusterdict(conlldata)
	goldspans = {span for spans in goldspansforcluster.values()
			for span in spans}
	for sentno, begin, end, text in sorted(goldspans):
		# smallest node spanning begin, end
		tree = trees[sentno][1]
		node = sorted((node for node in tree.findall('.//node')
					if begin >= int(node.get('begin'))
					and end <= int(node.get('end'))),
				key=lambda x: int(x.get('end')) - int(x.get('begin')))[0]
		headidx = getheadidx(node)
		if headidx >= end:
			headidx = max(int(x.get('begin')) for x in node.findall('.//node')
					if int(x.get('begin')) < end)
		# NB: no paragraph number
		mentions.append(Mention(
				len(mentions), sentno, 0, tree, node, begin, end, headidx,
				text.split(' '), ngdata, gadata))
	return mentions


def conllclusterdict(conlldata):
	"""Extract dict from CoNLL file mapping gold cluster IDs to spans."""
	spansforcluster = {}
	spans = {}
	lineno = 1
	for sentno, chunk in enumerate(conlldata):
		scratch = {}
		for idx, fields in enumerate(chunk):
			lineno += 1
			labels = fields[-1]
			for a in labels.split('|'):
				if a == '-' or a == '_':
					continue
				try:
					clusterid = int(a.strip('()'))
				except ValueError:
					raise ValueError('Cannot parse cluster %r at line %d'
							% (a.strip('()'), lineno))
				if a.startswith('('):
					scratch.setdefault(clusterid, []).append(
							(sentno, idx, lineno))
				if a.endswith(')'):
					try:
						sentno, begin, _ = scratch[int(a.strip('()'))].pop()
					except KeyError:
						raise ValueError(
								'No opening paren for cluster %s at line %d'
								% (a.strip('()'), lineno))
					text = ' '.join(line[3] for line in chunk[begin:idx + 1])
					span = (sentno, begin, idx + 1, text)
					if span in spans:
						debug('Warning: duplicate span %r '
								'in cluster %d and %d, sent %d, line %d'
								% (span[3], clusterid, spans[span],
									sentno + 1, lineno))
					spans[span] = clusterid
					spansforcluster.setdefault(clusterid, set()).add(span)
		lineno += 1
		for a, b in scratch.items():
			if b:
				raise ValueError('Unclosed paren for cluster %d at line %d'
						% (a, b[0][2]))
	return spansforcluster


def respclusterdict(mentions, clusters):
	"""Return dict that maps system cluster ID to set of coreferent spans."""
	spansforcluster = {}
	for n, cluster in enumerate(clusters):
		if cluster is None:
			continue
		for m in cluster:
			spansforcluster.setdefault(n, set())
			if not mentions[m].filter:
				spansforcluster[n].add((mentions[m].sentno,
						mentions[m].begin, mentions[m].end,
						' '.join(mentions[m].tokens)))
	return spansforcluster


def setverbose(verbose, debugfile):
	"""Set global verbosity variables.

	:param verbose: whether to print messages
	:param debugfile: file to redirect messages to."""
	global VERBOSE, DEBUGFILE
	VERBOSE = verbose
	DEBUGFILE = debugfile


def debug(*args, **kwargs):
	"""Print debug information if global variable VERBOSE is True;
	send output to file (or stdout) DEBUGFILE."""
	if VERBOSE:
		print(*args, **kwargs, file=DEBUGFILE)


def color(text, c):
	"""Returns colored text."""
	if c == 'red':
		return colorama.Fore.RED + text + colorama.Fore.RESET
	elif c == 'green':
		return colorama.Fore.GREEN + text + colorama.Fore.RESET
	elif c == 'yellow':
		return colorama.Fore.YELLOW + text + colorama.Fore.RESET
	raise ValueError


def postprocess(exclude, mentions, clusters, goldmentions):
	"""Filter certain mentions/links."""
	tests = {
			'singletons': lambda m: len(clusters[m.clusterid]) == 1,
			'npsingletons': lambda m: (len(clusters[m.clusterid]) == 1
				and m.type != 'name'),
			'relpronouns': lambda m: m.node.get('vwtype') == 'betr',
			'reflectives': lambda m: m.node.get('vwtype') == 'refl',
			'reciprocals': lambda m: m.node.get('vwtype') == 'recip',
			'appositives': lambda m: m.node.get('rel') == 'app',
			'predicatives': lambda m: m.node.get('rel') == 'predc',
			}
	for kind in exclude:
		if kind not in tests:
			raise ValueError('unrecognized --exclude argument: %s' % kind)
	if exclude:
		for mention in mentions:
			if any(tests[kind](mention) for kind in exclude):
				if goldmentions:
					# Remove links but keep gold mention
					if mention.clusterid != mention.id:
						clusters[mention.clusterid].remove(mention.id)
						mention.clusterid = mention.id
						clusters[mention.id] = {mention.id}
				else:  # Mark as filtered; removal would change mention IDs
					mention.filter = True


def process(path, output, ngdata, gadata,
		docname='-', part=0, conllfile=None, fmt=None,
		start=None, end=None, startcluster=0,
		goldmentions=False, exclude=(), outputprefix=None):
	"""Process a single directory with Alpino XML parses."""
	if os.path.isdir(path):
		path = os.path.join(path, '*.xml')
	if fmt == 'html':
		setverbose(VERBOSE, io.StringIO())
	debug('processing:', path)
	filenames = sorted(glob(path), key=parsesentid)[start:end]
	trees = [(parsesentid(filename), etree.parse(filename))
			for filename in filenames]
	if conllfile is not None:
		conlldata = readconll(conllfile, docname)[start:end]
	mentions = None
	if goldmentions:
		mentions = extractmentionsfromconll(conlldata, trees, ngdata, gadata)
	mentions, clusters, quotations, idx = resolvecoreference(
			trees, ngdata, gadata, mentions)
	postprocess(exclude, mentions, clusters, goldmentions)
	if conllfile is not None and VERBOSE:
		debug(color('evaluating against:', 'yellow'), conllfile, docname)
		compare(conlldata, trees, mentions, clusters, out=DEBUGFILE)
	if fmt == 'booknlp':
		getunivdeps(filenames, trees)
	if fmt == 'html':
		corefresults, debugoutput = htmlvis(
				trees, mentions, clusters, quotations)
		with open('templates/results.html') as inp:
			template = Template(inp.read())
		print(template.render(docname=docname, corefresults=corefresults,
					debugoutput=debugoutput),
				file=output)
	elif not VERBOSE:
		writetabular(trees, mentions, docname=docname, part=part,
				file=output, fmt=fmt, startcluster=startcluster)
	if outputprefix is not None:
		writeinfo(mentions, clusters, quotations, idx, outputprefix, docname)
	return len(clusters)


def clindev(ngdata, gadata, goldmentions):
	"""Run on CLIN26 shared task dev data and evaluate."""
	timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
	path = os.path.join('results/clindev/', timestamp)
	os.makedirs(path, exist_ok=False)
	for conllfile in glob('../groref/clinDevData/*.coref_ne'):
		dirname = os.path.join(
				os.path.dirname(conllfile),
				os.path.basename(conllfile).split('_')[0])
		docname = os.path.basename(conllfile)
		with open(os.path.join(path, docname), 'w') as out:
			process(dirname + '/*.xml', out, ngdata, gadata,
					docname=docname, part=None, conllfile=conllfile,
					goldmentions=goldmentions, start=0, end=6)
			# shared task says the first 7 sentences are annotated,
			# but in many documents only the first 6 sentences are annotated.
	with open('%s/blanc_scores' % path, 'w') as out:
		os.chdir('../groref/clin26-eval-master')
		subprocess.call(
				['bash', 'score_coref.sh',
					'coref_ne', 'dev_corpora/coref_ne',
					'../../dutchcoref/' + path, 'blanc'],
				stdout=out)
	os.chdir('../../dutchcoref')
	with open('%s/blanc_scores' % path) as inp:
		print(inp.read())


def semeval(ngdata, gadata, goldmentions):
	"""Run on semeval 2010 shared task dev data and evaluate."""
	timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
	path = os.path.join('results/semevaldev/', timestamp)
	os.makedirs(path, exist_ok=False)
	startcluster = 0
	with open(os.path.join(path, 'result.conll'), 'w') as out:
		for dirname in sorted(glob('data/semeval2010NLdevparses/*/'),
				key=lambda x: int(x.rstrip('/').split('_')[1])):
			docname = os.path.basename(dirname.rstrip('/'))
			startcluster += process(dirname, out, ngdata, gadata,
					fmt='semeval2010', docname=docname,
					conllfile='data/semeval2010/task01.posttask.v1.0/'
						'corpora/training/nl.devel.txt.fixed',
					startcluster=startcluster, goldmentions=goldmentions,
					exclude=('relpronouns', 'reflectives', 'reciprocals',
						'predicatives', 'appositives', 'npsingletons'))
	with open('%s/blanc_scores' % path, 'w') as out:
		subprocess.call([
				'../groref/conll_scorer/scorer.pl',
				'blanc',
				'data/semeval2010/task01.posttask.v1.0/'
					'corpora/training/nl.devel.txt.fixed',
				'%s/result.conll' % path],
				stdout=out)
	with open('%s/blanc_scores' % path) as inp:
		print(inp.read())


def runtests(ngdata, gadata):
	"""Some simple tests."""
	print('ref (each sentence should have a coreference link)')
	trees = [(parsesentid(filename), etree.parse(filename))
			for filename in sorted(glob('tests/ref/*.xml'), key=parsesentid)]
	for n, _ in enumerate(trees):
		mentions, clusters, _quotations, _idx = resolvecoreference(
				trees[n:n + 1], ngdata, gadata)
		print('%d. %s' % (n, ' '.join(gettokens(trees[n][1], 0, 999))))
		for m, mention in enumerate(mentions):
			print(m, mention)
		print(clusters)
		if not any(len(a) > 1 for a in clusters if a is not None):
			raise ValueError

	print('\nnonref (no sentence should have any coreference link)')
	trees = [(parsesentid(filename), etree.parse(filename))
			for filename in sorted(glob('tests/nonref/*.xml'), key=parsesentid)]
	for n, _ in enumerate(trees):
		mentions, clusters, _quotations, _idx = resolvecoreference(
				trees[n:n + 1], ngdata, gadata)
		print('%d. %s' % (n, ' '.join(gettokens(trees[n][1], 0, 999))))
		for m, mention in enumerate(mentions):
			print(m, mention)
		print(clusters)
		if not all(len(a) == 1 for a in clusters if a is not None):
			raise ValueError

	print('\nnomention (no sentence should have any mention)')
	trees = [(parsesentid(filename), etree.parse(filename))
			for filename in sorted(glob('tests/nomention/*.xml'),
				key=parsesentid)]
	for n, _ in enumerate(trees):
		mentions, clusters, _quotations, _idx = resolvecoreference(
				trees[n:n + 1], ngdata, gadata)
		print('%d. %s [%d mentions]' % (
				n, ' '.join(gettokens(trees[n][1], 0, 999)), len(mentions)))
		for m, mention in enumerate(mentions):
			print(m, mention)
		if mentions:
			raise ValueError

	print('\nall tests passed')


def main():
	"""CLI"""
	longopts = ['fmt=', 'slice=', 'gold=', 'exclude=', 'outputprefix=',
			'help', 'verbose', 'test', 'clindev', 'semeval', 'goldmentions']
	try:
		opts, args = getopt.gnu_getopt(sys.argv[1:], '', longopts)
	except getopt.GetoptError:
		print(__doc__)
		return
	opts = dict(opts)
	if '--help' in opts:
		print(__doc__)
		return
	if '--verbose' in opts:
		setverbose(True, sys.stdout)
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
		if len(args) != 1:
			print(__doc__)
			return
		path = args[0]
		exclude = [a for a in opts.get('--exclude', '').split(',') if a]
		if '--outputprefix' in opts:
			ext = '.html' if opts.get('--fmt') == 'html' else '.conll'
			with open(opts['--outputprefix'] + ext, 'w') as out:
				process(path, out, ngdata, gadata,
						fmt=opts.get('--fmt'), start=start, end=end,
						docname=os.path.basename(path.rstrip('/')),
						conllfile=opts.get('--gold'),
						goldmentions='--goldmentions' in opts,
						outputprefix=opts.get('--outputprefix'),
						exclude=exclude)
		else:
			process(path, sys.stdout, ngdata, gadata,
					fmt=opts.get('--fmt'), start=start, end=end,
					docname=os.path.basename(path.rstrip('/')),
					conllfile=opts.get('--gold'),
					goldmentions='--goldmentions' in opts,
					exclude=exclude)


if __name__ == '__main__':
	main()
