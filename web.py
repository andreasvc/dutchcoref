"""Web interface for coreference system."""
import io
import re
import sys
import json
import time
import logging
import requests
from lxml import etree
from flask import Flask, Markup
from flask import request, render_template, redirect, url_for
import coref

APP = Flask(__name__)
DEBUG = False
LIMIT = 5000  # maximum number of bytes of input to accept
STANDALONE = __name__ == '__main__'
ALPINOAPI = 'http://127.0.0.1:11200/json'


@APP.route('/')
@APP.route('/index')
def index():
	"""Start page where a text can be entered."""
	return render_template('index.html', limit=LIMIT)


@APP.route('/coref', methods=('POST', 'GET'))
def results():
	"""Get coreference and show results."""
	if request.method == 'GET':
		return redirect(url_for('index'))
	if 'text' not in request.form:
		return 'No text supplied'
	text = request.form['text']
	if len(text) > LIMIT:
		return 'Too much text; limit: %d bytes' % LIMIT

	parses = parse(simplifyunicodespacepunct(text))
	if parses is None:
		return 'Parsing failed!'
	trees = [(a, etree.parse(io.BytesIO(b))) for a, b in parses]
	coref.setverbose(True, io.StringIO())
	mentions, clusters, quotations, _idx = coref.resolvecoreference(
			trees, ngdata, gadata)
	corefhtml, coreftabular, debugoutput = coref.htmlvis(
			trees, mentions, clusters, quotations,
			parses=True, coreffmt='minimal')
	return render_template('results.html', docname='',
			corefhtml=Markup(corefhtml),
			coreftabular=coreftabular,
			debugoutput=Markup(debugoutput),
			parses=True)


def parse(text):
	"""Tokenize & parse text with Alpino API.

	Cf. https://github.com/andreasvc/alpino-api
	Expects alpino-api/demo/alpiner server to be running and accessible
	at URL ALPINOAPI."""
	def parselabel(label):
		"""
		>>> parselabel('doc.p.1.s.2')
		(1, 2)"""
		_doc, _p, a, _s, b = label.rsplit('.', 5)
		return int(a), int(b)

	command = {'request': 'parse', 'data_type': 'text', 'timeout': 60,
			'ud': False}
	data = '%s\n%s' % (json.dumps(command), text)
	log.info('submitting parse')
	resp = requests.post(ALPINOAPI, data=data.encode('utf8'))
	result = resp.json()
	if result['code'] != 202:
		log.error(resp.content)
		return None
	# log.info(resp.content)
	rid = result['id']
	interval = 1
	maxinterval = result['interval']
	parses = []
	while True:
		time.sleep(interval)
		log.info('waited %d seconds; checking results', interval)
		resp = requests.post(ALPINOAPI, json={'request': 'output', 'id': rid})
		result = resp.json()
		if result['code'] != 200:
			log.error(resp.content)
			return None
		log.info('got %d results; code=%s status=%s finished=%s',
				len(result['batch']), result['code'],
				result['status'], result['finished'])
		# log.info(resp.content.decode('utf8'))
		parses.extend((
				parselabel(line['label']), line['alpino_ds'].encode('utf8'))
				for line in result['batch']
				if line['line_status'] == 'ok')
		if result['finished']:
			break
		if interval < maxinterval:
			interval += 1
	return sorted(parses)


def simplifyunicodespacepunct(text):
	"""Turn various unicode whitespace and punctuation characters into simple
	ASCII equivalents where appropriate, and discard control characters.

	NB: this discards some information (e.g., left vs right quotes, dash vs
	hyphens), but given that such information is not consistently encoded
	across languages and texts, it is more reliable to normalize to a common
	denominator.

	>>> simplifyunicodespacepunct('‘De verraders’, riep de sjah.')
	"'De verraders', riep de sjah."
	"""
	# Some exotic control codes not handled:
	# U+0085 NEL: Next Line
	# U+2028 LINE SEPARATOR
	# U+2029 PARAGRAPH SEPARATOR

	# Normalize spaces
	# U+00A0 NO-BREAK SPACE
	# U+2000 EN QUAD
	# U+2001 EM QUAD
	# U+2002 EN SPACE
	# U+2003 EM SPACE
	# U+2004 THREE-PER-EM SPACE
	# U+2005 FOUR-PER-EM SPACE
	# U+2006 SIX-PER-EM SPACE
	# U+2007 FIGURE SPACE
	# U+2008 PUNCTUATION SPACE
	# U+2009 THIN SPACE
	# U+200A HAIR SPACE
	text = re.sub('[\u00a0\u2000-\u200a]', ' ', text)

	# remove discretionary hyphen, soft space
	# special case: treat soft hyphen at end of line as a regular hyphen,
	# to ensure that it will be dehyphenated properly.
	text = re.sub('\u00ad+\n', '-\n', text)
	#      8 BACKSPACE
	# U+00AD SOFT HYPHEN
	# U+200B ZERO WIDTH SPACE
	# U+2027 HYPHENATION POINT
	text = re.sub('[\b\u00ad\u200b\u2027]', '', text)

	# hyphens
	# U+00B7 MIDDLE DOT
	# U+2010 HYPHEN
	# U+2011 NON-BREAKING HYPHEN
	# U+2212 MINUS SIGN
	text = re.sub('[\u00b7\u2010\u2011\u2212]', '-', text)
	# dashes/bullet points
	# U+2012 FIGURE DASH
	# U+2013 EN DASH
	# U+2014 EM DASH
	# U+2015 HORIZONTAL BAR
	# U+2022 BULLET
	# U+2043 HYPHEN BULLET
	text = re.sub('[\u2012-\u2015\u2022\u2043]', ' - ', text)

	# U+2044 FRACTION SLASH
	# U+2215 DIVISION SLASH
	text = text.replace('[\u2044\u2215]', '/')  # e.g., 'he/she'

	# single quotes:
	# U+2018 left single quotation mark
	# U+2019 right single quotation mark
	# U+201A single low-9 quotation mark
	# U+201B single high-reversed-9 quotation mark
	# U+2039 single left-pointing angle quotation mark
	# U+203A single right-pointing angle quotation mark
	# U+02BC modifier letter apostrophe
	text = re.sub('[\u2018-\u201b\u2039\u203a\u02bc]', "'", text)

	# double quotes:
	# U+201C left double quotation mark
	# U+201D right double quotation mark
	# U+201E double low-9 quotation mark
	# U+201F double high-reversed-9 quotation mark
	# U+00AB left-pointing double angle quotation mark
	# U+00BB right-pointing double angle quotation mark
	text = re.sub("[\u201c-\u201f\u00ab\u00bb]|''", '"', text)

	return text


logging.basicConfig()
log = logging.getLogger()
log.setLevel(logging.DEBUG)
log.handlers[0].setFormatter(logging.Formatter(
		fmt='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
log.info('loading.')
ngdata, gadata = coref.readngdata()
log.info('done.')
if STANDALONE:
	from getopt import gnu_getopt, GetoptError
	try:
		opts, _args = gnu_getopt(sys.argv[1:], '',
				['port=', 'ip=', 'alpinoapi=', 'debug'])
		opts = dict(opts)
	except GetoptError as err:
		print('error: %r' % err, file=sys.stderr)
		sys.exit(2)
	DEBUG = '--debug' in opts
	if '--alpinoapi' in opts:
		ALPINOAPI = opts['--alpinoapi']
	APP.run(use_reloader=True,
			host=opts.get('--ip', '0.0.0.0'),
			port=int(opts.get('--port', 5005)),
			debug=DEBUG)
