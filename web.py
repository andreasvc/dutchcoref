"""Web interface for coreference system."""
import io
import sys
import json
import time
import logging
import requests
from lxml import etree
from flask import Flask
from flask import request, render_template
import coref

APP = Flask(__name__)
DEBUG = False
LIMIT = 5000  # maximum number of bytes of input to accept
STANDALONE = __name__ == '__main__'
ALPINOAPI = 'http://127.0.0.1:11300/json'


@APP.route('/')
@APP.route('/index')
def index():
	"""Start page where a text can be entered."""
	return render_template('index.html')


@APP.route('/coref', methods=('POST', ))
def results():
	"""Get coreference and show results."""
	if 'text' not in request.form:
		return 'No text supplied'
	text = request.form['text']
	if len(text) > LIMIT:
		return 'Too much text; limit: %d bytes' % LIMIT

	parses = parse(text)
	if parses is None:
		return 'Parsing failed!'
	trees = [(a, etree.parse(io.BytesIO(b))) for a, b in parses]
	coref.setverbose(True, io.StringIO())
	mentions, clusters, quotations = coref.resolvecoreference(
			trees, ngdata, gadata)
	output = io.StringIO()
	coref.writehtml(trees, mentions, clusters, quotations, file=output)
	return output.getvalue()
	# return render_template('results.html', data=output.getvalue())


def parse(text):
	"""Tokenize & parse text with Alpino API.

	Cf. https://github.com/rug-compling/alpino-api
	Expects alpino-api/demo/alpiner server to be running and accessible
	at URL ALPINOAPI."""
	def parselabel(label):
		"""
		>>> parselabel('doc.p.1.s.2')
		(1, 2)"""
		_doc, _p, a, _s, b = label.rsplit('.', 5)
		return int(a), int(b)

	command = {'request': 'parse', 'data_type': 'text', 'timeout': 60}
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
		# log.info(resp.content)
		parses.extend((
				parselabel(line['label']), line['alpino_ds'].encode('utf8'))
				for line in result['batch']
				if line['line_status'] == 'ok')
		if result['finished']:
			break
		if interval < maxinterval:
			interval += 1
	return sorted(parses)


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
				['port=', 'ip=', 'numproc=', 'debug'])
		opts = dict(opts)
	except GetoptError as err:
		print('error: %r' % err, file=sys.stderr)
		sys.exit(2)
	DEBUG = '--debug' in opts
if STANDALONE:
	APP.run(use_reloader=True,
			host=opts.get('--ip', '0.0.0.0'),
			port=int(opts.get('--port', 5005)),
			debug=DEBUG)
