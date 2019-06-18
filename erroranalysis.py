"""Coreference error analysis tool for CoNLL 2012 files.

Usage: python3 erroranalysis.py mentions gold.conll system.conll | less -R
Or: python3 erroranalysis.py links gold.conll system.conll | less -R

Options:
	--hidecorrectlinks    only show incorrect links
	--html                write output in HTML instead of with ANSI color codes
"""
import io
import sys
import getopt
import ansi2html
from coref import conllclusterdict, readconll, color


def compare(cmd, goldfile, respfile, hidecorrectlinks=False, out=sys.stdout):
	"""Compare mentions and links across CoNLL 2012 files."""
	gold = readconll(goldfile, '-')
	resp = readconll(respfile, '-')
	print('comparing gold file:', goldfile, file=out)
	print('against system output:', respfile, file=out)
	goldspansforcluster = conllclusterdict(gold)
	respspansforcluster = conllclusterdict(resp)
	goldspans = {span for spans in goldspansforcluster.values()
			for span in spans}
	respspans = {span for spans in respspansforcluster.values()
			for span in spans}
	if cmd == 'mentions':
		comparementions(gold, resp, goldspans, respspans, out=out)
	elif cmd == 'links':
		comparecoref(resp, goldspans, respspans, goldspansforcluster,
				respspansforcluster, hidecorrectlinks, out)
	else:
		raise ValueError('unknown cmd: %s' % cmd)


def comparementions(gold, resp, goldspans, respspans,
		out=sys.stdout):
	"""Human-readable printing of a comparison between the output of the
	mention detection sieve and the 'gold' standard. Green brackets are
	correct, yellow brackets are mention boundaries only found in the gold
	standard, and red brackets are only found in our output."""
	sentences = [[line[3] for line in sent] for sent in gold]
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
	for sentno, sent in enumerate(sentences):
		out.write('%d: ' % sentno)
		# FIXME: ensure parentheses are well-nested
		for idx, token in enumerate(sent):
			respopen = resp[sentno][idx][-1].count('(')
			respclose = resp[sentno][idx][-1].count(')')
			goldopen = gold[sentno][idx][-1].count('(')
			goldclose = gold[sentno][idx][-1].count(')')
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


def comparecoref(resp, goldspans, respspans,
		goldspansforcluster, respspansforcluster,
		hidecorrectlinks=False, out=sys.stdout):
	"""List correct/incorrect coreference links.

	Assuming perfect mentions, gold and system coreference are partitions
	of a set of mentions into clusters. For each cluster identified by the
	system, there are three possibilities:
		- exact match with gold
		- completely disjoint with gold
		- partial match: links that should and should not be there.
	Imperfect mentions add an additional error type: incorrect link because
	one or two of the mentions should not have been a mention.
	"""
	def clustersforspan(conlldata, span):
		"""Look up span in conll file. Return set of cluster IDs X coreferent
		with the span; i.e., look for "(X)" or "(X" at begin index and "X)"
		at end index."""
		sentno, begin, end, _text = span
		return {int(a.strip('()')) for a in
				conlldata[sentno][begin][-1].split('|')
				if (begin + 1 == end
					and a.startswith('(') and a.endswith(')'))
				or (a.startswith('(') and a[1:] + ')'
					in conlldata[sentno][end - 1][-1].split(
						'|'))}

	def correctlink(span1, span2):
		"""Return True if mention1 and mention2 are coreferent in response."""
		a = clustersforspan(resp, span1)
		b = clustersforspan(resp, span2)
		return a and b and not a.isdisjoint(b)

	print('\n' + color('coreference clusters:', 'yellow'), file=out)
	# iterate over clusters in gold, report first mention
	for clusterid, spans in goldspansforcluster.items():
		# skip clusters that are singleton in both response and gold
		if len(spans) == 1 and all(
				len(respspansforcluster[cid]) == 1
				for span in spans
					for cid in clustersforspan(resp, span)):
			continue
		# select a span to be representative of this cluster
		# cand = sorted(span for span in spans if span in respspans)
		# span = cand[0] if cand else min(spans)
		span = min(spans)
		correctmention = span in respspans
		c = 'green' if correctmention else 'yellow'
		sentno, begin, _end, text = span
		print('%d %d %s (%d)' % (
				sentno, begin,
				color('[', c) + text + color(']', c),
				clusterid), file=out)
		result = []
		# collect links in gold and print as green or yellow
		for otherspan in spans:
			if span == otherspan:
				continue
			correct = correctlink(span, otherspan)
			correctmention = otherspan in respspans
			sentno, begin, _end, text = otherspan
			if correct and hidecorrectlinks:
				continue
			result.append((
					sentno, begin,
					color('<-->', 'green' if correct else 'yellow'),
					color('[', 'green' if correctmention else 'yellow')
					+ text
					+ color(']', 'green' if correctmention else 'yellow')))
		# look up wrong response links and print as 'red'
		for cid in sorted(clustersforspan(resp, span)):
			for otherspan in respspansforcluster[cid]:
				if span == otherspan or otherspan in spans:
					continue
				sentno, begin, _end, text = otherspan
				result.append((
						sentno, begin,
						color('<-->', 'red'),
						color('[', 'green'
							if otherspan in goldspans else 'red')
						+ text
						+ color(']', 'green'
							if otherspan in goldspans else 'red')))
		for sentno, begin, link, mention in sorted(result):
			print('\t', link, sentno, begin, mention, file=out)


def main():
	"""CLI."""
	try:
		opts, args = getopt.gnu_getopt(
				sys.argv[1:], '', ['hidecorrectlinks', 'html'])
		opts = dict(opts)
		cmd, goldfile, respfile = args
		if cmd not in ('mentions', 'links'):
			raise ValueError('unknown cmd: %s' % cmd)
	except (ValueError, getopt.GetoptError) as err:
		print(err)
		print(__doc__)
		return
	hidecorrectlinks = '--hidecorrectlinks' in opts
	if '--html' in opts:
		out = io.StringIO()
		compare(cmd, goldfile, respfile,
				hidecorrectlinks=hidecorrectlinks, out=out)
		conv = ansi2html.Ansi2HTMLConverter(scheme='xterm', dark_bg=True)
		print(conv.convert(out.getvalue()))
	else:
		compare(cmd, goldfile, respfile, hidecorrectlinks=hidecorrectlinks)


if __name__ == '__main__':
	main()
