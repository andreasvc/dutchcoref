"""Print basic stats for one or more CoNLL 2012 files.

Usage: conllstats.py [options] FILE...

Options:
    --help          this message
    --parses=<dir>  to enable mention type stats for Dutch data, specify a
                    directory of Alpino parses (containing directories with
					same basename as CoNLL files, containing XML parses)"""


def getstats(args, parsesdir=None):
	"""Print stats for a list of CoNLL 2012 files."""
	import os
	from glob import glob
	from lxml import etree
	import coref
	sents = tokens = nummentions = numentities = numlinks = 0
	pronouns = nominals = names = 0
	ngdata, gadata = coref.readngdata()
	for fname in args:
		data = []
		try:
			docs = coref.readconll(fname)
		except Exception as err:
			print('file:', fname)
			print(err)
			return
		for docname, data in docs.items():
			try:
				goldspansforcluster = coref.conllclusterdict(data)
			except Exception as err:
				print('file:', fname)
				print(err)
				return
			if parsesdir is not None:
				# given docname, read <parsesdir>/<docname>/*.xml
				if docname.startswith('('):
					docname = docname.split(';')[0].strip('()')
				path = os.path.join(parsesdir, docname, '*.xml')
				filenames = sorted(glob(path), key=coref.parsesentid)
				if len(data) != len(filenames):
					raise ValueError('filename: %s; document %s '
							'sentences in CoNLL (%d) '
							'and number of .xml parses (%d) not equal' %
							(fname, docname, len(data), len(filenames)))
				trees = [(coref.parsesentid(filename), etree.parse(filename))
						for filename in filenames]
				mentions = coref.extractmentionsfromconll(
						data, trees, ngdata, gadata)
				pronouns += sum(m.type == 'pronoun' for m in mentions)
				nominals += sum(m.type == 'noun' for m in mentions)
				names += sum(m.type == 'name' for m in mentions)
			sents += len(data)
			tokens += sum(len(sent) for sent in data)
			nummentions += len({span for spans in goldspansforcluster.values()
					for span in spans})
			numentities += len(goldspansforcluster)
			numlinks += sum(int((len(cluster) * (len(cluster) - 1)) / 2)
					for cluster in goldspansforcluster.values())
	print('sents:', sents)
	print('tokens:', tokens)
	print('mentions:', nummentions)
	print('entities:', numentities)
	print('links:', numlinks)
	print('tok/sent:', tokens / sents)
	print('mentions / tokens:', nummentions / tokens)
	print('entities / tokens:', numentities / tokens)
	print('links / tokens:', numlinks / tokens)
	print('mentions / entities:', nummentions / numentities)
	print('links / entities:', numlinks / numentities)
	if parsesdir is None:
		print('specify --parses to get % pronouns, nominals, names')
	else:
		print('% pronouns', 100 * pronouns / nummentions)
		print('% nominals', 100 * nominals / nummentions)
		print('% names', 100 * names / nummentions)


def main():
	"""CLI."""
	import sys
	import getopt
	longopts = ['help', 'parses=']
	try:
		opts, args = getopt.gnu_getopt(sys.argv[1:], '', longopts)
	except getopt.GetoptError:
		print(__doc__)
		return
	opts = dict(opts)
	if not args or '--help' in opts:
		print(__doc__)
		return
	getstats(args, opts.get('--parses'))


if __name__ == '__main__':
	main()
