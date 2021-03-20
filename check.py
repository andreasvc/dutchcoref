"""Load CoNLL 2012 files and report any errors or warnings."""
import sys
from coref import readconll, conllclusterdict, setverbose

setverbose(True, sys.stdout)
for filename in sys.argv[1:]:
	try:
		for docname, conlldata in readconll(filename).items():
			print('\n', filename, docname)
			conllclusterdict(conlldata)
	except Exception as err:
		print(err)
		print('NB: Not checking for further errors in this file.')
