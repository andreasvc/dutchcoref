"""Load CoNLL 2012 files and report any errors or warnings."""
import sys
from coref import readconll, conllclusterdict, setverbose

setverbose(True, sys.stdout)
for filename in sys.argv[1:]:
	print('\n', filename)
	try:
		conllclusterdict(readconll(filename))
	except Exception as err:
		print(err)
