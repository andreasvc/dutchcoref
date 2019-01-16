Dutch coreference resolution & dialogue analysis using deterministic rules
==========================================================================

Usage
-----
```
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
```

Datasets
--------

Dutch first names
^^^^^^^^^^^^^^^^^
download `Top_eerste_voornamen_NL_2010.csv`
from https://www.meertens.knaw.nl/nvb/
and put it in the `data/` directory.

CLIN26 shared task data, number & gender data from Web text
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Clone this repository under same parent folder as this repository::

    ~/code/dutchcoref $ cd ..
    ~/code $ git clone https://bitbucket.org/robvanderg/groref.git

SemEval 2010 shared task data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Download the data from http://www.lsi.upc.edu/~esapena/downloads/index.php?id=1
Apply the Unicode fix in `fixsemeval2010.sh`.
The directory `data/semeval2010NLdevparses` contains Alpino parses for the
Dutch development set of this task.


References
----------
This code base is a Dutch implementation of the Stanford Multi-Pass Coreference System:

Heeyoung Lee, Angel Chang, Yves Peirsman, Nathanael Chambers, Mihai Surdeanu, and Dan Jurafsky. Deterministic coreference resolution based on entity-centric, precision-ranked rules. Computational Linguistics, 39 (4):885–916, 2013. http://aclweb.org/anthology/J/J13/J13-4004.pdf

See also these previous implementations
https://bitbucket.org/robvanderg/groref
and https://github.com/antske/coref_draft
