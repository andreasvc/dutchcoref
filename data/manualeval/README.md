A manual evaluation of Corea web service and dutchcoref
=======================================================

A version of the Corea coreference system is available as a web service.
The following is a manual evaluation of a single document: the first document
of the SemEval 2010 Dutch development set.

dutchcoref output produced with:

	$ python3 coref.py --semeval=dev

Evaluation tool: https://github.com/andreasvc/coval

Files
-----

    knack_file7.txt                     Detokenized input, from SemEval 2010 data
    knack_file7.coreawebservice.xml     Output from http://corea.tst-centrale.org/
    knack_file7.coreawebservice.conll   Manual conversion of XML to tabular format
    knack_file7.dutchcoref.conll        dutchcoref output for same file, tabular format
    knack_file7.gold.conll              Gold annotations, from SemEval 2010 data

Results
-------

    $ python3 ~/code/coval/scorer.py knack_file7.gold.conll knack_file7.coreawebservice.conll
                 recall  precision         F1
    mentions      50.00      58.82      54.05
    muc           50.00      38.46      43.48
    bcub          40.00      32.63      35.94
    ceafe         18.36      45.90      26.23
    ceafm         40.00      47.06      43.24
    lea           25.00      25.49      25.24
    CoNLL score:  35.22

    $ python3 ~/code/coval/scorer.py knack_file7.gold.conll knack_file7.dutchcoref.conll
                 recall  precision         F1
    mentions      70.00      87.50      77.78
    muc           70.00      77.78      73.68
    bcub          60.00      80.21      68.65
    ceafe         60.09      85.84      70.70
    ceafm         70.00      87.50      77.78
    lea           55.00      75.00      63.46
    CoNLL score:  71.01

Alternatively, evaluation with singletons removed:

	$ python3 ~/code/coval/scorer.py knack_file7.gold.conll knack_file7.coreawebservice.conll remove_singletons
	5 and 0 singletons are removed from the key and system files, respectively
				 recall  precision         F1
	mentions      53.33      47.06      50.00
	muc           50.00      38.46      43.48
	bcub          40.00      30.32      34.50
	ceafe         35.38      44.23      39.32
	ceafm         53.33      47.06      50.00
	lea           33.33      25.49      28.89
	CoNLL score:  39.10
	$ python3 ~/code/coval/scorer.py knack_file7.gold.conll knack_file7.dutchcoref.conll remove_singletons
	5 and 2 singletons are removed from the key and system files, respectively
				 recall  precision         F1
	mentions      80.00      85.71      82.76
	muc           70.00      77.78      73.68
	bcub          66.67      77.38      71.63
	ceafe         80.18      80.18      80.18
	ceafm         80.00      85.71      82.76
	lea           60.00      71.43      65.22
	CoNLL score:  75.16

