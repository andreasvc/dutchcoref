Dutch coreference resolution & dialogue analysis
================================================
An implementation of the Stanford Multi-Pass Coreference System.

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

### Dutch first names

download `Top_eerste_voornamen_NL_2010.csv`
from https://www.meertens.knaw.nl/nvb/
and put it in the `data/` directory.

### CLIN26 shared task data, number & gender data from Web text

Clone this repository under same parent folder as this repository::

    ~/code/dutchcoref $ cd ..
    ~/code $ git clone https://bitbucket.org/robvanderg/groref.git

### SemEval 2010 shared task data

Download the data from http://www.lsi.upc.edu/~esapena/downloads/index.php?id=1
Apply the Unicode fix in `fixsemeval2010.sh`.
The directory `data/semeval2010NLdevparses` contains Alpino parses for the
Dutch development set of this task.

Example
-------
```
$ cat /tmp/example.txt
' Ik ben de directeur van Fecalo , van hierachter , ' zei hij .
' Mag ik u iets vragen ? '
Ik vroeg hem binnen te komen .
$ mkdir example
$ cat example.txt | Alpino number_analyses=1 end_hook=xml -flag treebank example -parse
[...]
```

![verbose output](https://github.com/andreasvc/dutchcoref/raw/master/datat/output.png "verbose output")

```
#begin document (example);
1       6-1     1       '       '       LET()   5       punct   -       14      -       B       -
2       6-1     2       Ik      ik      VNW(pers,pron,nomin,vol,1,ev)   5       nsubj   -       14      -       I       (0)
3       6-1     3       ben     zijn    WW(pv,tgw,ev)   5       cop     -       14      -       I       -
4       6-1     4       de      de      LID(bep,stan,rest)      5       det     -       14      -       I       (0
5       6-1     5       directeur       directeur       N(soort,ev,basis,zijd,stan)     0       root    -       14      -       I       0
6       6-1     6       van     van     VZ(init)        7       case    -       14      -       I       0
7       6-1     7       Fecalo  Fecalo  N(eigen,ev,basis,zijd,stan)     5       nmod    ORG     14      -       I       0)|(1)
8       6-1     8       ,       ,       LET()   5       punct   -       14      -       I       -
9       6-1     9       van     van     VZ(init)        10      case    -       14      -       I       -
10      6-1     10      hierachter      hierachter      BW()    5       nmod    -       14      -       I       -
11      6-1     11      ,       ,       LET()   5       punct   -       14      -       I       -
12      6-1     12      '       '       LET()   5       punct   -       14      -       I       -
13      6-1     13      zei     zeggen  WW(pv,verl,ev)  5       parataxis       -       -       -       O       -
14      6-1     14      hij     hij     VNW(pers,pron,nomin,vol,3,ev,masc)      13      nsubj   -       -       -       O       (0)
15      6-1     15      .       .       LET()   5       punct   -       -       -       O       -

16      6-2     1       '       '       LET()   6       punct   -       14      -       B       -
17      6-2     2       Mag     mogen   WW(pv,tgw,ev)   6       aux     -       14      -       I       -
18      6-2     3       ik      ik      VNW(pers,pron,nomin,vol,1,ev)   6       nsubj   -       14      -       I       (0)
19      6-2     4       u       u       VNW(pers,pron,nomin,vol,2b,getal)       6       iobj    -       14      -       I       (5)
20      6-2     5       iets    iets    VNW(onbep,pron,stan,vol,3o,ev)  6       obj     -       14      -       I       -
21      6-2     6       vragen  vragen  WW(inf,vrij,zonder)     0       root    -       14      -       I       -
22      6-2     7       ?       ?       LET()   6       punct   -       14      -       I       -
23      6-2     8       '       '       LET()   6       punct   -       14      -       I       -

24      7-1     1       Ik      ik      VNW(pers,pron,nomin,vol,1,ev)   2       nsubj   -       -       -       O       (6)
25      7-1     2       vroeg   vragen  WW(pv,verl,ev)  0       root    -       -       -       O       -
26      7-1     3       hem     hem     VNW(pers,pron,obl,vol,3,ev,masc)        2       iobj    -       -       -       O       (0)
27      7-1     4       binnen  binnen  VZ(fin) 6       compound:prt    -       -       -       O       -
28      7-1     5       te      te      VZ(init)        6       mark    -       -       -       O       -
29      7-1     6       komen   binnen_komen    WW(inf,vrij,zonder)     2       xcomp   -       -       -       O       -
30      7-1     7       .       .       LET()   2       punct   -       -       -       O       -

#end document
```

References
----------
This code base is a Dutch implementation of the Stanford Multi-Pass Coreference System for English:

Heeyoung Lee, Angel Chang, Yves Peirsman, Nathanael Chambers, Mihai Surdeanu, and Dan Jurafsky. Deterministic coreference resolution based on entity-centric, precision-ranked rules. Computational Linguistics, 39 (4):885–916, 2013. http://aclweb.org/anthology/J13-4004.pdf

See also these previous implementations
https://bitbucket.org/robvanderg/groref
and https://github.com/antske/coref_draft
