Dutch coreference resolution & dialogue analysis
================================================
An implementation of the Stanford Multi-Pass Sieve Coreference System.

Dependencies and Datasets
-------------------------
Get the repository and install the required packages:

    $ git clone https://github.com/andreasvc/dutchcoref.git
    $ cd dutchcoref
    $ pip3 install -r requirements.txt

Unless you are working on an already parsed corpus, you will want to
install the [Alpino parser](http://www.let.rug.nl/vannoord/alp/Alpino/AlpinoUserGuide.html).

To get parse tree visualizations in the HTML output,
install https://github.com/andreasvc/disco-dop/

### Dutch first names (required)

Download `Top_eerste_voornamen_NL_2010.csv`
from https://www.meertens.knaw.nl/nvb/ (click on "Veelgestelde vragen"
and then on the bottom "Klik hier").
Unzip it and put the csv file in the `data/` directory.

### Number & gender data from Web text (required), CLIN26 shared task data (optional)

Clone this repository under same parent folder as this repository:

    ~/code/dutchcoref $ cd ..
    ~/code $ git clone https://bitbucket.org/robvanderg/groref.git

### SemEval 2010 shared task data (optional)

Download the data from http://www.lsi.upc.edu/~esapena/downloads/index.php?id=1

Apply the Unicode fix in `fixsemeval2010.sh`.
The directory `data/semeval2010NLdevparses` contains Alpino parses for the
Dutch development set of this task.

Usage examples
--------------
See `coref.py --help` for command line options.

### Parsing and coreference of a text file
```
$ cat /tmp/example.txt
' Ik ben de directeur van Fecalo , van hierachter , ' zei hij .
' Mag ik u iets vragen ? '
Ik vroeg hem binnen te komen .
$ mkdir example
$ cat example.txt | Alpino number_analyses=1 end_hook=xml -parse -flag treebank example
[...]
$ python3 coref.py --verbose /tmp/example
```

![verbose output](https://github.com/andreasvc/dutchcoref/raw/master/data/output.png "verbose output")

$ python3 coref.py --fmt=booknlp /tmp/example
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

### Error analysis against a gold standard .conll file
UPDATE: use https://github.com/andreasvc/berkeley-coreference-analyser

The following creates lots of output; scroll to the end for the error analysis.
Mention boundaries and links are printed in green if they are correct,
yellow if in gold but missing from output,
and red if in output but not in gold.
```
$ ls mydocument/
1.xml 2.xml [...]
$ python3 coref.py mydocument/ --gold=mydocument.conll --verbose | less -R
```
alternatively, use the HTML visualization and view the results in your favorite browser:
```
$ python3 coref.py mydocument/ --gold=mydocument.conll --verbose --fmt=html >output.html
```
See https://andreasvc.github.io/voskuil.html for an example of the HTML visualization.

For further analysis with other tools, use the option `--outputprefix' to dump information
on clusters, mentions, links and quotations:
```
$ python3 coref.py mydocument/ --fmt=booknlp --outputprefix=output
```
This creates the files `output.{mentions,clusters,links,quotes}.tsv` (tabular format)
`output.conll` (--fmt), and `output.icarus` (ICARUS allocation format).
Make sure you don't overwrite the gold standard conll file!


### Evaluation against a gold standard .conll file
Get the scorer script: https://github.com/andreasvc/coval
which is an improved version of: https://github.com/ns-moosavi/coval
```
$ python3 coref.py mydocument/ --fmt=conll2012 >output.conll
$ python3 ../coval/scorer.py mydocument.conll output.conll
mentions   Recall: 90.52  Precision: 81.43  F1: 85.73
muc        Recall: 79.44  Precision: 74.43  F1: 76.85
bcub       Recall: 51.72  Precision: 55.65  F1: 53.61
ceafe      Recall: 66.64  Precision: 46.58  F1: 54.83
lea        Recall: 49.48  Precision: 52.74  F1: 51.05
CoNLL score: 61.77
```


Column types
------------
With `--fmt=booknlp`, the output contains the following columns:

1. Global token number
2. Sentence ID
3. Token number within sentence
4. Token
5. Lemma
6. Rich POS tag (including morphological features)
7. UD parent token (ID as in column 3)
8. UD dependency label
9. Named entity class (PER, ORG, LOC, ...)
10. Speaker ID (if a speaker is found, every token in a direct speech utterance
    is assigned the speaker ID; the ID is the cluster ID of the speaker)
11. Similar as above, but for addressee.
12. Whether token is part of direct speech (B, I) or not (O)
13. Coreference cluster in CoNLL notation

Web demo
--------
The web demo accepts short pieces of text, takes care of parsing, and presents
a visualization of coreference results. Requires a running instance of
[alpiner](https://github.com/andreasvc/alpino-api/tree/master/demo).
Run with `python3 web.py`

Annotation workflow
-------------------
1. Preprocess, tokenize and parse a text with Alpino to get a directory of parse trees
    in XML files.
2. Run coreference resolution on the parse trees:
    `python3 coref.py --fmt=conll2012 path/to/parses/ > text.conll`
    (Forward slashes are required, also on Windows).
3. Get the latest stable release of [CorefAnnotator](https://github.com/nilsreiter/CorefAnnotator/releases).
    Run it with `java -jar CorefAnnotator-1.9.2-full.jar`
4. Import the `.conll` file (CoNLL 2012 button under "Import from other formats").
5. Read the [annotation guidelines](https://github.com/andreasvc/dutchcoref/raw/master/dutchcoref%20annotation%20guidelines.pdf) in this repository
6. Correct the annotation; save regularly
    (in the .xmi format used by CorefAnnotator)
7. When done, export to CoNLL 2012 format
8. The CoNLL 2012 file exported by CorefAnnotator does not contain POS tags and parse trees;
	to add those, run `addparsebits.py text.conll path/to/parses/`

References
----------
This code base is a Dutch implementation of the Stanford Multi-Pass Sieve
Coreference System for English:

Heeyoung Lee, Angel Chang, Yves Peirsman, Nathanael Chambers, Mihai Surdeanu, and Dan Jurafsky. Deterministic coreference resolution based on entity-centric, precision-ranked rules. Computational Linguistics, 39 (4):885–916, 2013. http://aclweb.org/anthology/J13-4004.pdf

See also these previous implementations
https://bitbucket.org/robvanderg/groref
and https://github.com/antske/coref_draft

The UD conversion uses a slightly modified version of https://github.com/gossebouma/lassy2ud

The number & gender data is derived from:

Shane Bergsma and Dekang Lin (2006). Bootstrapping Path-Based Pronoun Resolution, In Proceedings of COLING/ACL.
http://www.clsp.jhu.edu/~sbergsma/Gender/
