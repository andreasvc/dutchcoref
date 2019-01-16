#!/bin/sh
# Download SemEval 2010 shared task data
# http://www.lsi.upc.edu/~esapena/downloads/index.php?id=1
sed 's/‚Äö√Ñ¬∞/à/g; s/¬¨‚àë/á/g; s/‚àö√©/ë/g; s/‚àö√†/é/g; s/‚àö√£/è/g; s/‚àö√ß/ê/g; s/‚àö√Æ/ï/g; s/¬¨‚àè/ü/g; s/√Ä√ú/ö/g; s/‚àö√µ/ó/g; s/‚àö‚àë/Ö/g; s/‚àö√Ö/ç/g;' < nl.devel.txt > nl.devel.txt.fixed
