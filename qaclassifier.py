#!/usr/bin/env python

"""Quote attribution classifier.

Usage: qaclassifier.py -t <train> -v <validation> -p <parsesdir> -a <quoteannotationsdir>
Example: qaclassifier.py -t 'train/*.conll' -v 'dev/*.conll' -p parses/ -a annotations/riddlecoref/

"""

# requirements:
# - pip install 'transformers>=4.0' keras tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
import argparse
import bert
import numpy as np
import random as python_random
import xml.etree.ElementTree as ET

import sys
# from collections import Counter
from glob import glob
from lxml import etree
from sklearn import metrics
from tensorflow import keras
import tensorflow as tf
from coref import (readconll, parsesentid, readngdata, initialsegment,
                   extractmentionsfromconll, sameclause, debug, getquotations, Quotation, isspeaker)
from sklearn.metrics import precision_score, recall_score
from collections import Counter


DENSE_LAYER_SIZES = [500, 150, 150]
INPUT_DROPOUT_RATE = 0.2
DROPOUT_RATE = 0.5
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
EPOCHS = 100
PATIENCE = 5
LAMBD = 0.05  # L2 regularization

# do not link quote if all scores of candidates are below this value.
# the model does not have to be re-trained if this value is changed.
QUOTE_PAIR_THRESHOLD = 0.09 
MODELFILE = 'quote.pt'
VERBOSE = False


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", type=str, default='../riddlecoref/split/riddle/train/*.conll',
                        help="Train file(s). Default='../riddlecoref/split/riddle/train/*.conll'")
    parser.add_argument("-v", "--val", type=str, default='../riddlecoref/split/riddle/dev/*.conll',
                        help="Validation file(s). Default='../riddlecoref/split/riddle/dev/*.conll'")
    parser.add_argument("-p", "--parses", type=str, default='../riddlecoref/parses',
                        help="Validation file(s). Default='../riddlecoref/parses'")
    parser.add_argument("-a", "--annotations", type=str, default='../riddlecoref/annotations/riddlecoref',
                        help="Validation file(s). Default='../riddlecoref/annotations/riddlecoref'")
    parser.add_argument("-e", "--eval", default=False, action='store_true',
                        help="Evaluate without retraining the classifier (default=False)")
    args = parser.parse_args()
    return args


def get_mention_object(mention_element, mention_objects, idx):
    """Return the mention object that matches with the mention xml element"""
    me_begin, me_end = int(mention_element.get("begin")), int(mention_element.get("end"))
    me_ttokenno = int(mention_element.get("ttokenno"))
    for m in mention_objects:
        ttokenno = idx[(m.sentno, m.begin)] + 1  # Add 1 because of indexing difference
        # Check whether total token numbers align as well as token length:
        if ttokenno == me_ttokenno and (m.end - m.begin - 1) == (me_end - me_begin):
            return m


def get_candidates(quote, mentions, candidates, idx):
    """Get all the mention candidates and add them to the candidates list"""
    for m in mentions:
        # Check whether paragraph number exists
        if m.parno in [quote.parno, quote.parno+1, quote.parno-1]: #Todo: range can be increased
            # Get mention global start- and end tokenno
            m.gstart = idx[m.sentno, m.begin]
            m.gend = m.gstart + (m.end - m.begin)
            # Check whether mention does not appear within quote
            if not (m.gstart >= quote.start and m.gend <= quote.end):
                # Collect names, nouns (also inanimate) and only some specific pronouns
                if m.type != "pronoun" or m.node.get('vwtype') in ['bez', 'pers', 'pr']:
                    candidates.append(m)


def build_mlp_model(input_shape):
    """Define a binary classifier."""
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=input_shape),
        keras.layers.Dropout(INPUT_DROPOUT_RATE),

        keras.layers.Dense(DENSE_LAYER_SIZES[0], name='dense0'),
        keras.layers.BatchNormalization(name='bn0'),
        keras.layers.Activation('relu'),
        keras.layers.Dropout(DROPOUT_RATE),

        keras.layers.Dense(DENSE_LAYER_SIZES[1], name='dense1'),
        keras.layers.BatchNormalization(name='bn1'),
        keras.layers.Activation('relu'),
        keras.layers.Dropout(DROPOUT_RATE),

        # keras.layers.Dense(DENSE_LAYER_SIZES[2], name='dense2'),
        # keras.layers.BatchNormalization(name='bn2'),
        # keras.layers.Activation('relu'),
        # keras.layers.Dropout(DROPOUT_RATE),

        keras.layers.Dense(
            1, name='output',
            kernel_regularizer=keras.regularizers.l2(LAMBD)),
        keras.layers.Activation('sigmoid'),
    ])
    return model


class QuoteFeatures:
    def __init__(self):
        self.result = []  # collected feature vectors for pairs
        self.labels = []  # the target: mention is speaker (1) or not (0)
        self.candidates = []  # the candidate mentions in each pair
        self.quotedata = []  # tuple containing row indices for candidates and quote object

    def add(self, trees, embeddings, quotations, gqm_dict, mentions, idx):
        addgold(quotations, gqm_dict)  # Add goldquote elements as attribute to quotation objects
        result = []

        for q in quotations:
            cand_startcount = len(self.candidates)

            # 1. Get candidate mentions
            get_candidates(q, mentions, self.candidates, idx)

            # 2. Get gold mention and store target label for each mention candidate
            gold_mention = None
            if hasattr(q, 'gold'):  # Work only with gold quotes:
                me = gqm_dict[q.gold]
                mo = get_mention_object(me, mentions, idx) if me is not None else None
                if mo is not None:
                    gold_mention = mo
            target_labels = [m == gold_mention for m in self.candidates[cand_startcount:]]
            self.labels += target_labels
            q.mo = gold_mention

            # 3. get features for candidates
            for m in self.candidates[cand_startcount:]:
                # Get quote length
                length = q.end - q.start
                # Get paragraph distance between quote and mention
                par_distance = abs(q.parno - m.parno)
                # Get distance in tokens between quote and mention
                m_globbegin = idx[(m.sentno, m.begin)]
                m_globend = m_globbegin + len(m.tokens)
                token_dist = min(abs(m_globbegin - q.end), abs(q.start - m_globend))
                # Get token distance from this quote to previous quote
                qdist = q.start - self.quotedata[-1][-1].end if len(self.quotedata) > 0 else -1
                # See whether mention or mention's clusterid appears in previous quote
                minprevious, mcinprevious = False, False
                if len(self.quotedata) >= 1:
                    previousquote = self.quotedata[-1][-1]
                    # Get all mentions in previous quote
                    previousmentions = [mn for mn in mentions if previousquote.start <= idx[mn.sentno, mn.begin] and
                                        previousquote.end >= idx[mn.sentno, mn.begin] + (mn.end - mn.begin)]
                    if m in previousmentions:
                        minprevious = True
                    if m.clusterid in [pm.clusterid for pm in previousmentions]:
                        mcinprevious = True

                # See whether mention is speaker of speechverb
                speaks = isspeaker(m)

                # Number of quotes between current quote and mention candidate
                between_quotes = 0
                for quote in quotations:
                    # see whether global end is > current globend but < than global mention end
                    # If candidate mention comes after quote, get quotes in between
                    if (m_globend > q.end) and (q.end < quote.end < m_globend):
                        between_quotes += 1
                    # Elif candidate mention comes before quote, get quotes in between
                    elif (m_globend < q.end) and (m_globend < quote.end < q.end):
                        between_quotes += 1

                # Add all features
                feats = (q.sentno, q.start, q.end, m.sentno, m.begin, m.end,
                         m.type == "pronoun", m.type == "noun", m.type =="name",
                         m.features['person'] == '1', m.features['person'] == '2', m.features['person'] == '3',
                         m.features['human'] == 1, m.features['human'] == 0, m.features['gender'] == 'f',
                         m.features['gender'] == 'fm', m.features['gender'] == 'm', m.features['gender'] == 'n',
                         par_distance, token_dist, speaks, between_quotes)

                # Or uncomment this to just use embeddings
                # feats = (q.sentno, q.start, q.end, m.sentno, m.begin, m.end)
                result.append(feats)

            # 4. store quote data
            self.quotedata.append((cand_startcount, len(self.candidates), q))

        if not result:
            return

        numotherfeats = len(result[0]) - 6  # First 6 features about sentno, start, end
        buf = np.zeros((len(result),
                        2 * embeddings.shape[-1] + numotherfeats))
        for n, featvec in enumerate(result):
            # mean of BERT token representations of the tokens in the mentions.
            qsent, qbegin, qend = featvec[:3]
            msent, mbegin, mend = featvec[3:6]
            # buf[n, :embeddings.shape[-1]] = embeddings[
            #                                 idx[qsent, qbegin]:idx[qsent, qend - 1] + 1].mean(axis=0)
            buf[n, :embeddings.shape[-1]] = embeddings[qbegin:qend + 1].mean(axis=0)
            buf[n, embeddings.shape[-1]:2 * embeddings.shape[-1]] = embeddings[
                                                                    idx[msent, mbegin]:idx[msent, mend - 1] + 1].mean(
                axis=0)
            buf[n, -numotherfeats:] = featvec[-numotherfeats:]
        self.result.append(buf)

    def getvectors(self):
        return (np.vstack(self.result),
                np.array(self.labels, dtype=int),
                self.candidates,
                self.quotedata)


def addgold(quotations, goldquotes):
    """Adds the gold quote element as attribute to the quotation object
    if the quotation objects appears in the gold quotes
    (based on matching -start and end token index)"""
    for quote_object in quotations:
        # Get goldquote element when start- and end token index match
        goldquote = [q for q in goldquotes if (quote_object.start, quote_object.end) ==
                     (int(q.get("ttokenno")) - 1,
                      int(q.get("ttokenno")) + (int(q.get("end")) - int(q.get("begin"))))]
        if goldquote:
            quote_object.gold = goldquote[0]  # Should only be one matching quote in list


def loadmentions(conllfile, parsesdir):
    ngdata, gadata = readngdata()
    # assume single document
    conlldata = next(iter(readconll(conllfile).values()))
    pattern = os.path.join(parsesdir, '*.xml')
    filenames = sorted(glob(pattern), key=parsesentid)
    if not filenames:
        raise ValueError('parse trees not found: %s' % pattern)
    trees = [(parsesentid(filename), etree.parse(filename))
             for filename in filenames]
    # extract gold mentions with gold clusters
    mentions = extractmentionsfromconll(conlldata, trees, ngdata, gadata,
                                        goldclusters=True)
    return trees, mentions


def get_gqm(xml_file):
    """Returns a dictionary with gold quote and mention elements"""
    gqm_dict = {}
    with open(xml_file, 'r') as f:
        tree = ET.parse(f)
        root = tree.getroot()
        for quote in root.iter('quote'):
            connection = quote.get("connection")
            mention = root.find(f".//mention[@id='{connection}']")
            gqm_dict[quote] = mention

    return gqm_dict


def getfeatures(pattern, parsesdir, tokenizer, bertmodel, annotationsdir=None):
    data = QuoteFeatures()
    files = glob(pattern)
    if not files:
        raise ValueError('pattern did not match any files: %s' % pattern)

    # For each Conll file, get the trees, mentions and quotes
    for n, conllfile in enumerate(files, 1):
        basename = os.path.basename(conllfile.rsplit('.', 1)[0])
        parses = os.path.join(parsesdir, basename)
        trees, mentions = loadmentions(conllfile, parses)
        embeddings = bert.getvectors(parses, trees, tokenizer, bertmodel)
        quotations, idx, doc = getquotations(trees)
        if annotationsdir:
            gqm_dict = get_gqm(os.path.join(annotationsdir, basename + ".xml"))  # gold quotes and mentions
        else:
            gqm_dict = dict()
        data.add(trees, embeddings, quotations, gqm_dict, mentions, idx)
        print(f'encoded {n}/{len(files)}: {conllfile}', file=sys.stderr)
    X, y, candidates, quotedata = data.getvectors()
    return X, y, candidates, quotedata


def train(trainfiles, validationfiles, parsesdir, annotationsdir, tokenizer, bertmodel):
    # Set random seeds
    np.random.seed(1)
    python_random.seed(1)
    tf.random.set_seed(1)

    # Define train and test data
    X_train, y_train, _clusters, _indices = getfeatures(
        trainfiles, parsesdir, tokenizer, bertmodel, annotationsdir=annotationsdir)
    X_val, y_val, _clusters, _indices = getfeatures(
        validationfiles, parsesdir, tokenizer, bertmodel, annotationsdir=annotationsdir)
    print('training data', X_train.shape)
    print('validation data', X_val.shape)

    classif_model = build_mlp_model([X_train.shape[-1]])
    classif_model.summary()
    classif_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy')
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=PATIENCE,
            restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(
            MODELFILE, monitor='val_loss', verbose=0,
            save_best_only=True, mode='min',
            save_weights_only=True),
    ]

    classif_model.fit(x=X_train, y=y_train, epochs=EPOCHS,
                      batch_size=BATCH_SIZE, callbacks=callbacks,
                      validation_data=(X_val, y_val), verbose=1)
    with open(MODELFILE.replace('.pt', '.txt'), 'w', encoding='utf8') as out:
        print(' '.join(sys.argv), file=out)


def get_predictions(a, b, quote, candidates, probs, gold, pred, closest=False, verbose=False):
    """Get mention and speaker predictions and gold labels, using either
    the most likely (highest probability) or the closest mentions"""
    gold.append(quote.mo if quote.mo is not None else None)

    if closest:  # Get mention that is closest to the quote
        lowest = 100
        candidate = None
        for m in candidates[a:b]:
            distance = min(abs(quote.end - m.gstart), abs(quote.start - m.gend))
            if distance < lowest:
                lowest = distance
                candidate = m
        pred.append(candidate)

    elif a == b:  # a quote with no candidates
        pred.append(None)

    else:  # Get most likely mention
        if a == b:  # a quote with no candidates
            pred.append(None)
            mention = None
        else:
            mention = candidates[a + probs[a:b].argmax()]
            # NB: if none of the candidates is likely enough, predict None
            pred.append(mention if probs[a:b].max() > QUOTE_PAIR_THRESHOLD else None)

        # Optional: print prediction information
        if verbose and mention is not None:
            print(f'{int(pred[-1] == gold[-1])} {probs[a:b].max():.3f}',
                  quote.sentno, quote.start, quote.text, '->', end=' ')
            if probs[a:b].max() > QUOTE_PAIR_THRESHOLD:
                print(mention.sentno, mention.begin, ' '.join(mention.tokens))
            else:
                print('(none)')
        elif verbose:
            print('(none)')


def evaluate(validationfiles, parsesdir, annotationsdir, tokenizer, bertmodel):
    X_val, y_val, candidates, quotedata = getfeatures(
        validationfiles, parsesdir, tokenizer, bertmodel, annotationsdir=annotationsdir)

    model = build_mlp_model([X_val.shape[-1]])
    model.load_weights(MODELFILE).expect_partial()
    probs = model.predict(X_val)

    # Get the gold and predicted mentions and clusters:
    gold, pred = [], []
    for a, b, quote in quotedata:
        # a and b are row indices in X_train with all the candidates
        # for a single quote object.
        get_predictions(a, b, quote, candidates, probs, gold, pred, closest=False, verbose=VERBOSE)
    pairpred = probs > QUOTE_PAIR_THRESHOLD

    # Print classification report for quote-candidate pairs
    print('(quote, candidate) pair classification scores:')
    print(metrics.classification_report(y_val, pairpred, digits=3, zero_division=0))

    # print precision and recall for mentions and then for clusters:
    print_results(pred, gold, "mention")
    print_results(pred, gold, "cluster")


def predictions(trees, embeddings, quotations, mentions, idx):
    data = QuoteFeatures()
    gqm_dict = dict()
    data.add(trees, embeddings, quotations, gqm_dict, mentions, idx)

    X_val, y_val, candidates, quotedata = data.getvectors()

    model = build_mlp_model([X_val.shape[-1]])
    model.load_weights(MODELFILE).expect_partial()
    probs = model.predict(X_val)

    mo_predictions = []
    # Get the predicted mention objects for each quote:
    for a, b, quote in quotedata:
        if a == b:
            mo_predictions.append(None)
        else:
            # Get most likely mention that is above a certain threshold
            most_likely = candidates[a + probs[a:b].argmax()]
            m = most_likely if probs[a:b].max() > QUOTE_PAIR_THRESHOLD else None
            mo_predictions.append(m)

    assert len(quotations) == len(mo_predictions)
    for quote, prediction in zip(quotations, mo_predictions):
        quote.speaker = prediction


def print_results(predicted_mentions, gold_mentions, typestring="cluster"):
    """Prints precision, recall and F1-score for either mentions or clusters"""
    if typestring == "mention":
        print("\n## Results for mentions")
        gold = [m.id if m is not None else -1 for m in gold_mentions]
        pred = [m.id if m is not None else -1 for m in predicted_mentions]

    else:  # typestring == "cluster":
        print("\n## Results for clusters")
        gold = [m.clusterid if m is not None else -1 for m in gold_mentions]
        pred = [m.clusterid if m is not None else -1 for m in predicted_mentions]

    n, correct, nospeaker = 0, 0, 0
    for g, pre in zip(gold, pred):
        if g != -1:  # Ignore cases where there is no gold speaker
            correct += g == pre
            nospeaker += pre == -1
            n += 1

    print(f"Precision: {correct} / {n - nospeaker} = {correct / (n - nospeaker):.3f}")
    print(f"Recall: {correct} / {n} = {correct / n:.3f}")
    print(
        f"F score: {2 * (correct / (n - nospeaker) * (correct / n)) / (correct / (n - nospeaker) + (correct / n)):.3f}")
    print(f"No speaker:", nospeaker)


def main():
    args = create_arg_parser()
    trainfiles, validationfiles, parsesdir, annotationsdir = args.train, args.val, args.parses, args.annotations
    tokenizer, bertmodel = bert.loadmodel()

    # Train and evaluate
    if not args.eval:
        train(trainfiles, validationfiles, parsesdir, annotationsdir, tokenizer, bertmodel)
    evaluate(validationfiles, parsesdir, annotationsdir, tokenizer, bertmodel)


if __name__ == '__main__':
    main()
