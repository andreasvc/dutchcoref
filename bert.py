"""Extract BERT token embeddings for sentences.

Based on https://github.com/Filter-Bubble/e2e-Dutch/blob/master/e2edutch/bert.py
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, logging

logging.set_verbosity_error()


def loadmodel(name='GroNLP/bert-base-dutch-cased', numthreads=1):
	"""Load BERT model."""
	torch.set_num_threads(numthreads)
	tokenizer = AutoTokenizer.from_pretrained(name)
	bertmodel = AutoModel.from_pretrained(name)
	return tokenizer, bertmodel


def getvectors(parsespath, sentences, tokenizer, model, cache=True):
	"""Encode `sentences` (list of lists with tokens) and cache in file next
	to directory with parses."""
	cachefile = parsespath + '.bertvectors.npy'
	if (cache and os.path.exists(cachefile) and os.stat(cachefile).st_mtime
			> os.stat(parsespath).st_mtime):
		embeddings = np.load(cachefile)
	else:
		# use BERT to obtain vectors for the given sentences
		# NB: this encodes each sentence independently
		# embeddings = encode_sentences(sentences, tokenizer, model)
		result = []
		for n in range(len(sentences)):
			# FIXME: encode multiple sentences at a time for context
			# result.extend(encode_sentences_overlap(
			# 		sentences, n, tokenizer, model))
			# NB: this encodes each sentence independently
			# print('sent:', n, ' '.join(sentences[n]))
			for sent in _encode_sentences(
					sentences[n:n + 1], tokenizer, model):
				result.extend(sent)
		embeddings = np.array(result)
		if cache:
			np.save(cachefile, embeddings)
	return embeddings


# NB: the following function is currently not used
def encode_sentences(sentences, tokenizer, model, layer=9):
	"""Encode tokens with BERT.

	:returns: a list with n_sentences items;
		each item is an array of shape (sent_length, hidden_size=768).

	Layer 9 gives the best results with coreference, according to
	https://www.aclweb.org/anthology/2020.findings-emnlp.389.pdf"""
	result = []
	# Encode 25 sentences at a time:
	for n in range(0, len(sentences), 25):
		for sent in _encode_sentences(
				sentences[n:n + 25], tokenizer, model, layer):
			result.extend(sent)
	return np.array(result)


def encode_sentences_overlap(sentences, n, tokenizer, model,
		layer=9, maxsegmentlen=128):
	"""Encode tokens of sentences[n] with BERT.

	Encodes a segment of up to 128 subwords consisting of sentences that
	precede sentences[n] and sentences[n] itself.

	:returns: an array of shape (sent_length, hidden_size=768)

	Layer 9 gives the best results with coreference, according to
	https://www.aclweb.org/anthology/2020.findings-emnlp.389.pdf"""
	# Apply BERT tokenizer (even if sentences are already tokenized, since BERT
	# uses subword tokenization).
	if n < 0 or n >= len(sentences):
		raise ValueError('n (%d) is out of bounds; len(sentences) == %d'
				% (n, len(sentences)))
	tokenized = [tokenizer.tokenize(word) for word in sentences[n]]
	nnumtokens = sum(1 for word in tokenized for tok in word)

	segmentlen = 0
	nn = n
	while nn >= 0:
		tokenized = [tokenizer.tokenize(word) for word in sentences[nn]]
		numtokens = sum(1 for word in tokenized for tok in word)
		if segmentlen + numtokens >= maxsegmentlen:
			break
		segmentlen += numtokens
		nn -= 1
	if segmentlen == 0 and nnumtokens < 512:
		nn = n - 1  # long sentence, use all subwords, but disable context.
	elif segmentlen == 0:
		raise ValueError('Sentence %d longer (%d subwords) than 512 subwords?'
				% (n, numtokens))
	sentence = sum(sentences[nn + 1:n + 1], [])
	print('encoding', sentence, file=sys.stderr)
	sentence_tokenized = [tokenizer.tokenize(word) for word in sentence]
	sentence_tokenized_flat = [tok for word in sentence_tokenized
			for tok in word]
	indices_flat = [i for i, word in enumerate(sentence_tokenized)
				for tok in word]

	max_nrtokens = len(sentence_tokenized_flat)
	indexed_tokens = np.zeros((1, max_nrtokens), dtype=int)
	idx = tokenizer.convert_tokens_to_ids(sentence_tokenized_flat)
	indexed_tokens[0, :len(idx)] = np.array(idx)

	# Convert inputs to PyTorch tensors
	tokens_tensor = torch.tensor(indexed_tokens)
	with torch.no_grad():
		# torch tensor of shape (n_sentences, sent_length, hidden_size=768)
		outputs = model(tokens_tensor, output_hidden_states=True)
		bert_output = outputs.hidden_states[layer].numpy()

	# Add up tensors for subtokens coming from same word
	max_sentence_length = len(sentence)
	bert_final = np.zeros((max_sentence_length, bert_output.shape[2]))
	counts = np.zeros(len(sentence))
	for tok_id, word_id in enumerate(indices_flat):
		bert_final[word_id, :] += bert_output[0, tok_id, :]
		counts[word_id] += 1
	for word_id, count in enumerate(counts):
		if count > 1:
			bert_final[word_id, :] /= count
	bert_final = np.array(bert_final)
	return bert_final[-nnumtokens:, :]


def _encode_sentences(sentences, tokenizer, model, layer=9):
	"""Encode tokens with BERT.

	:returns: an array of shape (n_sentences, sent_length, hidden_size=768)

	Layer 9 gives the best results with coreference, according to
	https://www.aclweb.org/anthology/2020.findings-emnlp.389.pdf"""
	# Apply BERT tokenizer (even if sentences are already tokenized, since BERT
	# uses subword tokenization).
	# https://discuss.huggingface.co/t/is-transformers-using-gpu-by-default/8500
	# device = "cuda:0" if torch.cuda.is_available() else "cpu"
	sentences_tokenized = [
			[tokenizer.tokenize(word) for word in sentence]
			for sentence in sentences]
	sentences_tokenized_flat = [
			[tok for word in sentence for tok in word]
			for sentence in sentences_tokenized]
	indices_flat = [
			[i for i, word in enumerate(sentence)
				for tok in word]
			for sentence in sentences_tokenized]

	max_nrtokens = max(len(s) for s in sentences_tokenized_flat)
	indexed_tokens = np.zeros((len(sentences), max_nrtokens), dtype=int)
	for i, sent in enumerate(sentences_tokenized_flat):
		idx = tokenizer.convert_tokens_to_ids(sent)
		indexed_tokens[i, :len(idx)] = np.array(idx)

	# Convert inputs to PyTorch tensors
	tokens_tensor = torch.tensor(indexed_tokens)
	with torch.no_grad():
		# torch tensor of shape (n_sentences, sent_length, hidden_size=768)
		outputs = model(tokens_tensor, output_hidden_states=True)  # .to(device)
		bert_output = outputs.hidden_states[layer].numpy()

	# Add up tensors for subtokens coming from same word
	max_sentence_length = max(len(s) for s in sentences)
	bert_final = np.zeros((bert_output.shape[0],
			max_sentence_length,
			bert_output.shape[2]))
	for sent_id in range(len(sentences)):
		counts = np.zeros(len(sentences[sent_id]))
		for tok_id, word_id in enumerate(indices_flat[sent_id]):
			bert_final[sent_id, word_id, :] += bert_output[sent_id, tok_id, :]
			counts[word_id] += 1
		for word_id, count in enumerate(counts):
			if count > 1:
				bert_final[sent_id, word_id, :] /= count
	bert_final = np.array(bert_final)
	return bert_final
