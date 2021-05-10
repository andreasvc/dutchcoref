import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, logging
from coref import gettokens

logging.set_verbosity_error()


def loadmodel(name):
	"""Load BERT model."""
	tokenizer = AutoTokenizer.from_pretrained(name)
	bertmodel = AutoModel.from_pretrained(name)
	return tokenizer, bertmodel


def getvectors(parses, trees, tokenizer, model, cache=True):
	"""Encode sentences in `trees` and cache in file next to directory
	with parses."""
	cachefile = parses + '.bertvectors.npy'
	if (cache and os.path.exists(cachefile) and os.stat(cachefile).st_mtime
			> os.stat(parses).st_mtime):
		embeddings = np.load(cachefile)
	else:
		# now use BERT to obtain vectors for the text of these spans
		sentences = [gettokens(tree, 0, 9999) for _, tree in trees]
		# NB: this encodes each sentence independently
		# embeddings = encode_sentences(sentences, tokenizer, model)
		result = []
		for n in range(len(sentences)):
			# FIXME: encode multiple sentences at a time without padding
			for sent in _encode_sentences(
					sentences[n:n + 1], tokenizer, model):
				result.extend(sent)
		embeddings = np.array(result)
		if cache:
			np.save(cachefile, embeddings)
	return embeddings


def encode_sentences(sentences, tokenizer, model, layer=9):
	"""Encode tokens with BERT.

	:returns: a list with n_sentences items;
		each item is an array of shape (sent_length, hidden_size=768).

	Layer 9 is the most useful for coreference, according to
	https://www.aclweb.org/anthology/2020.findings-emnlp.389.pdf"""
	result = []
	# Encode 25 sentences at a time:
	for n in range(0, len(sentences), 25):
		for sent in _encode_sentences(
				sentences[n:n + 25], tokenizer, model, layer):
			result.extend(sent)
	return np.array(result)


def _encode_sentences(sentences, tokenizer, model, layer=9):
	"""Encode tokens with BERT.

	:returns: an array of shape (n_sentences, sent_length, hidden_size=768)

	Layer 9 is the most useful for coreference, according to
	https://www.aclweb.org/anthology/2020.findings-emnlp.389.pdf"""
	# Apply BERT tokenizer (even if sentences are already tokenized, since BERT
	# uses subword tokenization).
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
		outputs = model(tokens_tensor, output_hidden_states=True)
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
