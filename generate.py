import sys, random, re
from nltk import ngrams
from nltk.tokenize import sent_tokenize, word_tokenize
#from intertools import islice, izip
from collections import Counter, defaultdict
import random
from math import log


class NGramModel:
	def __init__(self, train_file):
		self.train_file = train_file
		self.unigrams = defaultdict(int)
		self.all_words = []
		self.n_grams = []
		self.bi_grams = []
		self.start_sent = set()
		self.logprobs = defaultdict(lambda: defaultdict(float))


	def generate_sentences(self, ngram_size=2):
		self._extract_ngram_probabilities(ngram_size)

		def yield_unigram(k):
			for i, w in enumerate(k):
				if w != '</s>' and w != '<s>':
					if i <= 15:
						yield w
					else:
						if w == '</s>':
							break
		# generate random sentences with unigrams
		# very random, no order whatsoever
		if ngram_size == 1:
			k = random.sample(self.all_words, k=30)
			print(' '.join(yield_unigram(k)))

		else:
			# find beginning of a sentence:
			# print(self.start_sent)
			BOS = random.sample(self.start_sent, 1)
			BOS = BOS[0]
			print(BOS)

			# this only works with bi_grams so far
			print(BOS+' '+' '.join(i for i in self._yield_next_word((BOS,)) if i != '</s>'))

	def _yield_next_word(self, tuple):
		while tuple[0] != '</s>':
			w = max(self.logprobs[tuple])
			tuple = (w,)
			yield w

	def _extract_ngram_probabilities(self, ngram_size):
		for line in self.train_file:
			line = line.split()
			line.insert(0, '<s>')
			line.append('</s>')
			self.n_grams += ngrams(line, ngram_size)
			self.bi_grams += ngrams(line, 2)
			for w in line:
				self.unigrams[w] += 1
				self.all_words.append(w)

		# ngram frequency dict
		cnt_ngrams = Counter(self.n_grams)
		cnt_bigrams = Counter(self.bi_grams)
		# print(cnt_ngrams)
		# probability of each ngram is:
		# total count(ngram) / total count of first element in bigram
		for ngram in self.n_grams:
			# print(ngram)
			# if unigrams:
			if ngram_size == 1:
				self.logprobs[ngram[0]] = log(self.unigrams[ngram[0]]/sum(self.unigrams.values()))
			# for bigger n-grams:
			elif ngram_size == 2:
				head, history = ngram[-1], ngram[:-1]
				if history[0] == '<s>':
					self.start_sent.add(head)
				probs = {}
				try:
					prob = cnt_ngrams[ngram] / self.unigrams[history[0]]
					# add current probabilities
					probs[head] = log(prob)
					# find hiest value
					max_value = max(probs.values())
					# add only ngram with highest probability
					self.logprobs[history] = probs
				except ZeroDivisionError:
					print("Unigram is not in dictionary", ngram)
			elif ngram_size == 3:
				# print(ngram)
				head, history = ngram[-1], ngram[:-1]
				# print(history, head)
				if history[0] == '<s>':
					self.start_sent.add((history[1], head))
				probs = {}
				try:
					prob = cnt_ngrams[ngram] / cnt_bigrams[history]
					# add current probabilities
					probs[head] = log(prob)
					# find hiest value
					max_value = max(probs.values())
					# add only ngram with highest probability
					self.logprobs[history] = probs
				except ZeroDivisionError:
					print("Unigram is not in dictionary", ngram)


		# print(self.logprobs)
		# print('BOS',self.start_sent)

def main():
	pass

if __name__ == "__main__":
	main()
