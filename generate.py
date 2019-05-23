#!/usr/bin/env python3
#-*- coding: utf-8 -*-

# Author: Anastassia Shaitarova

import sys, random, re
from nltk import ngrams
from collections import Counter, defaultdict
import random
from math import log
import operator


class NGramModel:
	'''
	This class reads in training data, collects n_grams of user-defined
	length, and finds their probabilities. Then it generates sentences
	based on those probabilities.
	'''
	def __init__(self, train_file):
		self.train_file = train_file
		self.ngrams = []
		self.ngram_frequencies = defaultdict(float)
		self.histories = defaultdict(float)
		self.start_sent = set() # set of sentence starters
		self.logprobs = defaultdict(lambda:defaultdict(float))
		self.bestpath = defaultdict(str)

	def generate_sentences(self, ngram_size):
		'''
		generates most probable sentences from training data
		'''
		self._extract_ngram_probabilities(ngram_size)

		def yield_unigram(k):
			for i, w in enumerate(k):
				if w[0] != '</s>' and w[0] != '<s>':
					if i <= 15:
						yield w[0]

		# generate sentences based on uni-grams
		if ngram_size == 1:
			k = random.sample(self.ngrams, k=30)
			print(' '.join(yield_unigram(k)))

		else:
			# initialize sentence
			sent = []
			# choose a starter at random
			BOS = random.sample(self.start_sent, 1)
			BOS = BOS[0]
			# start sentence
			for i in BOS:
				if i != '<s>':
					sent.append(i)

			# start generating sentences
			head = self.bestpath[BOS]
			# keep sentence under 20 tokens
			while head != '</s>' and len(sent) < 20:
				sent.append(head)
				# use slice of sentence as key to get new head
				hist = (tuple(sent[-len(BOS):]))
				head = self.bestpath[hist]

			print(' '.join(sent))

	def _extract_ngram_probabilities(self, ngram_size):
		'''
		collects n_grams and stores them in a dictionary
		'''
		for line in self.train_file:
			line = line.split()
			line.insert(0, '<s>')
			line.append('</s>')
			self.ngrams += ngrams(line, ngram_size)

		if ngram_size > 1:
			for n in self.ngrams:
				head, history = n[-1], n[:-1]
				# count n-grams
				self.ngram_frequencies[n] += 1
				# count the n-grams without the last word
				self.histories[history] +=1
				# save n-grams that start sentences for better generation
				if n[0] == '<s>':
					self.start_sent.add(history)

			# calculate probabilities
			for n in self.ngrams:
				head, history = n[-1], n[:-1]
				prob = self.ngram_frequencies[n] / self.histories[history]
				self.logprobs[history][head] = prob

			# save only the most probable heads for each n-gram
			for k, v in self.logprobs.items():
				v = max(v.items(), key=operator.itemgetter(1))[0]
				self.bestpath[k] = v


def main():
	pass

if __name__ == "__main__":
	main()
