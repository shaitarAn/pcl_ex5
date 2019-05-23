#!/usr/bin/env python3
#-*- coding: utf-8 -*-

# Author: Anastassia Shaitarova

import sys
from classify import LyricsClassifier
from generate import NGramModel
import argparse
from pathlib import Path
from collections import defaultdict

# create an entry point for each subparser that accepts the args namespace
# as an argument
# https://stackoverflow.com/questions/38315599/not-getting-namespace-returned-from-subparser-in-parse-args

def classify_me(args):
    data = defaultdict(list)
    for f in args.train_file:
        label = Path(str(f)).stem
        for line in f:
            data[label].append(line.strip())

    classy = LyricsClassifier(data)

    if args.evaluate:
        classy.evaluate(data)
    elif args.text:
        if isinstance(args.text, str) == True:
            print(classy.predict_label(args.text))
        else:
            for line in args.text:
                print(classy.predict_label(line))

def generate_me(args):
	opn = NGramModel(args.train_file)
	if args.forever:
		while True:
			opn.generate_sentences(args.n)
	elif args.n and args.l:
		for _ in range(args.l):
			opn.generate_sentences(args.n)
	elif args.l:
		for _ in range(args.l):
			opn.generate_sentences()
	elif args.n:
		opn.generate_sentences(args.n)

parser = argparse.ArgumentParser(description="A tool to classify and generate lyrics from training data.")
subparsers = parser.add_subparsers(help="classify or generate lyrics?")
parser_class = subparsers.add_parser('classify', description='Classify lyrics using Naïve Bayes.')
parser_gen = subparsers.add_parser('generate', description="Generate lyrics using an n-gram model.")

parser_class.add_argument('train_file', help="files containing training data for each label; filename format: <label>.train", nargs='+', type=argparse.FileType('r'))
parser_class.add_argument('--text', help='text to use as input (otherwise, standard input is used)', default=sys.stdin)
parser_class.add_argument('--evaluate', help='evaluate performance on the training set', action="store_true")
parser_class.set_defaults(func=classify_me)

parser_gen.add_argument('train_file', help="file containing training data", type=argparse.FileType('r'), default=sys.stdin, nargs='?')
parser_gen.add_argument('-l', help='generate L lines', type=int, default=1)
parser_gen.add_argument('-n', choices=[1,2,3,4,5], help='n-gram order to use', type=int, default=3)
parser_gen.add_argument('--forever', help='generate lines in an infinite loop', action="store_true")
parser_gen.set_defaults(func=generate_me)
args = parser.parse_args()
args.func(args)
