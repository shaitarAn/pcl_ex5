#!/usr/bin/env python3
#-*- coding: utf-8 -*-

# Author: Anastassia Shaitarova

from collections import defaultdict
from typing import Dict, Iterable, List
from math import log, exp


class LyricsClassifier:
    """Implements a Naïve Bayes model for song lyrics classification."""

    def __init__(self, train_data: Dict[str, Iterable[str]]):
        """Initialize and train the model.

        Args:
            train_data: A dict mapping labels to iterables of lines of
                        (e.g. a file object).
        """
        self.train_data = train_data
        self.label_counts = {} # count lines for each label
        self.label_feature_value_counts = {} # global dict
        self.total_line_count = 0
        self.feature_values = defaultdict(lambda: set()) # saves set of feature values globaly
        self.feature_posterior_probabilities = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:0)))

        for label, lines in self.train_data.items():
            # calculate raw counts given this label
            label_count = 0
            # this dict will go into self.label_feature_value_counts
            feature_value_counts = defaultdict(lambda: defaultdict(int))
            for line in lines:
                self.total_line_count += 1
                label_count += 1
                # get features in each line
                features = self._extract_features(line)
                for feature_id, value in features.items():
                    feature_value_counts[feature_id][value] += 1
                    self.feature_values[feature_id].add(value)

            self.label_counts[label] = label_count
            # a dict: for every label we get feature and its counts
            self.label_feature_value_counts[label] = feature_value_counts

        # save posterior probability of a feature, given a label
        for label, features in self.label_feature_value_counts.items():
            for feat, values in features.items():
                for val, num in values.items():
                    # print(feat, val, num)
                    self.feature_posterior_probabilities[label][feat][val] = self._aposteriori_probability(label, feat, val)

    def _apriori_probability(self, N):
        # number of lines with label devided by total number of lines
        return log(N/self.total_line_count)

    def _aposteriori_probability(self, l, f, v):
        # number of feature occurrences in label + 1 for smoothing devided by
        # number of lines with label + number of feature values for smoothing
        return log((self.label_feature_value_counts[l][f][v]+1)/(self.label_counts[l]+len(self.feature_values[f])))

    @staticmethod
    def _extract_features(line: str) -> Dict:
        """Return a dict containing features values extracted from a line.

        Args:
            line: A line of song lyrics.

        Returns:
            A dict mapping feature IDs to feature values.
        """
        return {
            'pray':'pray' in line, # nobody
            'grunts': any(grunt in line for grunt in ['oh', 'ah', 'uh']), # bobo
            'tokens':  len(line.split()),
            'charset': len(set(line)) // 4,
            'parenths': any(par in line for par in ['(',')']),
            'char_bigrams': len(set(list(zip(line, line[1:])))),
            'vowels': len(set(list(v for v in line if v in 'euoa')))/len(set(line)),

            # ## features that didn't help or made it worse:
            # ## ##########################################
            # 'chihuaha': "chihuaha" in line,
            # 'nums': any(num in line for num in ['1', '2', '3', '4', '5', '6', '7', '8', '9']),
            # 'we': sum(1 for i in line.split() if i == 'we'),
            # 'wh': any(wh in line for wh in ['what', 'where', 'why', 'whatever', 'when']),
            # 'startswith': line.startswith('i '),
            # 'ascii': all(ord(char) < 128 for char in line),
            # 'tokens':  len(line.split()) < 6, # bobo
            # 'long_lines': len(line.split()) > 15,
            # 'know': 'know' in line, # mike
            # 'bad': 'bad' in line,
            # '!': '!' in line,
            # '?': '?' in line,
            # 'short words': sum(1 for i in line.split() if len(i) == 3)/len(line.split()),
            # 'pronouns': any(pron in line for pron in ['i', 'me', 'we', 'you']),
            # 'areyouready': 'ready' in line, # bobo
            # 'char_trigrams': len(list(zip(line, line[1:], line[2:]))),
            # 'bigrams': len(set(list(zip(line.split(), line[1:].split()))))/len(list(zip(line.split(), line[1:].split()))),
            # 'bobo': 'bobo' in line, # bobo
            # 'colors': any(c in line for c in ['black', 'white']),
            # 'vocab': len(set(line.split()))/len(line.split()) < 0.50,

        }

    def _probability(self, label: str, features: Dict) -> float:
        """Return P(label|features) using the Naïve Bayes assumption.

        Args:
            label: The label.
            features: A dict mapping feature IDs to feature values.

        Returns:
            The non-logarithmic probability of `label` given `features`.
        """
        # probability of 'label' given 'features' in non-log space
        # apriori probability of label + sum of posterior probabilities
        N = self.label_counts[label]
        post_probs = []
        for f, v in features.items():
            post_probs.append(self.feature_posterior_probabilities[label][f][v])
        p = self._apriori_probability(N)+sum(post_probs)
        prob = exp(p)
        return prob

    def predict_label(self, line: str) -> str:
        """Return the most probable prediction by the model.

        Args:
            line: The line to classify.

        Returns:
            The most probable label according to Naïve Bayes.
        """
        values = []
        f_dict = self._extract_features(line)
        for label in self.label_counts.keys():
            values.append((self._probability(label, f_dict), label))

        return max(values)[1]

    def evaluate(self, test_data: Dict[str, Iterable[str]]):
        """
        Evaluate the model and print recall, precision, accuracy and f1-score.

        Args:
            test_data: A dict mapping labels to iterables of lines
                       (e.g. a file object).
        """
        # save counts for TP, FN for each label
        counts = defaultdict(lambda:defaultdict(int))
        total_lines = 0
        for label, lines in test_data.items():
            total_lines += len(lines)
            true_labels = []
            predictions = []
            for line in lines:
                true_labels.append(label)
                predictions.append(self.predict_label(line))

            zipped = zip(true_labels, predictions)

            for i in zipped:
                # find mislabeled lines as FN
                # FNs for one label are the FPs for the other label
                if i[0] == label and i[1] != label:
                    counts[label]['fn'] += 1
                    counts[i[1]]['fp'] += 1
                    # for micro-average f1 FN are also FP
                    counts['total']['fp'] += 1
                    counts['total']['fn'] += 1
                else:
                    counts[label]['tp'] += 1
                    counts['total']['tp'] += 1

        # print to see TP, FP, and FN for both classes and as total binary
        # print(counts)

        for l, c in counts.items():
            if l != 'total':
                tp = c['tp']
                fp = c['fp']
                fn = c['fn']
                tn = total_lines - (tp+fp+fn)
                prec, recall, f1, accuracy= self._calculate_eval_metrics(tp, fp, fn, tn)

                print('-'*20)
                print("{0}\nprecision:{1:10.2f}%\nrecall:{2:13.2f}%\nf1-score:{3:11.2f}%\naccuracy:{4:11.2f}%".format(l, prec, recall, f1, accuracy))

        for l, c in counts.items():
            if l == 'total':
                tps = c['tp']
                fps = c['fp']
                fns = c['fn']
                tns = 0
                _, _, F1, _= self._calculate_eval_metrics(tps, fps, fns, tns)

                print('-'*20)
                print("Micro-average F1: {0:.2f}%".format(F1))
                print()


    def _calculate_eval_metrics(self, tp, fp, fn, tn):
        try:
            prec = tp /(tp+fp)
        except ZeroDivisionError:
            prec = 1
        try:
            recall = tp / (tp+fn)
        except ZeroDivisionError:
            recall = 1
        try:
            accuracy = tp/(tp+fp+fn+tn)
        except ZeroDivisionError:
            accuracy = 1
        try:
            f1 = 2*prec*recall / (prec+recall)
        except ZeroDivisionError:
            f1 = 0
        return prec, recall, f1, accuracy
