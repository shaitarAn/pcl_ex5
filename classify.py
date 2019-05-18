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
        test_data = train_data
        self.train_data = train_data
        self.test_data = {}
        self.label_counts = {}
        self.label_feature_value_counts = {}
        self.total_line_count = 0
        self.feature_values = defaultdict(lambda: set())
        self.feature_posterior_probabilities = defaultdict(lambda:defaultdict(float))

        for label, lines in self.train_data.items():
            # calculate raw counts given this label
            label_count = 0
            feature_value_counts = defaultdict(lambda: defaultdict(int))
            list_of_lines = []
            for line in lines:
                list_of_lines.append(line.strip())
                self.total_line_count += 1
                label_count += 1
                # for every line we check if feature is present, count
                features = self._extract_features(line)
                for feature_id, value in features.items():
                    feature_value_counts[feature_id][value] += 1
                    self.feature_values[feature_id].add(value)
            self.test_data[label] = list_of_lines

            self.label_counts[label] = label_count
            # a dict: for every label we get feature and its counts
            self.label_feature_value_counts[label] = feature_value_counts

        # apriori probability per class in logarithmic space
        self.pBOBO = self._apriori_probability(self.label_counts['bobo'])
        self.pMIKE = self._apriori_probability(self.label_counts['mike'])

        # posterior probability of a feature, given a label
        for label, features in self.label_feature_value_counts.items():
            for f, v in features.items():
                self.feature_posterior_probabilities[label][f] = self._aposteriori_probability(label, f)

        self.evaluate(self.test_data)

    def _apriori_probability(self, N):
        # number of lines with label devided by total number of lines
        return log(N/self.total_line_count)

    def _aposteriori_probability(self, l, f):
        # number of feature occurrences in label + 1 for smoothing devided by
        # number of lines with label + number of feature values for smoothing
        return log((self.label_feature_value_counts[l][f][True]+1)/(self.label_counts[l]+len(self.feature_values[f])))

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
            'tokens':  len(line.split()) < 6, # bobo
            'long_lines': len(line.split()) > 15,
            'areyouready': 'ready' in line, # bobo
            'know': 'know' in line, # mike
            # 'bobo': 'bobo' in line, # bobo
            '!': '!' in line,
            '?': '?' in line,
            'parenths': any(grunt in line for grunt in ['(',')']),
            # 'chihuaha': "chihuaha" in line,
            'nums': any(num in line for num in ['1', '2', '3', '4', '5', '6', '7', '8', '9']),
            'pronouns': any(pron in line for pron in ['i', 'me', 'we', 'you']),
            # 'colors': any(pron in line for pron in ['black', 'white']),
            # 'vocab': len(set(line.split()))/len(line.split()) < 0.20,

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
            if v == True:
                post_probs.append(self.feature_posterior_probabilities[label][f])
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
        pass
        f_dict = self._extract_features(line)
        b = self._probability('bobo', f_dict)
        m = self._probability('mike', f_dict)
        if b > m:
            return 'bobo'
        else:
            return 'mike'

    def evaluate(self, test_data: Dict[str, Iterable[str]]):
        """
        Evaluate the model and print recall, precision, accuracy and f1-score.

        Args:
            test_data: A dict mapping labels to iterables of lines
                       (e.g. a file object).
        """
        true_labels = []
        predictions = []
        for labels, lines in test_data.items():
            for line in lines:
                true_labels.append(labels)
                predictions.append(self.predict_label(line))

        zipped = zip(true_labels, predictions)
        tp = 0
        fp = 0
        for i in zipped:
            if i[1] != i[0]:
                fp += 1
            else:
                tp += 1

        accuracy = tp / len(true_labels)
        fn = 0
        prec, recall, f1 = self.calculate_eval_metrics(tp, fp, fn)
        print("Precision = {0:.2f}%; recall = {1:.2f}%; micro-average F1 = {2:.2f}%; accuracy = {3:.2f}% ".format(prec, recall*100, f1, accuracy))


    def calculate_eval_metrics(self, tp, fp, fn):
        prec = tp /(tp+fp)
        recall = tp / (tp+fn)
        try:
            f1 = 2*prec*recall / (prec+recall)
        except ZeroDivisionError:
            f1 = 0
        return prec, recall, f1
