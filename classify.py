from collections import defaultdict
from typing import Dict, Iterable, List
from math import log


class LyricsClassifier:
    """Implements a Naïve Bayes model for song lyrics classification."""

    def __init__(self, train_data: Dict[str, Iterable[str]]):
        """Initialize and train the model.

        Args:
            train_data: A dict mapping labels to iterables of lines of
                        (e.g. a file object).
        """
        label_counts = {}
        label_feature_value_counts = {}
        self.line_count = 0
        # store the different seen values for each feature (across the entire
        # training set, independent of label), needed for smoothing
        feature_values = defaultdict(lambda: set())

        for label, lines in train_data.items():
            # calculate raw counts given this label
            label_count = 0
            feature_value_counts = defaultdict(lambda: defaultdict(int))
            for line in lines:
                self.line_count += 1
                # print()
                # print(line.strip())
                # we look at every line and count how many bobos and mikes
                label_count += 1
                # for every line we check if feature is present, count
                features = self._extract_features(line)
                for feature_id, value in features.items():
                    # feature_id is name of feature
                    # value is True or False; number of what we count
                    # for particular label we count features and count them
                    # we enter info into dict
                    feature_value_counts[feature_id][value] += 1
                    # for i, v in feature_value_counts.items():
                    #     print(i, v)
                    # for every feature regardless of label update its set of values
                    feature_values[feature_id].add(value)
                    # for k, v in feature_values.items():
                    #     print(k, v)
            label_counts[label] = label_count
            # a dict: for every label we get feature and its counts
            label_feature_value_counts[label] = feature_value_counts
        # print(label_counts)
        # print(len(label_feature_value_counts))
        # for key, value in label_feature_value_counts.items():
        #     print(key, len(value))
        #     for k, i in value.items():
        #         print(key, k, len(i))
        for label, count in label_counts.items():
            # total += count
            print(self._apriori_probability(count))

        print(self.line_count)

        #TODO calculate probabilities
        # yNB = max(_apriori_probability(label)) + sum(P(feature, label)))

    def _apriori_probability(self, N):
        return N/self.line_count

    def _aposteriori_probability(self, f, l):
        pass


    @staticmethod
    def _extract_features(line: str) -> Dict:
        """Return a dict containing features values extracted from a line.

        Args:
            line: A line of song lyrics.

        Returns:
            A dict mapping feature IDs to feature values.
        """
        #TODO find better features
        return {
            'pray':    'pray' in line,
            'grunts':  any(grunt in line for grunt in ['oh', 'ah', 'uh']),
            'tokens':  len(line.split()),
            'charset': len(set(line)) // 4,
        }

    def _probability(self, label: str, features: Dict) -> float:
        """Return P(label|features) using the Naïve Bayes assumption.

        Args:
            label: The label.
            features: A dict mapping feature IDs to feature values.

        Returns:
            The non-logarithmic probability of `label` given `features`.
        """
        #TODO implement
        raise NotImplementedError()

    def predict_label(self, line: str) -> str:
        """Return the most probable prediction by the model.

        Args:
            line: The line to classify.

        Returns:
            The most probable label according to Naïve Bayes.
        """
        #TODO implement
        raise NotImplementedError()

    def evaluate(self, test_data: Dict[str, Iterable[str]]):
        """Evaluate the model and print recall, precision, accuracy and f1-score.

        Args:
            test_data: A dict mapping labels to iterables of lines
                       (e.g. a file object).
        """
        #TODO implement
        raise NotImplementedError()
