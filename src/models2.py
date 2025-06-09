from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

from typing import Tuple, List, Dict, Any
import random
import math
import numpy as np
from sklearn.utils import resample



from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, List


# from resources.constants import Uncertainty
# from utils.uncertainty_utils import calculate_entropy_uncertainties

def calculate_entropy_uncertainties(labels: list, end_leafs: np.ndarray, leafs_split: List[Dict[int, List[int]]]):
    """
    Based on the paper Shaker MH, Hüllermeier E. Aleatoric and epistemic uncertainty with random forests. In International Symposium on
    Intelligent Data Analysis 2020 Apr 27 (pp. 444-456). Springer, Cham. (https://arxiv.org/pdf/2001.00893.pdf)
    Given a single sample x, we calculate three types of uncertainties:
    1. total
    2. aleatoric (statistical)
    3. epistemic (information related)

    1. This is the **total uncertainty estimation using entropy**.
        For discrete labels this is H[p(y | x)]=−∑y∈Y p(y | x) log2 p(y | x),
        An approximaton for ensemble techniques (and what we calculate here) is:
        −∑y∈Y (1/M * ∑i∈M p(y | hi, x) log2 (1/M * ∑i∈M p(y | hi, x))
    2. The aleatoric uncertainty can be estimated by:
        −(1/M * ∑i∈M ∑y∈Yp(y | hi, x) log2 p(y | hi, x)
    3. The epistemic uncertainty, which is the subtraction of total − aleatoric
    :param labels: a list with all the possible labels
    :param end_leafs: a list with all the leafs our sample ends up in, one per each tree in the ensemble.
    :param leafs_split: a summary of training samples ended up in each leaf and their split between classes. This is a list of dictionaries
    the length of all trees. Each dictionary points from leaf number to a list [n_neg, n_pos] such that n_neg is the number of negative
    samples in this leaf and n_pos is the number of positive samples in this leaf. Σ (n_neg+n_pos) in each dict should equal to
    X_train.shape[0].
    :return: A named tuple with the three uncertainties calculated
    """
    n_labels = len(labels)
    tot_u = 0  # total uncertainty
    al_u = 0  # aleatoric uncertainty
    for label in labels:  # go over the labels
        tot_p = 0  # total uncertainty
        tot_p_log_p = 0  # helper for aleatoric uncertainty
        for tree_leafs_split, end_leaf in zip(leafs_split, end_leafs):  # go over all the hypotheses (trees)
            # We first want to calculate p(y | hi, x) for each tree ('hi'), based on the leaf where each sample ends up. In random forest
            # this is the (n_(i,j)(y) + 1) / (n_(i,j) + |Y|), where n_(i,j) are the number of samples in tree i, leaf j and n_(i,j)(y) are
            # the number of samples in tree i, leaf j with label y
            p = _calculate_class_conditional_probabilities(label, n_labels, end_leaf, tree_leafs_split)
            tot_p += p
            tot_p_log_p += p * np.log2(p)

        # Total uncertainty for label i:
        mean_tot_p = tot_p / len(end_leafs)  # get the average over all trees
        log_mean_tot_p = np.log2(mean_tot_p)
        tot_u -= mean_tot_p * log_mean_tot_p

        # Aleatoric uncertainty for label i:
        mean_tot_p_log_p = tot_p_log_p / len(end_leafs)  # get the average over all trees
        al_u -= mean_tot_p_log_p
    return (tot_u, al_u, tot_u - al_u)


def _calculate_class_conditional_probabilities(label, n_labels, end_leaf, tree_leafs_split) -> float:
    """
    We calculate p(y | hi, x) for a given label y and a specific model(tree).
    :param label: label number∈[0,1,2,...], corresponds to the index in leaf_split to recover the number of training sample in a leaf
    with this label
    :param n_labels: total number of possible labels (i.e. for binary this would be 2)
    """
    print("end_leaf", end_leaf,"label",label)
    n_y = tree_leafs_split[end_leaf][label]
    n = sum(tree_leafs_split[end_leaf])
    return (n_y + 1) / (n + n_labels)

class RFWithUncertainty(RandomForestClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.leafs_content = None  # the split of training samples between the leafs of all decision trees within the forest
        self._labels = None
        self.used_features = None

    def fit(self, X, y, sample_weight=None):
        super().fit(X, y, sample_weight=sample_weight)
        print("y_values",y.unique().tolist())
        # produce a map of the counts of labels in each leaf
        self.leafs_content = self._binary_leaf_split_counter(X, y)
        # all possible labels:
        self._labels = list(set(y)) #y.unique().tolist()

        self._labels.sort() 
        
        # summarize the features used in the trees:
        self.used_features = self._output_used_features(X)

    def predict_with_uncertainty(self, X_test):
        predictions = self.predict(X_test)
        end_leafs = self.apply(X_test)
        uncertainties = self._extract_uncertainty_of_prediction(end_leafs, method='entropy')
        return predictions, uncertainties

    def predict_proba_with_uncertainty(self, X_test):
        predictions = self.predict_proba_1d(X_test)
        end_leafs = self.apply(X_test)
        uncertainties = self._extract_uncertainty_of_prediction(end_leafs, method='entropy')
        return uncertainties

    def predict_proba_1d(self, x: pd.DataFrame) -> np.ndarray:
        res = self.predict_proba(x)
        if isinstance(res, pd.Series):  # already what we need
            return res
        if isinstance(res, list):
            res = np.asarray(res)
        res = res.squeeze()
        if len(res.shape) != 1:  # if it has 1 dim, then it is already in the format we need, just not as a series yet
            if len(x) > 1:  # there is more than a single sample to predict on, but gave 2 predictions for each sample (P(0), P(1))
                assert len(res.shape) == 2 and res.shape[1] == 2, f'Invalid result shape for binary classification: {res.shape}'
                res = res[:, 1]
            elif len(res.shape) == 0:  # returned a single value
                pass
            else:
                res = res[:, 1]
        elif len(x) == 1 and len(res) == 2:  # a single sample, but two predictions (P(0), P(1))
            res = res[1]
        predictions = res
        return predictions

    def _output_used_features(self, X_train) -> OrderedDict:
        """
        Go through all the trees and sum up the usage of each feature. Then summarize it in a sorted descending dictionary, from feature
        name to count.
        :param X_train: dataframe with the training data, used for feature names
        :return: oredered dictionary from feature name to usage count
        """
        feature_names = list(X_train.columns)
        features_count = {key: 0 for key in feature_names}
        for estimator in self.estimators_:
            tree_features = np.where(estimator.feature_importances_)[0]
            for n in tree_features:
                features_count[feature_names[n]] += 1
        features_count = {key: value for key, value in features_count.items() if value > 0}
        sorted_x = sorted(features_count.items(), key=lambda kv: kv[1], reverse=True)
        return OrderedDict(sorted_x)

    def _binary_leaf_split_counter(self, X_train, y_train) -> List[Dict[int, List[int]]]:
        """
        A method to count the number of training data samples that end up in each node and the split between the classes.
        Note that this method is only valid for binary case.
        We summarize the results per each tree in a separate dictionary, such that len(output) == num_trees.
        Each dictionary points from leaf number to a list [n_neg, n_pos] such that n_neg is the number of negative samples in this leaf
        and n_pos is the number of positive samples in this leaf. Σ (n_neg+n_pos) in each dict should equal to X_train.shape[0].
        :return: list of dictionaries the length of all trees
        """
        def _summarize_into_dict(r):
            unique_nodes, counts = np.unique(r, return_counts=True)
            d = {k: [0, 0] for k in list(set((abs(unique_nodes))))}
            for node, count in zip(unique_nodes, counts):
                s = (np.sign(node) > 0).astype(int)
                d[abs(node)][s] = count
            return d
        leaves_index = self.apply(X_train)
        f = lambda x: [-1, 1][x]
        y_train_ = np.expand_dims(np.vectorize(f)(y_train.values.astype(int)),  axis=1)  # map False to -1 and adjust dimensions
        leaves_index_with_signs = np.multiply(leaves_index, y_train_)  # multiply with labels to sum the different classes
        return np.apply_along_axis(_summarize_into_dict, 0, leaves_index_with_signs)

        # leaves_index = self.apply(X_train)  # shape: [n_samples, n_trees]
        # leafs_content = []

        # for tree_id in range(leaves_index.shape[1]):
        #     tree_leafs = leaves_index[:, tree_id]
        #     tree_dict = {}

        #     for leaf, label in zip(tree_leafs, y_train.values):
        #         label = int(label)
        #         if leaf not in tree_dict:
        #             tree_dict[leaf] = [0] * 2
        #         tree_dict[leaf][label] += 1

        #     leafs_content.append(tree_dict)

        # return leafs_content

    def _extract_uncertainty_of_prediction(self, end_leafs, method='entropy'):
        """
        Using the method specified calculate the uncertainty of a prediction that was made
        :param method: Currently we support "entropy" method only
        :return: list of uncertainty objects, each per sample.
        """
        uncertainty = []
        if method == 'entropy':
            for row in end_leafs:  # each row is the result for one sample
                uncertainty.append(calculate_entropy_uncertainties(self._labels, row, self.leafs_content))
        else:
            raise NotImplementedError
        return uncertainty

class UncertForest:
    def __init__(self, start: int, stop : int,batch_per: float = 0):
        self.model = RFWithUncertainty(bootstrap = True, n_estimators=30, max_depth=6)
        self.batch_per = batch_per
        self.scores = []
        self.done: pd.DataFrame
        self.todo: pd.DataFrame
        self.test_set = []
        self.start = start
        self.stop = stop
        self.columns= []

    def fit(self, train: pd.DataFrame, test: pd.DataFrame) -> None:
        
        self.columns = train.columns.tolist()
        self.test_set = test.values.tolist()
        self.batch_size = max(1, int(len(train) * self.batch_per))
        
        shuffled = train.values.tolist()
 
        random.shuffle(shuffled)
      
        self.done = shuffled[:self.start]
        self.todo = shuffled[self.start:]
        

        

        count = 0

        while self.todo and len(self.done) <= self.stop:
            # Train model on current done set
            X_done = pd.DataFrame([row[:-1] for row in self.done], columns=self.columns[:-1])
            y_done = pd.Series([row[-1] for row in self.done],dtype=int)

            

            self.model.fit(X_done, y_done)

            # Get most uncertain sample
            
            most_uncertain, *self.todo = self.calculate_uncertainty()
            self.done += [most_uncertain]

            count += 1
            if count % self.batch_size == 0 or not self.todo:
                self.scores.append(self.get_scores(self.test_set))

    def calculate_uncertainty(self) -> List[Tuple[List[float], Any]]:
        if not self.todo:
            return []

        X_todo = pd.DataFrame([x[:-1] for x in self.todo],columns=self.columns[:-1])
        #print(X_todo.info())
        uncertainties = self.model.predict_proba_with_uncertainty(X_todo)

        # Ensure uncertainties are structured correctly
        # Each item should be (total, aleatoric, epistemic)
        scored = sorted(
            zip((u[2] for u in uncertainties), self.todo),
            key=lambda tup: tup[0], reverse=True  # sort by epistemic uncertainty
        )

        return [sample for _, sample in scored]

    def get_scores(self, test: List[Tuple[List[float], Any]]) -> Dict[str, float]:
        X_test = pd.DataFrame([row[:-1] for row in test], columns=self.columns[:-1])
        y_test = pd.Series([row[-1] for row in test],dtype=int)
        y_pred, uncertainty = self.model.predict_with_uncertainty(X_test)

        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred, average= 'macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        cm = confusion_matrix(y_test, y_pred)
        if cm.shape == (2, 2):
            fp_rate = cm[0][1] / max((cm[0][0] + cm[0][1]), 1)
        else:
            fp_rate = float('nan')

        return {
            'accuracy': acc,
            'recall': rec,
            'f1': f1,
            'false_positive_rate': fp_rate
        }

    def get_model(self) -> Any:
        return self.model

    def get_all_scores(self) -> List[Dict[str, float]]:
        return self.scores

