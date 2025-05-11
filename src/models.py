from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any
import random
import math

class Model(ABC):
    """
    Abstract base class for machine learning models with uncertainty propagation support.
    """

    def __init__(self):
        pass

    @abstractmethod
    def fit(self, train, test) -> None:
        pass

    @abstractmethod
    def get_scores(self, test) -> Dict[str, float]:
        pass

    @abstractmethod
    def calculate_uncertainty(self) -> List[Tuple[List[float], Any]]:
        pass

    @abstractmethod
    def get_model(self) -> Any:
        pass


class NaiveBayesModel(Model):
    def __init__(self, batch_per: float = 0):
        super().__init__()
        self.model = GaussianNB()
        self.batch_per = batch_per
        self.scores = []
        self.done: List[Tuple[List[float], Any]] = []
        self.todo: List[Tuple[List[float], Any]] = []
        self.test_set = []

    def fit(self, train: List[Tuple[List[float], Any]], test: List[Tuple[List[float], Any]]) -> None:
        self.test_set = test
        self.batch_size = max(1, int(len(train) * self.batch_per))
        shuffled = train.copy()
        random.shuffle(shuffled)
        self.done = shuffled[:16]
        self.todo = shuffled[16:]

        count = 0

        while self.todo:
            # Train model on current done set
            X_done = [x for x, _ in self.done]
            y_done = [y for _, y in self.done]
            self.model.fit(X_done, y_done)

            # Get most uncertain sample
            most_uncertain, *self.todo = self.calculate_uncertainty()
            self.done += [most_uncertain]

            count += 1
            if count % self.batch_size == 0 or not self.todo:
                self.scores.append(self.get_scores(self.test_set))

    def calculate_uncertainty(self) -> List[Tuple[List[float], Any]]:
        """
        Rank remaining TODO samples by uncertainty (entropy of predicted probabilities).
        """
        if not self.todo:
            return []

        X_todo = [x for x, _ in self.todo]
        probs = self.model.predict_proba(X_todo)

        entropies = [-sum(p * math.log(p + 1e-9) for p in prob) for prob in probs]
        scored = list(zip(entropies, self.todo))
        scored.sort(reverse=True, key=lambda tup: tup[0])  # High entropy = high uncertainty

        return [sample for _, sample in scored]

    def get_scores(self, test: List[Tuple[List[float], Any]]) -> Dict[str, float]:
        X_test = [x for x, _ in test]
        y_test = [y for _, y in test]
        y_pred = self.model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred, average='macro')
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
