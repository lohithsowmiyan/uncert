from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any
import random
import math
import numpy as np
from sklearn.utils import resample

class Model(ABC):
    """
    Abstract base class for machine learning models with uncertainty propagation support.
    """

    def __init__(self):
        pass

    @abstractmethod
    def fit(self, train, test, sample) -> None:
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


class NaiveBayesModel():
    def __init__(self, start: int, stop : int, batch_per: float = 0):
        #super().__init__()
        self.model = GaussianNB()
        self.batch_per = batch_per
        self.scores = []
        self.done: List[Tuple[List[float], Any]] = []
        self.todo: List[Tuple[List[float], Any]] = []
        self.test_set = []
        self.stop = stop
        self.start = start


    def fit(self, train: List[Tuple[List[float], Any]], test: List[Tuple[List[float], Any]], sample = 'uncertainity') -> None:
        self.test_set = test
        self.batch_size = max(1, int(len(train) * self.batch_per))
        shuffled = train.copy()
        random.shuffle(shuffled)
        self.done = shuffled[:self.start]
        self.todo = shuffled[self.start:]

        count = 0

        while self.todo and len(self.done) <= self.stop:
            # Train model on current done set
            X_done = [x for x, _ in self.done]
            y_done = [y for _, y in self.done]
            self.model.fit(X_done, y_done)

            # Get most uncertain sample
            if sample == 'entropy':
                most_uncertain, *self.todo = self.ent()
            else:
                most_uncertain, *self.todo = self.todo

            self.done += [most_uncertain]

            count += 1
            if count % self.batch_size == 0 or not self.todo:
                self.scores.append(self.get_scores(self.test_set))

    def ent(self) -> List[Tuple[List[float], Any]]:
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
    
    def fit_2(self, train: List[Tuple[List[float], Any]], test: List[Tuple[List[float], Any]], uncertainty = 'epistemic'):
        """
        Rank remaining TODO samples by uncertainty (variance in likelihood of predictions).
        """
        def train_models_with_perturbations(X, y, n_models=20, seed=42):

            rng = np.random.RandomState(seed)
            models = []
            log_likelihoods = []

            for _ in range(n_models):
                X_resampled, y_resampled = resample(X, y, random_state=rng)
                model = GaussianNB()
                model.fit(X_resampled, y_resampled)

                log_proba = model.predict_log_proba(X_resampled)
                class_indices = {cls: idx for idx, cls in enumerate(model.classes_)}

                # Compute total log-likelihood safely
                log_likelihood = np.sum([
                    log_proba[i, class_indices[y_resampled[i]]] for i in range(len(y_resampled))
                ])

                # Append as scalar float
                log_likelihoods.append(float(log_likelihood))

                models.append(model)

            max_log_likelihood = max(log_likelihoods)
            log_likelihoods = np.array(log_likelihoods)
            pi_theta = np.exp(log_likelihoods - max_log_likelihood)  # normalized

            return models, pi_theta
        
        f = lambda a : 2 * a -1

        def compute_plausibility(models, pi_theta, x_instance):
            """
            Computes π(1|x) and π(0|x) for a given instance x.
            """
            min_1 = []
            min_0 = []

            for model, pi in zip(models, pi_theta):
                prob = model.predict_proba([x_instance])[0]
                if len(model.classes_) == 2:
                    idx0 = np.where(model.classes_ == 0)[0][0] if 0 in model.classes_ else None
                    idx1 = np.where(model.classes_ == 1)[0][0] if 1 in model.classes_ else None
                    p0 = prob[idx0] if idx0 is not None else 0.0
                    p1 = prob[idx1] if idx1 is not None else 0.0
                else:
                    # Only one class seen during training
                    if model.classes_[0] == 0:
                        p0, p1 = prob[0], 0.0
                    else:
                        p0, p1 = 0.0, prob[0]

                min_1.append(min(pi, p1))
                min_0.append(min(pi, p0))

            pi_1 = max(min_1)
            pi_0 = max(min_0)
            return pi_1, pi_0
        
        def compute_uncertainties(models, pi_theta):
            """
            Computes epistemic and aleatoric uncertainties for all instances in X_test.
            """
            epistemic = []
            aleatoric = []

            for x, y in self.todo:
                pi1, pi0 = compute_plausibility(models, pi_theta, x)
                ue = min(pi1, pi0)
                ua = 1 - max(pi1, pi0)
                epistemic.append((ue, (x, y)))
                aleatoric.append((ua, (x, y)))

            # Sort in descending order by uncertainty
            epistemic_sorted = sorted(epistemic, key=lambda t: t[0], reverse=True)
            aleatoric_sorted = sorted(aleatoric, key=lambda t: t[0], reverse=True)

            return epistemic_sorted, aleatoric_sorted
        # Train ensemble of GNB models

        self.test_set = test
        self.batch_size = max(1, int(len(train) * self.batch_per))
        shuffled = train.copy()
        random.shuffle(shuffled)
        self.done = shuffled[:self.start]
        self.todo = shuffled[self.start:]

        count = 0
        
        while self.todo and len(self.done) <= self.stop:
            # Train model on current done set
            X_done = [x for x, _ in self.done]
            y_done = [y for _, y in self.done]
            #self.model.fit(X_done, y_done)
            models, pi_theta = train_models_with_perturbations(X_done, y_done, n_models=30)

            # Get most uncertain sample
            
            epistemic_unc, aleatoric_unc = compute_uncertainties(models, pi_theta)
            if uncertainty == 'epistemic':
                most_uncertain, *self.todo = epistemic_unc
            else:
                most_uncertain, *self.todo = aleatoric_unc

            self.done += [most_uncertain]

            count += 1
            scores = []
            if count % self.batch_size == 0 or not self.todo:
                for model in models:
                    self.model = model
                    scores.append(self.get_scores(self.test_set))
                    
            # Compute average of each metric
            avg_score = {
                "accuracy": np.mean([s["accuracy"] for s in scores]),
                "recall": np.mean([s["recall"] for s in scores]),
                "f1": np.mean([s["f1"] for s in scores]),
                "false_positive_rate": np.mean([s["false_positive_rate"] for s in scores])
            }

            self.scores.append(avg_score) 
            

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
    

from sklearn.linear_model import LogisticRegression


class LogisticRegressionModel:
    def __init__(self,start: int, stop : int, batch_per: float = 0):
        self.model = LogisticRegression(max_iter=1000)
        self.batch_per = batch_per
        self.scores = []
        self.done: List[Tuple[List[float], Any]] = []
        self.todo: List[Tuple[List[float], Any]] = []
        self.test_set = []
        self.start = start
        self.stop = stop

    def fit(self, train: List[Tuple[List[float], Any]], test: List[Tuple[List[float], Any]],sample = 'uncertainty') -> None:
        self.test_set = test
        self.batch_size = max(1, int(len(train) * self.batch_per))
        shuffled = train.copy()
        random.shuffle(shuffled)
        self.done = shuffled[:self.start]
        self.todo = shuffled[self.start:]

        count = 0

        while self.todo and len(self.done) <= self.stop:
            # Train model on current done set
            X_done = [x for x, _ in self.done]
            y_done = [y for _, y in self.done]
            self.model.fit(X_done, y_done)

            # Get most uncertain sample
            if sample == 'uncertainty':
                most_uncertain, *self.todo = self.calculate_uncertainty()
            else:
                most_uncertain, *self.todo = self.todo
            self.done += [most_uncertain]

            count += 1
            if count % self.batch_size == 0 or not self.todo:
                self.scores.append(self.get_scores(self.test_set))

    def calculate_uncertainty(self) -> List[Tuple[List[float], Any]]:
        if not self.todo:
            return []

        X_todo = [x for x, _ in self.todo]
        probs = self.model.predict_proba(X_todo)

        entropies = [-sum(p * math.log(p + 1e-9) for p in prob) for prob in probs]
        scored = list(zip(entropies, self.todo))
        scored.sort(reverse=True, key=lambda tup: tup[0])  # High entropy = high uncertainty
        print(scored)
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
    
from sklearn.ensemble import RandomForestClassifier


class RandomForestClassifierModel:
    def __init__(self, start: int, stop : int,batch_per: float = 0):
        self.model = RandomForestClassifier(n_estimators=100)
        self.batch_per = batch_per
        self.scores = []
        self.done: List[Tuple[List[float], Any]] = []
        self.todo: List[Tuple[List[float], Any]] = []
        self.test_set = []
        self.start = start
        self.stop = stop

    def fit(self, train: List[Tuple[List[float], Any]], test: List[Tuple[List[float], Any]], sample : str = "uncertainty") -> None:
        self.test_set = test
        self.batch_size = max(1, int(len(train) * self.batch_per))
        shuffled = train.copy()
        random.shuffle(shuffled)
        self.done = shuffled[:self.start]
        self.todo = shuffled[self.start:]

        count = 0

        while self.todo and len(self.done) <= self.stop:
            # Train model on current done set
            X_done = [x for x, _ in self.done]
            y_done = [y for _, y in self.done]
            self.model.fit(X_done, y_done)

            # Get most uncertain sample
            
            most_uncertain, *self.todo = self.calculate_uncertainty() if sample == 'uncertainity' else self.todo
            self.done += [most_uncertain]

            count += 1
            if count % self.batch_size == 0 or not self.todo:
                self.scores.append(self.get_scores(self.test_set))

    def calculate_uncertainty(self) -> List[Tuple[List[float], Any]]:
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

