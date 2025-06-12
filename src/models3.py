from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
import random
import math
import numpy as np
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any


class UncertNB():
    """
    A Naive Bayes model for classification tasks.
    
    Inherits from sklearn's GaussianNB and can be used for training and predicting
    with Gaussian Naive Bayes.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_0 = GaussianNB()
        self.model_1 = GaussianNB()
        
    
    def fit(self, X, y):
        """
        Fit the model to the training data.
        
        Parameters:
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target values.
        """

        class_0 = X[y == 0]
        class_1 = X[y == 1]
        
        if not class_0 or not_class_1:
            raise ValueError("Training data must contain both classes 0 and 1.")

        # Fit seperate model for each class
        self.model_0.fit(class_0, [0] * len(class_0))
        self.model_1.fit(class_1, [1] * len(class_1))
    
    def predict_proba_with_uncertainty(self, X):
        """
        Predict class probabilities for samples in X.

        Parameters:
        X : array-like, shape (n_samples, n_features)
            Samples to predict.
        
        Returns:
        y_pred : array, shape (n_samples,)
            Predicted class labels.
        """
        proba_0 = self.model_0.predict_proba(X)
        proba_1 = self.model_1.predict_proba(X)
        eps = 1e-12  # to avoid log(0)
        B = np.log(proba_0[:, 0] + eps)  # log-likelihood under class 0 model
        R = np.log(proba_1[:, 1] + eps)  # log-likelihood under class 1 model
        y_pred = (proba_1[:, 1] > proba_0[:, 1]).astype(int)

        epistemic = lambda B,R : 1 - max(B,R)
        #aleatoric = lambda B,R : 1 - 

        uncertainties = np.array([epistemic(b, r) for b, r in zip(B, R)])
        return y_pred, uncertainties

class NBActiveLearner():
    """
    An active learner using Naive Bayes for classification tasks.
    
    This class implements an active learning strategy using Gaussian Naive Bayes.
    It allows for incremental training and uncertainty-based sample selection.
    """

    def __init__(self, start: int, stop : int,batch_per: float = 0):
        self.model = UncertNB()
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

            
            try:
                self.model.fit(X_done, y_done)
            
            except Exception as e:
                print(e)
                break

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


    
    