from abc import ABC, abstractmethod


class EvaluationMetric(ABC):
    @abstractmethod
    def evaluate(self, actual_data, predicted_data):
        pass
