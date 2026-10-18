import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any
from pydantic import BaseModel, Field, field_validator, ValidationError

class BaseNormalizer(ABC):
    @abstractmethod
    def forward(self, scores: List[float]) -> List[float]:
        pass

    def __call__(self, scores: List[float]) -> List[float]:
        return self.forward(scores)

    @staticmethod
    def validate_scores(scores: List[float]) -> None:
        if not isinstance(scores, (list, tuple)):
            raise TypeError("scores must a List or Tuple")
        if len(scores) == 0:
            raise ValueError("scores don't must empty")
        
        for x in scores:
            if isinstance(x, str) or not isinstance(x, (int, float)):
                raise TypeError(f"x must a number")
            if math.isnan(x) or math.isinf(x):
                raise ValueError(f"x Error")

class Softmax(BaseNormalizer):
    
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature  
        
    @property
    def temperature(self) -> float:
        return self._temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        if value <= 0:
            raise ValueError("temperature must > 0.")
        self._temperature = value

    def forward(self, scores: List[float]) -> List[float]:
        self.validate_scores(scores)
        
        exps = [math.exp(x / self.temperature) for x in scores]
        sum_exps = sum(exps)
        
        return [e / sum_exps for e in exps]


class StableSoftmax(BaseNormalizer):
    
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    @property
    def temperature(self) -> float:
        return self._temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        if value <= 0:
            raise ValueError("temperature must > 0.")
        self._temperature = value

    def forward(self, scores: List[float]) -> List[float]:
        self.validate_scores(scores)
        
        m = max(scores)
        exps = [math.exp((x - m) / self.temperature) for x in scores]
        sum_exps = sum(exps)
        
        return [e / sum_exps for e in exps]

class SoftmaxConfig(BaseModel):
    temperature: float = Field(gt=0, allow_inf_nan=False)


class ScoresInput(BaseModel):
    scores: List[float] = Field(min_length=1)

    @field_validator('scores')
    @classmethod
    def check_finite_values(cls, v: List[float]) -> List[float]:
        for x in v:
            if math.isnan(x) or math.isinf(x):
                raise ValueError("Scores Contain INF and None")
        return v


@dataclass(frozen=True)
class InferenceReport:
    layer_name: str
    temperature: float
    probabilities: List[float]

    def report(self, scores: List[float]) -> 'InferenceReport':
        pass


def evaluate_all(layers: List[BaseNormalizer], scores: List[float]) -> List[InferenceReport]:
    reports = []
    for layer in layers:
        probs = layer(scores)
        layer_name = layer.__class__.__name__
        temp = getattr(layer, 'temperature', 1.0)
        
        report = InferenceReport(
            layer_name=layer_name,
            temperature=temp,
            probabilities=probs
        )
        reports.append(report)
    return reports

def from_config(cls, data: dict):
    config = SoftmaxConfig(**data)
    return cls(temperature=config.temperature)

Softmax.from_config = classmethod(from_config)
StableSoftmax.from_config = classmethod(from_config)


if __name__ == "__main__":
    print("--- Test cơ bản ---")
    x = [1.0, 2.0, 3.0]
    normal = Softmax(temperature=1.0)
    stable = StableSoftmax(temperature=1.0)

    print([round(v, 4) for v in normal(x)])
    
    print([round(v, 4) for v in stable(x)])
    print("\n--- Test tràn số ---")
    large = [1000.0, 1001.0, 1002.0]
    print([round(v, 4) for v in stable(large)])
    print(round(sum(stable(x)), 8))
    print("\n--- Test Validation ---")
    try:
        Softmax(temperature=0)
    except ValueError as e:
        print(f"ValueError : {e}")

    try:
        Softmax.from_config({"temperature": -1})
    except ValidationError as e:
        print(f"ValidationError (temperature > 0)")

    try:
        normal([])
    except ValueError as e:
        print(f"Data Empty: {e}")

    try:
        normal([1.0, float('nan')])
    except ValueError as e:
        print(f"Data NaN: {e}")
    print("\n--- Test tính độc lập ---")
    sm1 = Softmax(temperature=1.0)
    sm2 = Softmax(temperature=2.0)
    print(f"Temp 1.0: {[round(v, 4) for v in sm1(x)]}")
    print(f"Temp 2.0: {[round(v, 4) for v in sm2(x)]}")

    print("\n--- Test evaluate_all ---")
    layers = [normal, stable]
    reports = evaluate_all(layers, x)
    for r in reports:
        print(f"Layer: {r.layer_name} | Probabilities: {[round(p, 4) for p in r.probabilities]}")

    print("\n--- Overflow ---")
    try:
        normal(large)
    except OverflowError:
        print("Softmax will have OverFlow")