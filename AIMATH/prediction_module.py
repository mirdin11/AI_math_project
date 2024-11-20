import random
import market as mk


class Prediction:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.expected_price = None
        self.evidence = None

    def predict(self, market):
        return {stock_name: price * random.uniform(0.9, 1.1) for stock_name, price in market.get_current_prices().items()}
    
class RandomPrediction(Prediction):
    def predict(self):
        return self.model.predict(self.data)

class LSTMPrediction(Prediction):
    def __init__(self, data):
        super().__init__(LSTMModel(), data)

class LLMPrediction(Prediction):
    def __init__(self, data):
        super().__init__(LLMModel(), data)

class OptimistPrediction(Prediction):
    def __init__(self, data):
        super().__init__(OptimistModel(), data)

class PessimistPrediction(Prediction):
    def __init__(self, data):
        super().__init__(PessimistModel(), data)

class GoldfishMemoryPrediction(Prediction):
    def __init__(self, data):
        super().__init__(GoldfishMemoryModel(), data)

