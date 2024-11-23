import random
import market as mk
import LLM_search as llm


class Prediction:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.expected_price = None
        self.evidence = None

    def predict(self, tickers, market):
        # dictionary of ticker and expected price
        expected_prices = {}
        #random prediction
        for ticker in tickers:
            # get the current price of the ticker
            current_price = market.get_current_price(ticker)
            # generate a random expected price
            expected_price = random.uniform(0.5, 1.5) * current_price
            # store the expected price in the dictionary
            expected_prices[ticker] = expected_price
        return expected_prices
    
    #risk factor
    #returns porfolio that maximizes the objective function
    def objective_func(self, risk_factor, profit_factor, predictions):
        raise NotImplementedError("objective_func() must be implemented by subclasses.")
    
class RandomPrediction(Prediction):
    def predict(self, tickers, market):
        return super().predict(self,tickers, market)

class LSTMPrediction(Prediction):
    def predict(self, tickers, market):
        return super().predict(self,tickers, market)

class LLMPrediction(Prediction):
    def predict(self, tickers, market):
        return super().predict(self,tickers, market)

class OptimistPrediction(Prediction):
    def predict(self, tickers, market):
        return super().predict(self,tickers, market)

class PessimistPrediction(Prediction):
    def predict(self, tickers, market):
        return super().predict(self,tickers, market)

class GoldfishMemoryPrediction(Prediction):
    def predict(self, tickers, market):
        return super().predict(self,tickers, market)

