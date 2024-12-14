import LLM_search as llm

class Action:
    def __init__(self, action_type, stock_name, quantity=0):
        self.action_type = action_type  # 'buy', 'sell', or 'hold'
        self.stock_name = stock_name
        self.quantity = quantity


class Strategy:
    def decide_actions(self, predictions, curr_prices, portfolio, curr_balance, curr_date, obj_date, market):
        raise NotImplementedError("decide_actions() must be implemented by subclasses.")
    
    #risk factor
    def objective_func(self, risk_factor, profit_factor):
        raise NotImplementedError("objective_func() must be implemented by subclasses.")


class ScaredyCatStrategy(Strategy):
    """A strategy that sells all holdings immediately."""

    def decide_actions(self, predictions, curr_prices, portfolio, curr_balance, curr_date, obj_date, market):
        actions = []
        for stock in portfolio:
            if stock.quantity > 0:
                actions.append(Action('sell', stock.name, stock.quantity))
        return actions


class GoldfishMemoryStrategy(Strategy):
    """A strategy that only considers the most recent predictions."""

    def decide_actions(self, predictions, curr_prices, portfolio, curr_balance, curr_date, obj_date, market):
        actions = []
        for stock_name, current_price in curr_prices.items():
            predicted_price = predictions.get(stock_name, current_price)
            if predicted_price > current_price:
                max_shares = int(curr_balance // current_price)
                if max_shares > 0:
                    actions.append(Action('buy', stock_name, max_shares))
            elif predicted_price < current_price:
                stock = self.get_stock_from_portfolio(portfolio, stock_name)
                if stock and stock.quantity > 0:
                    actions.append(Action('sell', stock_name, stock.quantity))
            else:
                actions.append(Action('hold', stock_name))
        return actions

    @staticmethod
    def get_stock_from_portfolio(portfolio, stock_name):
        for stock in portfolio:
            if stock.name == stock_name:
                return stock
        return None
    
class OptimisticStrategy(Strategy):
    """A strategy that buys all stocks."""

    def decide_actions(self, predictions, curr_prices, portfolio, curr_balance, curr_date, obj_date, market):
        actions = []
        for stock_name, current_price in curr_prices.items():
            max_shares = int(curr_balance // current_price)
            if max_shares > 0:
                actions.append(Action('buy', stock_name, max_shares))
        return actions

class DeepStateStrategy(Strategy):
    """A strategy that buys all stocks."""

    def decide_actions(self, predictions, curr_prices, portfolio, curr_balance, curr_date, obj_date, market):
        actions = []
        for stock_name, current_price in curr_prices.items():
            max_shares = int(curr_balance // current_price)
            if max_shares > 0:
                actions.append(Action('buy', stock_name, max_shares))
        return actions

class DemocraticStrategy(Strategy):
    """A strategy that buys all stocks."""
    """total 10 people, 3people are optimistic, 3people are deepstate, 4people are scaredycat"""
    """people dynamically change their strategy"""
    def decide_actions(self, predictions, curr_prices, portfolio, curr_balance, curr_date, obj_date, market):
        actions = []
        for stock_name, current_price in curr_prices.items():
            max_shares = int(curr_balance // current_price)
            if max_shares > 0:
                actions.append(Action('buy', stock_name, max_shares))
        return actions

class ConspiracyTheoryStrategy(Strategy):
    """A strategy that buys all stocks."""

    def decide_actions(self, predictions, curr_prices, portfolio, curr_balance, curr_date, obj_date, market):
        actions = []
        for stock_name, current_price in curr_prices.items():
            max_shares = int(curr_balance // current_price)
            if max_shares > 0:
                actions.append(Action('buy', stock_name, max_shares))
        return actions

class GamblerStrategy(Strategy):
    """buys stocks with high risk"""
    def decide_actions(self, predictions, curr_prices, portfolio, curr_balance, curr_date, obj_date, market):
        actions = []
        for stock_name, current_price in curr_prices.items():
            max_shares = int(curr_balance // current_price)
            if max_shares > 0:
                actions.append(Action('buy', stock_name, max_shares))
        return actions
    
class NewsReaderStrategy(Strategy):
    """GPT analysis"""
    def __init__(self):
        super().__init__()
        self.model = llm.init_llm()
    def decide_actions(self, predictions, curr_prices, portfolio, curr_balance, curr_date, obj_date, market):
        actions = []
        stock_names = []
        for stock in market.stocks.keys():
            stock_names.append(stock)
        predictions = llm.ask_llm(self.model, stock_names, curr_date, obj_date)
        #ex) prediction = "stockA:up/down, 0.5\nstockBup/down, 0.5\n"
        stock_list =[]
        for stock in predictions.split("\n"):
            stock_list.append(stock)
        updownDict = {}
        for stock in stock_list:
            splited = stock.split(":")
            if len(splited) <= 1:
                continue
            updownDict[splited[0]] = splited[1]
        for stock_name, current_price in curr_prices.items():
            up_down = updownDict[stock_name].split(",")[0]
            intensity = float(updownDict[stock_name].split(",")[1])
            if 'up' in up_down:
                max_shares = int(curr_balance // current_price)
                if max_shares > 0:
                    actions.append(Action('buy', stock_name, max_shares* intensity))
            elif 'down' in up_down:
                for stock in portfolio:
                    if stock.name == stock_name and stock.quantity > 0:
                        actions.append(Action('sell', stock_name, stock.quantity* intensity))
        return actions

class NewsReaderStrategy_Hidden(Strategy):
    """GPT analysis"""
    def __init__(self):
        super().__init__()
        self.model = llm.init_llm()
    def decide_actions(self, predictions, curr_prices, portfolio, curr_balance, curr_date, obj_date, market):
        actions = []
        stock_names = []
        for stock in market.stocks.keys():
            stock_names.append(stock)
        predictions = llm.ask_llm_info_hidden(self.model, stock_names, curr_prices)
        #ex) prediction = "up/down,
        stock_list =[]
        for stock in predictions.split("\n"):
            stock_list.append(stock)
        updownDict = {}
        for stock in stock_list:
            splited = stock.split(":")
            if len(splited) <= 1:
                continue
            updownDict[splited[0]] = splited[1]
        for stock_name, current_price in curr_prices.items():
            up_down = updownDict[stock_name].split(",")[0]
            intensity = float(updownDict[stock_name].split(",")[1])
            if 'up' in up_down:
                max_shares = int(curr_balance // current_price)
                if max_shares > 0:
                    actions.append(Action('buy', stock_name, max_shares* intensity))
            elif 'down' in up_down:
                for stock in portfolio:
                    if stock.name == stock_name and stock.quantity > 0:
                        actions.append(Action('sell', stock_name, stock.quantity* intensity))
        return actions
    
class PoliticianStrategy(Strategy):
    """A strategy that buys all stocks."""

    def decide_actions(self, predictions, curr_prices, portfolio, curr_balance):
        actions = []
        for stock_name, current_price in curr_prices.items():
            max_shares = int(curr_balance // current_price)
            if max_shares > 0:
                actions.append(Action('buy', stock_name, max_shares))
        return actions

class UnbelieverStrategy(Strategy):
    """Price is irellevent to past price"""

    def decide_actions(self, predictions, curr_prices, portfolio, curr_balance):
        actions = []
        for stock_name, current_price in curr_prices.items():
            max_shares = int(curr_balance // current_price)
            if max_shares > 0:
                actions.append(Action('buy', stock_name, max_shares))
        return actions

class IndexfundStrategy(Strategy):
    """A strategy that buys all stocks."""

    def decide_actions(self, predictions, curr_prices, portfolio, curr_balance):
        actions = []
        for stock_name, current_price in curr_prices.items():
            max_shares = int(curr_balance // current_price)
            if max_shares > 0:
                actions.append(Action('buy', stock_name, max_shares))
        return actions