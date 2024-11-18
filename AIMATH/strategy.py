import random
import market as mk
import LLM_search as llm


class Stock:
    def __init__(self, name, quantity):
        self.name = name
        self.quantity = quantity


class Action:
    def __init__(self, action_type, stock_name, quantity=0):
        self.action_type = action_type  # 'buy', 'sell', or 'hold'
        self.stock_name = stock_name
        self.quantity = quantity


class Strategy:
    def decide_actions(self, predictions, curr_prices, portfolio, curr_balance, curr_date, obj_date):
        raise NotImplementedError("decide_actions() must be implemented by subclasses.")
    
    def objective_func(self):
        raise NotImplementedError("objective_func() must be implemented by subclasses.")


class ScaredyCatStrategy(Strategy):
    """A strategy that sells all holdings immediately."""

    def decide_actions(self, predictions, curr_prices, portfolio, curr_balance, curr_date, obj_date):
        actions = []
        for stock in portfolio:
            if stock.quantity > 0:
                actions.append(Action('sell', stock.name, stock.quantity))
        return actions


class GoldfishMemoryStrategy(Strategy):
    """A strategy that only considers the most recent predictions."""

    def decide_actions(self, predictions, curr_prices, portfolio, curr_balance, curr_date, obj_date):
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

    def decide_actions(self, predictions, curr_prices, portfolio, curr_balance, curr_date, obj_date):
        actions = []
        for stock_name, current_price in curr_prices.items():
            max_shares = int(curr_balance // current_price)
            if max_shares > 0:
                actions.append(Action('buy', stock_name, max_shares))
        return actions

class DeepStateStrategy(Strategy):
    """A strategy that buys all stocks."""

    def decide_actions(self, predictions, curr_prices, portfolio, curr_balance, curr_date, obj_date):
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
    def decide_actions(self, predictions, curr_prices, portfolio, curr_balance, curr_date, obj_date):
        actions = []
        for stock_name, current_price in curr_prices.items():
            max_shares = int(curr_balance // current_price)
            if max_shares > 0:
                actions.append(Action('buy', stock_name, max_shares))
        return actions

class ConspiracyTheoryStrategy(Strategy):
    """A strategy that buys all stocks."""

    def decide_actions(self, predictions, curr_prices, portfolio, curr_balance, curr_date, obj_date):
        actions = []
        for stock_name, current_price in curr_prices.items():
            max_shares = int(curr_balance // current_price)
            if max_shares > 0:
                actions.append(Action('buy', stock_name, max_shares))
        return actions

class GamblerStrategy(Strategy):
    """buys stocks with high risk"""
    def decide_actions(self, predictions, curr_prices, portfolio, curr_balance, curr_date, obj_date):
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
    def decide_actions(self, predictions, curr_prices, portfolio, curr_balance, curr_date, obj_date):
        actions = []
        for stock_name, current_price in curr_prices.items():
            prediction = llm.ask_llm(self.model, stock_name, curr_date, obj_date)
            #ex) prediction = "up/down, 0.5"
            up_down = prediction.split(",")[0]
            intensity = float(prediction.split(",")[1])
            if up_down == "up":
                max_shares = int(curr_balance // current_price)
                if max_shares > 0:
                    actions.append(Action('buy', stock_name, max_shares* intensity))
            elif up_down == "down":
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

class Trader:
    def __init__(self, name, description, init_balance, trade_freq):
        self.name = name
        self.description = description
        self.balance = init_balance
        self.history = []
        self.portfolio = []
        self.strategy = None
        self.trade_freq = trade_freq
        self.days_since_last_trade = 0

    def get_predictions(self, curr_prices):
        # Generate more meaningful predictions
        return {stock_name: price * random.uniform(0.9, 1.1) for stock_name, price in curr_prices.items()}

    def change_strategy(self, strategy, trade_freq):
        self.strategy = strategy
        self.trade_freq = trade_freq

    def take_action(self, predictions, curr_prices, market):
        if self.days_since_last_trade < self.trade_freq:
            self.days_since_last_trade += 1
            return
        self.days_since_last_trade = 0

        if self.strategy is None:
            raise Exception(f"{self.name} has no strategy set.")

        actions = self.strategy.decide_actions(predictions, curr_prices, self.portfolio, self.balance, market.current_date, market.get_next_day())

        for action in actions:
            stock_name = action.stock_name
            quantity = action.quantity
            price = curr_prices.get(stock_name, 0)

            if action.action_type == 'buy':
                total_cost = quantity * price
                if total_cost <= self.balance:
                    self.balance -= total_cost
                    self.add_stock_to_portfolio(stock_name, quantity)
                    print(f"{self.name} bought {quantity} shares of {stock_name} at {price:.2f}.")
                else:
                    print(f"{self.name} has insufficient balance to buy {quantity} shares of {stock_name}.")

            elif action.action_type == 'sell':
                stock = self.get_stock_from_portfolio(stock_name)
                if stock and stock.quantity >= quantity:
                    stock.quantity -= quantity
                    self.balance += quantity * price
                    print(f"{self.name} sold {quantity} shares of {stock_name} at {price:.2f}.")
                else:
                    print(f"{self.name} has insufficient shares to sell {quantity} shares of {stock_name}.")

            elif action.action_type == 'hold':
                print(f"{self.name} is holding {stock_name}.")

        self.portfolio = [stock for stock in self.portfolio if stock.quantity > 0]

    def add_stock_to_portfolio(self, stock_name, quantity):
        stock = self.get_stock_from_portfolio(stock_name)
        if stock:
            stock.quantity += quantity
        else:
            self.portfolio.append(Stock(stock_name, quantity))

    def get_stock_from_portfolio(self, stock_name):
        for stock in self.portfolio:
            if stock.name == stock_name:
                return stock
        return None

    def get_portfolio_value(self, prices):
        total_value = 0
        for stock in self.portfolio:
            stock_name = stock.name
            current_price = prices.get(stock_name, 0)
            total_value += stock.quantity * current_price
        return total_value

    def get_total_value(self, prices):
        return self.balance + self.get_portfolio_value(prices)


def main():
    # Initialize the market with data up to '2021-01-10'
    market = mk.Market(['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'TSLA'], curr_date='2021-01-10')

    traders = [
        Trader(name="Alice", description="ScaredyCat trader", init_balance=10000, trade_freq=1),
        Trader(name="Bob", description="GoldfishMemory trader", init_balance=10000, trade_freq=1)
    ]

    traders[0].change_strategy(ScaredyCatStrategy(), 1)
    traders[1].change_strategy(GoldfishMemoryStrategy(), 1)

    # Simulate market days and let traders take actions each day
    for _ in range(5):
        current_prices = market.get_stock_prices(market.current_date)
        print(f"\n=== Current Date: {market.current_date} ===")

        for trader in traders:
            predictions = trader.get_predictions(current_prices)
            trader.take_action(predictions, current_prices, market)
            print(f"{trader.name}'s total value: {trader.get_total_value(current_prices):.2f}")

        # Move to the next day in the market
        market.next_day()


if __name__ == "__main__":
    main()
