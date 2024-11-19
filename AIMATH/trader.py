import random
import market as mk

class Stock:
    def __init__(self, name, quantity):
        self.name = name
        self.quantity = quantity

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
        
        if curr_prices is None:
            print(f"{self.name} could not take action because current prices are not available.")
            return

        actions = self.strategy.decide_actions(predictions, curr_prices, self.portfolio, self.balance, market.current_date, market.get_next_day(), market)

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

