
class Stock:
    def __init__(self, name, quantity):
        self.name = name
        self.quantity = quantity

class Trader:
    def __init__(self, name, description, init_balance,trade_freq =1):
        self.name = name
        self.description = description
        self.balance = init_balance
        self.history = []
        self.portfolio = {}
        self.strategy = None
        self.trade_freq = trade_freq
        self.days_since_last_trade = 0

    def change_strategy(self, strategy, trade_freq):
        self.strategy = strategy
        self.trade_freq = trade_freq

    def take_action(self, obj_portfolio, market):
        if self.days_since_last_trade < self.trade_freq:
            self.days_since_last_trade += 1
            return
        self.days_since_last_trade = 0

        if self.strategy is None:
            raise Exception(f"{self.name} has no strategy set.")
        
        actions = self.strategy.decide_actions(obj_portfolio)
        prices = market.get_current_prices()
        for action in actions:
            if action.action_type == 'buy':
                print(f"{self.name} buys {action.quantity} shares of {action.stock_name} at ${prices[action.stock_name]} each.")
                self.add_stock_to_portfolio(action.stock_name, action.quantity, prices)
            elif action.action_type == 'sell':
                print(f"{self.name} sells {action.quantity} shares of {action.stock_name} at ${prices[action.stock_name]} each.")
                self.add_stock_to_portfolio(action.stock_name, -action.quantity, prices)

    def add_stock_to_portfolio(self, stock_name, quantity, prices):
        self.balance -= quantity * prices[stock_name]
        if self.portfolio.get(stock_name) is None:
            self.portfolio[stock_name] = Stock(stock_name, quantity)
        else:
            self.portfolio[stock_name].quantity += quantity

    def get_portfolio_value(self, prices):
        total_value = 0
        for stock in self.portfolio:
            stock_name = stock.name
            current_price = prices.get(stock_name, 0)
            total_value += stock.quantity * current_price
        return total_value

    def get_total_value(self, prices):
        return self.balance + self.get_portfolio_value(prices)

