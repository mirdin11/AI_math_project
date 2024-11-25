import LLM_search as llm
import prediction_module as pm

class Action:
    def __init__(self, action_type, stock_name, quantity=0):
        self.action_type = action_type  # 'buy', 'sell', or 'hold'
        self.stock_name = stock_name
        self.quantity = quantity


class Strategy:
    def __init__(self, prediction_model):
        self.prediction_model = prediction_model

    def decide_actions(self, curr_prices, portfolio, curr_balance, curr_date, obj_date, market):
        raise NotImplementedError("decide_actions() must be implemented by subclasses.")
    
    def predict_prices(self, tickers, market):
        return self.prediction_model.predict(tickers, market)
    
    
    def change_prediction_model_to(self, prediction_model):
        self.prediction_model = prediction_model

    def construct_action_list(self, obj_portfolio, portfolio):
        actions = []
        for stock_name, obj_quantity in obj_portfolio.items():
            curr_quantity = portfolio.get(stock_name, 0)
            if obj_quantity > curr_quantity:
                actions.append(Action('buy', stock_name, obj_quantity - curr_quantity))
            elif obj_quantity < curr_quantity:
                actions.append(Action('sell', stock_name, curr_quantity - obj_quantity))
            else:
                actions.append(Action('hold', stock_name))
        return actions


class ScaredyCatStrategy(Strategy):
    """A strategy that sells all holdings immediately."""

    def decide_actions(self, curr_prices, portfolio, curr_balance, curr_date, obj_date, market):
        actions = []
        for stock in portfolio:
            if stock.quantity > 0:
                actions.append(Action('sell', stock.name, stock.quantity))
        return actions


class GoldfishMemoryStrategy(Strategy):
    """A strategy that only considers the most recent predictions."""

    def decide_actions(self, curr_prices, portfolio, curr_balance, curr_date, obj_date, market):
        predictions = self.predict_prices(market)
        #obj_portfolio = self.objective_func(risk_factor, profit_factor, predictions)
        return self.construct_action_list(predictions, portfolio)

    @staticmethod
    def get_stock_from_portfolio(portfolio, stock_name):
        for stock in portfolio:
            if stock.name == stock_name:
                return stock
        return None
    
class OptimisticStrategy(Strategy):
    """A strategy that buys all stocks."""

    def decide_actions(self, curr_prices, portfolio, curr_balance, curr_date, obj_date, market):
        predictions = self.predict_prices(market)
        return self.construct_action_list(predictions, portfolio)

class DeepStateStrategy(Strategy):
    """A strategy that buys all stocks."""

    def decide_actions(self, curr_prices, portfolio, curr_balance, curr_date, obj_date, market):
        predictions = self.predict_prices(market)
        return self.construct_action_list(predictions, portfolio)

class DemocraticStrategy(Strategy):
    class Personality:
        def __init__(self, name, buy_inclination, sell_inclination):
            self.name = name
            self.buy_inclination = buy_inclination
            self.sell_inclination = sell_inclination

        def __repr__(self):
            return f"{self.name}(buy={self.buy_inclination}, sell={self.sell_inclination})"
    class AgentWeight:
        def __init__(self, personality, weight):
            self.personality = personality
            self.weight = weight

        def __repr__(self):
            return f"{self.personality.name}({self.weight})"
    def __init__(self):
        self.personalities = [
            self.Personality("Optimist", buy_inclination=0.7, sell_inclination=0.3),
            self.Personality("Pessimist", buy_inclination=0.3, sell_inclination=0.7),
            self.Personality("Neutral", buy_inclination=0.5, sell_inclination=0.5),
            # 추가 성격 유형...
        ]
        self.agent_weights = [
            self.AgentWeight(self.personalities[0], weight=0.3),  # Optimist 30%
            self.AgentWeight(self.personalities[1], weight=0.2),  # Pessimist 20%
            self.AgentWeight(self.personalities[2], weight=0.5),  # Neutral 50%
            # 초기 비율 설정
        ]

    def decide_actions(self, curr_prices, portfolio, curr_balance, curr_date, obj_date, market):
        predictions = self.predict_prices(market)
        #obj_portfolio = self.objective_func(risk_factor, profit_factor, predictions)
        return self.construct_action_list(predictions, portfolio)
    
    def update_parameters(self, actual_prices, predicted_prices):
        # 오차 계산
        errors = {}
        for stock_name in actual_prices.keys():
            actual = actual_prices[stock_name]
            predicted = predicted_prices.get(stock_name, actual)
            errors[stock_name] = actual - predicted

        # 총 오차 계산 (여기서는 제곱 오차의 합)
        total_error = sum(error ** 2 for error in errors.values())

        # 파라미터 업데이트 (예: 경사 하강법)
        learning_rate = 0.01
        for agent_weight in self.agent_weights:
            # 성향 파라미터 업데이트
            personality = agent_weight.personality
            # 매수 성향 업데이트
            personality.buy_inclination -= learning_rate * total_error * self.gradient_wrt_buy_inclination(personality)
            personality.buy_inclination = max(0, min(personality.buy_inclination, 1))
            # 매도 성향 업데이트
            personality.sell_inclination -= learning_rate * total_error * self.gradient_wrt_sell_inclination(personality)
            personality.sell_inclination = max(0, min(personality.sell_inclination, 1))

            # 비율(weight) 업데이트
            agent_weight.weight -= learning_rate * total_error * self.gradient_wrt_weight(agent_weight)
            agent_weight.weight = max(0, agent_weight.weight)

        # 비율 재정규화 (합이 1이 되도록)
        total_weight = sum(aw.weight for aw in self.agent_weights)
        for agent_weight in self.agent_weights:
            agent_weight.weight /= total_weight

    def gradient_wrt_buy_inclination(self, personality):
        # 매수 성향에 대한 오차의 기울기 계산
        # 실제로는 복잡한 미분이 필요하며, 여기서는 간단히 가정
        return 1

    def gradient_wrt_sell_inclination(self, personality):
        # 매도 성향에 대한 오차의 기울기 계산
        return -1

    def gradient_wrt_weight(self, agent_weight):
        # 비율에 대한 오차의 기울기 계산
        return 1

class ConspiracyTheoryStrategy(Strategy):
    """A strategy that buys all stocks."""

    def decide_actions(self, curr_prices, portfolio, curr_balance, curr_date, obj_date, market):
        actions = []
        for stock_name, current_price in curr_prices.items():
            max_shares = int(curr_balance // current_price)
            if max_shares > 0:
                actions.append(Action('buy', stock_name, max_shares))
        return actions

class GamblerStrategy(Strategy):
    """buys stocks with high risk"""
    def decide_actions(self, curr_prices, portfolio, curr_balance, curr_date, obj_date, market):
        actions = []
        for stock_name, current_price in curr_prices.items():
            max_shares = int(curr_balance // current_price)
            if max_shares > 0:
                actions.append(Action('buy', stock_name, max_shares))
        return actions
    
class NewsReaderStrategy(Strategy):
    """GPT analysis"""
    def __init__(self):
        self.model = llm.init_llm()
    def decide_actions(self, curr_prices, portfolio, curr_balance, curr_date, obj_date, market):
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
        self.model = llm.init_llm()
    def decide_actions(self, curr_prices, portfolio, curr_balance, curr_date, obj_date, market):
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

    def decide_actions(self, curr_prices, portfolio, curr_balance):
        actions = []
        for stock_name, current_price in curr_prices.items():
            max_shares = int(curr_balance // current_price)
            if max_shares > 0:
                actions.append(Action('buy', stock_name, max_shares))
        return actions

class UnbelieverStrategy(Strategy):
    """Price is irellevent to past price"""

    def decide_actions(self, curr_prices, portfolio, curr_balance):
        actions = []
        for stock_name, current_price in curr_prices.items():
            max_shares = int(curr_balance // current_price)
            if max_shares > 0:
                actions.append(Action('buy', stock_name, max_shares))
        return actions

class IndexfundStrategy(Strategy):
    """A strategy that buys all stocks."""

    def decide_actions(self, curr_prices, portfolio, curr_balance):
        actions = []
        for stock_name, current_price in curr_prices.items():
            max_shares = int(curr_balance // current_price)
            if max_shares > 0:
                actions.append(Action('buy', stock_name, max_shares))
        return actions