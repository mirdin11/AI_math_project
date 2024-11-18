# main.py
import random
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime

# Import the market and strategy modules
import market as mk
from strategy import ScaredyCatStrategy, GoldfishMemoryStrategy,NewsReaderStrategy, NewsReaderStrategy_Hidden, Trader


def main():
    # Initialize the market with data up to '2021-01-10'
    market = mk.Market(['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'TSLA'], curr_date='2020-01-10')

    # Initialize traders with different strategies
    traders = [
        Trader(name="Alice", description="ScaredyCat trader", init_balance=10000, trade_freq=1),
        Trader(name="Bob", description="GoldfishMemory trader", init_balance=10000, trade_freq=1),
        Trader(name="Charlie", description="Random trader", init_balance=10000, trade_freq=1),
        Trader(name="Eve", description="NewsReader trader", init_balance=10000, trade_freq=1),
        Trader(name="Eve_Hidden", description="NewsReader_Hidden trader", init_balance=10000, trade_freq=1)
    ]

    # Assign strategies to traders
    traders[0].change_strategy(ScaredyCatStrategy(), trade_freq=1)
    traders[1].change_strategy(GoldfishMemoryStrategy(), trade_freq=1)
    traders[2].change_strategy(GoldfishMemoryStrategy(), trade_freq=10)
    traders[3].change_strategy(NewsReaderStrategy(), trade_freq=10)
    traders[4].change_strategy(NewsReaderStrategy_Hidden(), trade_freq=10)

    # Variables to track data for visualization
    dates = [market.current_date]
    market_prices = {ticker: [price] for ticker, price in market.get_stock_prices(market.current_date).items()}
    trader_values = {trader.name: [trader.get_total_value(market.get_stock_prices(market.current_date))]
                     for trader in traders}

    # Simulate market for 10 trading days
    for _ in range(300):
        market.next_day()
        current_prices = market.get_stock_prices(market.current_date)
        dates.append(market.current_date)

        # Record market prices for each stock
        for ticker in market_prices:
            market_prices[ticker].append(current_prices.get(ticker, None))

        # Traders make decisions based on predictions and current prices
        for trader in traders:
            predictions = trader.get_predictions(current_prices)
            trader.take_action(predictions, current_prices, market)

            # Record trader portfolio value
            trader_values[trader.name].append(trader.get_total_value(current_prices))

    # Plotting market prices for selected stocks
    #plt.figure(figsize=(14, 6))
    #for ticker, prices in market_prices.items():
    #    valid_dates = [dates[i] for i in range(len(prices)) if prices[i] is not None]
    #    valid_prices = [prices[i] for i in range(len(prices)) if prices[i] is not None]
    #    if valid_prices:
    #        plt.plot(valid_dates, valid_prices, label=ticker)

    #plt.xlabel('Date')
    #plt.ylabel('Stock Price ($)')
    #plt.title('Market Prices of Stocks Over Time')
    #plt.xticks(rotation=45)
    #plt.legend()
    #plt.tight_layout()
    #plt.show()

    # Plotting each trader's portfolio value over time
    plt.figure(figsize=(14, 6))
    for trader_name, values in trader_values.items():
        plt.plot(dates, values, label=trader_name)
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.title("Trader's Portfolio Value Over Time")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
