import yfinance as yf
from datetime import datetime, timedelta


class Market:
    def __init__(self, tickers, curr_date):
        """
        Initialize the Market class with stock data for given tickers up to the current date.
        :param tickers: List of stock tickers (e.g., ['AAPL', 'GOOGL'])
        :param curr_date: The current date in 'YYYY-MM-DD' format.
        """
        data = yf.download(tickers, start="2021-01-01", end=curr_date)
        # Restructure the data into a dictionary: {ticker: {date: price}}
        self.stocks = {}
        self.tickers = tickers
        for ticker in tickers:
            # Convert timestamps to string (YYYY-MM-DD) format
            self.stocks[ticker] = {
                date.strftime('%Y-%m-%d'): price
                for date, price in data['Close'][ticker].dropna().items()
            }
        self.current_date = curr_date

    def update_market(self, date):
        """
        Update the stock prices for a given date.
        :param date: Date as a string (e.g., '2021-01-01').
        """
        # Download data for the specific date range (which will just be this one day)
        stock_data = yf.download(self.tickers, start=date, end=(datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d'))

        for stock_name in self.tickers:
            # Extract and add the closing price if available
            if not stock_data.empty and 'Close' in stock_data:
                price = stock_data['Close'][stock_name].iloc[0]
                self.stocks[stock_name][date] = price
        self.current_date = date

    def next_day(self):
        """
        Move the market to the next trading day.
        If it's a weekend, move to the following Monday.
        """
        date = datetime.strptime(self.current_date, '%Y-%m-%d')
        next_date = date + timedelta(days=1)

        # Skip weekends
        while next_date.weekday() >= 5:  # 5 is Saturday, 6 is Sunday
            next_date += timedelta(days=1)

        next_date_str = next_date.strftime('%Y-%m-%d')
        self.update_market(next_date_str)

    def get_stock_prices(self, date):
        """
        Get stock prices for a specific date.
        :param date: Date as a string (e.g., '2021-01-01').
        :return: Dictionary of stock prices for the given date.
        """
        current_prices = {}
        for stock_name, prices in self.stocks.items():
            if date in prices:
                current_prices[stock_name] = prices[date]
        return current_prices
    
    def get_current_prices(self):
        """
        Get the current stock prices.
        :return: Dictionary of stock prices for the current date.
        """
        return self.get_stock_prices(self.current_date)

    def get_stock_price(self, name, date):
        """
        Get the price of a specific stock on a specific date.
        :param name: Stock ticker (e.g., 'AAPL').
        :param date: Date as a string (e.g., '2021-01-01').
        :return: Price of the stock on the given date.
        """
        return self.stocks.get(name, {}).get(date, None)
    
    def get_current_price(self, name):
        """
        Get the current price of a specific stock.
        :param name: Stock ticker (e.g., 'AAPL').
        :return: Current price of the stock.
        """
        return self.get_stock_price(name, self.current_date)

    def get_stock(self, stock_name):
        """
        Get all price data for a specific stock.
        :param stock_name: Stock ticker (e.g., 'AAPL').
        :return: Dictionary of date: price for the stock.
        """
        return self.stocks.get(stock_name, None)


def main():
    # Initialize the market with data up to '2021-01-10'
    market = Market(['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'TSLA'], curr_date='2021-01-10')

    # Get stock prices for a specific date
    print("Prices on 2021-01-05:")
    print(market.get_stock_prices('2021-01-05'))

    # Get price for a specific stock on a specific date
    print("\nAAPL price on 2021-01-05:")
    print(market.get_stock_price('AAPL', '2021-01-05'))

    # Move the market to the next day and update prices
    market.next_day()
    print(market.get_current_prices(), market.current_date)
    # Get updated stock prices
    print("\nUpdated prices on 2021-01-11:")
    print(market.get_stock_prices('2021-01-11'))

    # Check stock data
    print("\nAll prices for AAPL:")
    print(market.get_stock('AAPL'))


if __name__ == '__main__':
    main()
