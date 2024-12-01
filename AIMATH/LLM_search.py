
import google.generativeai as genai
import os
import market as mk
import pandas as pd

def init_llm():
    # Set up your API key
    os.environ["GEMINI_API_KEY"] = "AIzaSyAPycRHMOusIYlqyteZZ0Z0ovlk5bZHAWE"  # Replace with your actual API key
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    # Define the model and enable grounding with Google Search
    model = genai.GenerativeModel(model_name="gemini-1.5-pro")
    return model
def ask_llm(model, stock_names , curr_date = "2021-01-01", obj_date = "2021-01-10"):
    stocks = ""
    for stock in stock_names:
        stocks += stock + ","
    response = model.generate_content(contents="forget everything and answer only in up and down. and show the intensity as a number between 0 and 1. DO NOT COMMENT ANYTHING ELSE. ex) Q: Will these companies's stock price go up or down? A: stockA: up, 0.8\nstockB: down, 0.2\nstockC: up, 0.1 If today is "+curr_date+"Where will" +stocks+"'s stock price go in "+obj_date+"?",
                                  tools='google_search_retrieval')
    # Output the response
    return response.text

def ask_llm_info_hidden(model, stock_names, curr_prices):
    stocks = "" 
    for stock in stock_names:
        stocks += stock + ","
    prices = ""
    for price in curr_prices:
        prices += price + ","
    response = model.generate_content(contents="forget everything and answer only in up and down. and show the intensity as a number between 0 and 1. DO NOT COMMENT ANYTHING ELSE. ex) Q: Will these companies's stock price go up or down? A: stockA: up, 0.8\nstockB: down, 0.2\nstockC: up, 0.1 Where will" +stocks+"'s stock price go tomorrow if today's price is "+f'{prices}'+"respectively?",
                                  tools='google_search_retrieval')
    # Output the response
    #print(response.text)
    return response.text

def main():
    stock_names = ['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'TSLA'] 
    model = init_llm()
    days = 30
    market = mk.Market(stock_names, curr_date='2021-01-10')
    prices = market.get_stock_prices('2021-01-10')
    

if __name__ == "__main__":
    main()

