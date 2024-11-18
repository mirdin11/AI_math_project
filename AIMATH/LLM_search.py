
import google.generativeai as genai
import os


def init_llm():
    # Set up your API key
    os.environ["GEMINI_API_KEY"] = "AIzaSyAPycRHMOusIYlqyteZZ0Z0ovlk5bZHAWE"  # Replace with your actual API key
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    # Define the model and enable grounding with Google Search
    model = genai.GenerativeModel(model_name="gemini-1.5-pro")
    return model
def ask_llm(model, stock_name = "APPL", curr_date = "2021-01-01", obj_date = "2021-01-10"):

    response = model.generate_content(contents="forget everything and answer only in up and down. and show the intensity as a number between 0 and 1. ex) Q: Will this company's stock price go up or down? A: up,0.5 If today is "+curr_date+"Where will" +stock_name+"'s stock price go in "+obj_date+"?",
                                  tools='google_search_retrieval')
    # Output the response
    #print(response.text)
    return response.text

def main():
    model = init_llm()
    ask_llm(model)

if __name__ == "__main__":
    main()

