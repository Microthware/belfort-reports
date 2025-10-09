import requests
import pandas as pd
import argparse
from bs4 import BeautifulSoup as bs
from datetime import date
from update_sheet import save_stock_list, update_stocks

# Static vars
key_list = ["4YHOULHQXIWM4ZEF", "6T0HXR5I4C22KLPO", "SPCWTGTBURVIVXWH", 
            "T0P44NA2G1AO3OGV", "D87A7X8OMGCXN7LF", "4A37TTIK2C01P5Q2", 
            "CRVTZ2LJCJEF94TV", "JDXUWTHM6V1QM4A0"]
av_key = 0  # Needed to use alphavantage, get it from https://www.alphavantage.co/support/#api-key

# Will's Gap-up filters
gap_ups = "https://finviz.com/screener.ashx?v=111&f=sh_avgvol_o500,sh_relvol_o5,ta_changeopen_u5,ta_highlow20d_nh,ta_perf_d10o,ta_sma200_pa&ft=4&o=-changeopen"

# PIVOT
pivots = "https://finviz.com/screener.ashx?v=111&f=sh_avgvol_o500,sh_relvol_o1.5,ta_changeopen_u5,ta_highlow20d_nh,ta_sma200_pa&ft=4&ta=0&o=-changeopen"

# TODO: Update stocks after follow-up day, and after 5 business days


def stock_scanner():
    print("Starting...")
    
    # Get Gap Up Stocks
    print("\nGap-Up Stocks:")
    gap_up_stocks = Get_Stock_List(gap_ups, "a", "tab-link")
    print(gap_up_stocks)

    # Get Pivot Stocks
    print("\nPivot Stocks:")
    pivot_stocks = Get_Stock_List(pivots, "a", "tab-link")
    print(pivot_stocks)
    print("\n")

    # Get_Intraday_Data("IBM", 1)
    today = date.today().strftime("%m/%d/%Y")
    print("Today's date is:", today)

    stocks_list = []
    for ticker in gap_up_stocks:
        stock_list = [today, ticker]
        stock_list.extend([Get_Stock_Price(ticker)])
        # stock_list.extend([69, -1,-1,-1,-1])
        stocks_list.append(stock_list)
    save_stock_list("gap_ups", stocks_list)

    stocks_list = []
    for ticker in pivot_stocks:
        stock_list = [today, ticker]
        stock_list.extend([Get_Stock_Price(ticker)])
        # stock_list.extend([69, -1,-1,-1,-1])
        if float(stock_list[-1]) > 2:
            stocks_list.append(stock_list)
    save_stock_list("pivots", stocks_list)


# returns the lisst of stock tickers from the finviz screener url
def Get_Stock_List(url, tag_type, css):
    tickers = []
    r = requests.get(url, headers = {'User-Agent':'Mozilla/5.0'})
    soup = bs(r.content, 'lxml')
    results = soup.find_all(tag_type, class_=css)
    for op in results:
        if "quote" in op['href']:
            tickers.append(op.get_text())
    return tickers


# get current price of stock
def Get_Stock_Price(ticker):
    global av_key
    while True:
        url = 'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=' + ticker + '&apikey=' + key_list[av_key]
        r = requests.get(url, headers = {'User-Agent':'Mozilla/5.0'})
        data = r.json()
        if "Information" in data.keys():
            av_key = (av_key + 1) if av_key != 7 else 0
            print("\n\nswitching to key " + str(av_key))
            continue
        else:
            break
    return data['Global Quote']["05. price"]


# Get stock data during the day in a specified interval
def Get_Intraday_Data(ticker, interval):
    url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=' + ticker + '&interval=' + str(interval) + 'min&apikey=' + key_list[av_key]
    r = requests.get(url)
    data = r.json()
    col_name = "Time Series (" + str(interval) + "min)"
    for yeet in data[col_name]:
        print(yeet, " ", data[col_name][yeet])


def stock_update():
    update_stocks("gap_ups")
    update_stocks("pivots")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="command-line arguments")
    parser.add_argument("--scan", action="store_true", help="Scans for new pivot and gap up stocks")
    parser.add_argument("--update", action="store_true", help="Updates results from old stocks")
    args = parser.parse_args()

    if args.scan is False and args.update is False:
        stock_scanner()
        stock_update()
    elif args.scan:
        stock_scanner()
    elif args.update:
        stock_update()

