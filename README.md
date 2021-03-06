# stock_anomaly

## What is the purpose of stock_anomaly?

Let's look at some use cases where this model will be useful
  
### Use Case 1
You want to invest in a certain set of stocks and is waiting for the right time and price. However, you do not know what is a good price for the stock is or if it is the right time to buy.
 
### Use Case 2
You want to invest in a certain set of stocks and is waiting for the right time and price. In this case, you do not keep up with the stock market and often miss low or high prices which may convince you to buy a stock.
    
### Use Case 3
You want to gain exposure to the market or diversify your portfolio. In order to find new trending stocks, you have to do a lot of manual work and research.
    
### Use Case 4
You recently came across some money and are looking to invest. After a period of inactivity in the stock market, you want to find out which stocks have dropped or increased in price in the past few months to narrow down the number of stocks to research.
    
### Summing it Up
  
My stock_anomaly model will help users keep track of a list of stocks and if the any of those individual stocks have had a low or high anomaly in it's price. If a certain stock's price, like Manulife Financial for example, increases or drops more than it usually should based on historical data, investors will want to know that fact. Keeping track of one stock that investors are interested in is EASY. However, investors are generally interested in SEVERAL stocks and are just waiting for the right time to buy or have some extra money available to invest.Thus I coded this model that detects anomalies in stocks for investors.
    
## How does it work? How do I use the model?

Based on the list of stock symbols provided, the code will extract 3 years of historical data and identify all the anomalous prices of each stock over the three years. This will help investors learn how volatile the stock has been over the past three years. 

Furthermore, for each anomaly, the difference in the anomalous price and the price of that stock 1,2,3,5,10 and 30 days after the anomaly has occured are stored for the investor. The investor can then analyze how much the price fluctuates after an anomaly occurs. 

In order to run the model, the investor has to provide an input file. Let's look at the input and output files.

### Input File

The model takes in one input file (csv) which must contain a column of stock symbols (in the format that Yahoo Finance identifies stocks symbols). Paste this filepath into the code and the coumn index of the stock symbols. 

### Output File
Run the code and it will do EVERYTHING for you. The code will output a csv file in the same directory of the input file.

## How does it identify anomalies?

This model detects anomalous stock prices based on 10 day Moving Average. The residual standard deviation for 10 days of historical data is calculated as well.

Anomaly if: Current Stock Price > Moving Average + Residual Std Dev * 5.5

5.5 is the threshold value that I decided to use. If you want bigger or smaller anomalies, change the threshold.

## Performance

For a list of 250 stocks, the model take about 8 minutes to run. I am working on improving the performance of this model.
 
