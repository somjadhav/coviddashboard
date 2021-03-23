This is a basic dashboard created using Python's Dash library. Shown are daily case numbers (both new and cumulative) for every Texas county and the state as a whole. I went ahead and tried to predict the number of cases 21 days into the future, using 3 approaches:

- LSTM
- ARIMA model
- Holt's Exponential Model with a Damped Trend

In the future, I would like to add more prediction models, but for now, this is all there is. 

The data was retrieved from https://dshs.texas.gov/coronavirus/TexasCOVID19DailyCountyCaseCountData.xlsx. Here are some additional steps I took:

- Remove the top 2 rows
- Delete everything after the 'Total' row
- save the file as datafile.csv in the root directory

How to run the application:

- Fork and download onto your local machine
- use Bash/Command Prompt to cd into the root directory
- create a virtual environment
- run the 'pip install -r requirements.txt' command
- run the 'python dashboard.py' command (make sure you have followed the instructions regarding the data file) 
