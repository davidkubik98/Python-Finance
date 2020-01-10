# Preprossing data from Machine Learning
from collections import Counter
import numpy as np
import pandas as pd
import pickle
from sklearn import svm, model_selection as cross_validation, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier

def process_data_for_labels(ticker):
    hm_days = 7 # how many days = 7
    df = pd.read_csv("sp500_joined_closes.csv", index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0,inplace=True)
    for i in range(1, hm_days+1):
        df["{}_{}d".format(ticker, i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker] # Shifting negatively to get future data 

    df.fillna(0,inplace=True)

    return tickers,df, hm_days

#process_data_for_labels("XOM")

def buy_sell_hold(*args): # lets us pass unlimited arguments, becomes iterable
    # this is where we could make a strategy to buy / sell stocks
    cols = [c for c in args]
    requirement = 0.02 # if the stock price changes by 2% in 7 days
    for col in cols:
        if col > requirement:
            return 1
        if col < -requirement:
            return -1
    return 0 

def extract_featuresets(ticker):
    tickers, df, hm_days = process_data_for_labels(ticker)
    df ['{}_target'.format(ticker)] = list(map(buy_sell_hold, *[df['{}_{}d'.format(ticker, i)]for i in range(1, hm_days+1)]))
    
    vals = df['{}_target'.format(ticker)].values.tolist()
    
    str_vals = [str(i) for i in vals]
    print ("Data spread:", Counter(str_vals))
    df.fillna(0, inplace=True)

    df = df.replace([np.inf, -np.inf], np.nan) 
    df.dropna(inplace=True)

    df_vals = df[[ticker for ticker in tickers]].pct_change() #percent change from yesterday to today
 
    X = df_vals.values #percent changes for all of the companies data  
    y = df["{}_target".format(ticker)].values
    return X, y, df 
    
def do_ml(ticker):
    X, y, df = extract_featuresets(ticker)
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.25)

    clf = neighbors.KNeighborsClassifier()

    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print ("Accuracy: ", confidence)
    predictions = clf.predict(X_test)
    print ("Predicted Spread: ", Counter(predictions))
    return confidence

do_ml("XOM")
    






    
