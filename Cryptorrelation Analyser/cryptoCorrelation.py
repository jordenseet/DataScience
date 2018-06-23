import os
import numpy as np
import pandas as pd
import pickle
import quandl
from datetime import datetime
import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
import sklearn.linear_model

def get_quandl_data(quandl_id):
    '''Download and cache Quandl dataseries'''
    cache_path = '{}.pkl'.format(quandl_id).replace('/','-')
    try:
        f = open(cache_path, 'rb')
        df = pickle.load(f)   
        print('Loaded {} from cache'.format(quandl_id))
    except (OSError, IOError) as e:
        print('Downloading {} from Quandl'.format(quandl_id))
        df = quandl.get(quandl_id, returns="pandas")
        df.to_pickle(cache_path)
        print('Cached {} at {}'.format(quandl_id, cache_path))
    return df
    
# Pull Kraken BTC price exchange data
btc_usd_price_kraken = get_quandl_data('BCHARTS/KRAKENUSD')
# Pull pricing data for 3 more BTC exchanges
exchanges = ['COINBASE','BITSTAMP','ITBIT']

exchange_data = {}

exchange_data['KRAKEN'] = btc_usd_price_kraken

for exchange in exchanges:
    exchange_code = 'BCHARTS/{}USD'.format(exchange)
    btc_exchange_df = get_quandl_data(exchange_code)
    exchange_data[exchange] = btc_exchange_df

def merge_dfs_on_column(dataframes, labels, col):
    '''Merge a single column of each dataframe into a new combined dataframe'''
    series_dict = {}
    for index in range(len(dataframes)):
        series_dict[labels[index]] = dataframes[index][col]
        
    return pd.DataFrame(series_dict)
    
btc_usd_datasets = merge_dfs_on_column(list(exchange_data.values()), list(exchange_data.keys()), 'Weighted Price')


# Plot all of the BTC exchange prices
# Remove "0" values
btc_usd_datasets.replace(0, np.nan, inplace=True)
# Calculate the average BTC price as a new column
btc_usd_datasets['avg_btc_price_usd'] = btc_usd_datasets.mean(axis=1)

def get_json_data(json_url, cache_path):
    '''Download and cache JSON data, return as a dataframe.'''
    try:        
        f = open(cache_path, 'rb')
        df = pickle.load(f)   
        print('Loaded {} from cache'.format(json_url))
    except (OSError, IOError) as e:
        print('Downloading {}'.format(json_url))
        df = pd.read_json(json_url)
        df.to_pickle(cache_path)
        print('Cached {} at {}'.format(json_url, cache_path))
    return df
    
base_polo_url = 'https://poloniex.com/public?command=returnChartData&currencyPair={}&start={}&end={}&period={}'
start_date = datetime.strptime('2015-01-01', '%Y-%m-%d') # get data from the start of 2015
end_date = datetime.now() # up until today
pediod = 86400 # pull daily data (86,400 seconds per day)

def get_crypto_data(poloniex_pair):
    '''Retrieve cryptocurrency data from poloniex'''
    json_url = base_polo_url.format(poloniex_pair, start_date.timestamp(), end_date.timestamp(), pediod)
    data_df = get_json_data(json_url, poloniex_pair)
    data_df = data_df.set_index('date')
    return data_df
    
altcoins = ['ETH','LTC','XRP','ETC','STR','DASH','XMR','XEM','BTS','BCH','DOGE','ZRX','OMG','LSK','STEEM']

altcoin_data = {}
for altcoin in altcoins:
    coinpair = 'BTC_{}'.format(altcoin)
    crypto_price_df = get_crypto_data(coinpair)
    altcoin_data[altcoin] = crypto_price_df

# Calculate USD Price as a new column in each altcoin dataframe
for altcoin in altcoin_data.keys():
    altcoin_data[altcoin]['price_usd'] =  altcoin_data[altcoin]['weightedAverage'] * btc_usd_datasets['avg_btc_price_usd']

# Merge USD price of each altcoin into single dataframe 
combined_df = merge_dfs_on_column(list(altcoin_data.values()), list(altcoin_data.keys()), 'price_usd')
# Add BTC price to the dataframe
combined_df['BTC'] = btc_usd_datasets['avg_btc_price_usd']

combined_df = combined_df.dropna()
factors = list(combined_df.columns)
combined_df2 = combined_df[combined_df.shape[0]-365:combined_df.shape[0]]

inner_best_factor = None
outer_best_factor = None
inner_best_r2 = 0
outer_best_r2 = 0
inner_best_compared = None
outer_best_compared = None

factorCount = 0
currentFactor = factors[factorCount]
allFactors = list(combined_df2.columns)

for i in allFactors:
    currentList = list(combined_df2.columns)
    currentList.remove(currentFactor)
    print('Factor is now ' + currentFactor)
    for f in currentList:
        print()
        print('-' * 80)
        print(f)
        print('-' * 80)
        # Prepare the data
        x = combined_df2[ [f] ].values  # extracting from data frame
        y = combined_df2[currentFactor].values  # extracting from data series

        # Fit a linear model
        model = sklearn.linear_model.LinearRegression()
        model.fit(x, y)

        # Evaluate linear model
        y_pred = model.predict(x)
        r2_score = sklearn.metrics.r2_score(y, y_pred)
        print("R^2: {}".format(r2_score))
        print("MAE: {}".format(sklearn.metrics.mean_absolute_error(y, y_pred)))
        print("MSE: {}".format(sklearn.metrics.mean_squared_error(y, y_pred)))

        # Track the best factor
        if r2_score > inner_best_r2:
            inner_best_factor = f
            inner_best_r2 = r2_score
            inner_best_compared = currentFactor
            print('*** New Best Factor ***')

        print()
        
    if factorCount + 1 < len(factors): 
        factorCount+=1
        currentFactor = factors[factorCount]

    print('Inner Best Factor is ' + str(inner_best_factor) + ' with ' + inner_best_compared + ' with R^2 =' + str(inner_best_r2))
    print()
    
    if inner_best_r2 > outer_best_r2:
        outer_best_factor = inner_best_factor
        outer_best_r2 = inner_best_r2
        outer_best_compared = inner_best_compared
print('Outer Best Factor is ' + str(outer_best_factor) + ' and ' + str(outer_best_compared) + ' with R^2 =' + str(outer_best_r2))