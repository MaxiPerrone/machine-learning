import pandas as pd
pd.set_option('display.max_columns', None)

data = pd.read_csv('../fraud-card-transactions.csv')

ONLINE_ORDER = 'online_order'
FRAUD = 'fraud'

dataframe = data[[ONLINE_ORDER, FRAUD]]

total_online_orders = dataframe[ONLINE_ORDER].sum()
total_online_fraud = dataframe[(dataframe[FRAUD] == 1) & (dataframe[ONLINE_ORDER] == 1)][FRAUD].count()
fraud_rate_online = total_online_fraud / total_online_orders

total_offline_orders = len(dataframe) - total_online_orders
total_offline_fraud = dataframe[(dataframe[FRAUD] == 1) & (dataframe[ONLINE_ORDER] == 0)][FRAUD].count()
fraud_rate_offline = total_offline_fraud / total_offline_orders

print(f'Fraud rate online transactions {fraud_rate_online:.2%} ({total_online_fraud} cases out of {total_online_orders} online transactions)')
print(f'Fraud rate offline transactions {fraud_rate_offline:.2%} ({total_offline_fraud} cases out of {total_offline_orders} offline transactions)')

