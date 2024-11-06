import pandas as pd
pd.set_option('display.max_columns', None)

data = pd.read_csv('../fraud-card-transactions.csv')

missing_values = data.isnull().any(axis=1)
duplicate_rows = data[data.duplicated()]

print("Missing values from the dataset:")
print(missing_values)

print("Duplicated rows from the dataset:")
print(duplicate_rows)

dataframe = data[["used_chip", "used_pin_number", "fraud"]]
total_transactions = len(dataframe)
total_fraud = dataframe["fraud"].sum()
total_fraud_by_chip = dataframe[dataframe["used_chip"] == 1]["fraud"].sum()
total_fraud_by_pin = dataframe[dataframe["used_pin_number"] == 1]["fraud"].sum()

print("Total transactions: ", total_transactions)
print("Total frauds: ", total_fraud)
print("Fraud cases using chip: ", total_fraud_by_chip)
print("Fraud cases using pin number: ", total_fraud_by_pin)