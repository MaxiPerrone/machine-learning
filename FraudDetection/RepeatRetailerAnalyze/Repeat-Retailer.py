import pandas as pd
pd.set_option('display.max_columns', None)

data = pd.read_csv('../fraud-card-transactions.csv')

repeat_retailer_df = data[data["repeat_retailer"] == 1]
fraud_sequences = []
current_sequence = []

for index, row in repeat_retailer_df.iterrows():
    repeat_retailer = row['repeat_retailer']
    is_fraud = row['fraud']

    if is_fraud == 1:
        if current_sequence:
            fraud_sequences.append(current_sequence.copy())
        current_sequence = []
    else:
        current_sequence.append('Repeat retailer' if repeat_retailer == 1 else 'Not repeat retailer')

for i, sequence in enumerate(fraud_sequences[:10], start=1):
    print(f"Fraud sequence {i}: {', '.join(map(str, sequence))}")