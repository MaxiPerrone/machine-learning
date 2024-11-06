import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)

data = pd.read_csv('../fraud-card-transactions.csv')

correlation_df = data[["ratio_to_median_purchase_price", "fraud"]]
correlation = correlation_df["ratio_to_median_purchase_price"].corr(correlation_df["fraud"])
print(f"Correlation between trx amount and fraud {correlation}")

avg_non_fraud_transactions = correlation_df[correlation_df["fraud"] == 0]["ratio_to_median_purchase_price"].mean()
avg_fraud_transactions = correlation_df[correlation_df["fraud"] == 1]["ratio_to_median_purchase_price"].mean()

categories = ["Non fraudulent", "Fraudulent"]
average_ratio = [avg_non_fraud_transactions, avg_fraud_transactions]
plt.bar(categories, average_ratio, color=["blue", "red"])
plt.title("Ratio to median purchase price")
plt.xlabel("Fraud category")
plt.ylabel("Average ratio to median purchase price")
plt.show()