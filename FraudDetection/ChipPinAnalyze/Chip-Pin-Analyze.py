import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)

data = pd.read_csv('../fraud-card-transactions.csv')

chip_pin_df = data[["used_chip", "used_pin_number", "fraud"]]
total_transactions = len(chip_pin_df)
total_fraud = chip_pin_df["fraud"].sum()
total_fraud_by_chip = chip_pin_df[chip_pin_df["used_chip"] == 1]["fraud"].sum()
total_fraud_by_pin = chip_pin_df[chip_pin_df["used_pin_number"] == 1]["fraud"].sum()

labels_chip = ["Legitimate transactions", "Fraud"]
sizes_chip = [total_transactions - total_fraud_by_chip, total_fraud_by_chip]
colors_chip = ["#90A4AE", "#EBEDEF"]

labels_pin = ["Legitimate transactions", "Fraud"]
sizes_pin = [total_transactions - total_fraud_by_pin, total_fraud_by_pin]
colors_pin = ["#90A4AE", "#EBEDEF"]


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.pie(sizes_chip, labels=labels_chip, colors=colors_chip, startangle=140)
plt.axis("equal")
plt.title("Chip transactions")

plt.subplot(1, 2, 2)
plt.pie(sizes_pin, labels=labels_pin, colors=colors_pin, startangle=140)
plt.axis("equal")
plt.title("Pin transactions")

plt.suptitle("Fraud cases in chip & pin transactions")
plt.show()