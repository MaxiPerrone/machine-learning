import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv('../fraud-card-transactions.csv')

# elimina la columna "fraud" del DataFrame, para predecir si una transacción es fraudulenta o no
features = data.drop("fraud", axis=1)
# selecciona la columna "fraud" del DataFrame
labels = data["fraud"]

# features_train contendrá las características de entrenamiento,
# features_test contendrá las características de prueba,
# labels_train contendrá las etiquetas de clase de entrenamiento
# labels_test contendrá las etiquetas de clase de prueba
# dividiendo los datos en conjuntos de entrenamiento y prueba utilizando la función train_test_split
# test_size=0.2: el 20% de los datos se utilizarán para el conjunto de prueba
# y el 80% restante se utilizará para el conjunto de entrenamiento
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# crea un objeto clasificador RandomForestClassifier
rf_classifier = RandomForestClassifier(random_state=42)
# entrena el objeto clasificador
# -> rf_classifier tendra un modelo entrenado que puede ser utilizado para hacer predicciones sobre nuevos datos.
rf_classifier.fit(features_train, labels_train)

# sample(1) obtiene aleatoreamente una fila del dataset
transaction = data.sample(1).drop('fraud', axis=1)
print("\nRandomly sampled features for transaction:")
print(transaction)
prediction = rf_classifier.predict(transaction)
print("\nPrediction for random transaction:")
print("Fraud" if prediction[0] == 1 else "Legitimate")


#22.047408126627378,4.056965816703991,3.846141584961766,1.0,1.0,0.0,1.0,0.0
legitimate_transaction = pd.DataFrame({
    'distance_from_home': [22.047408126627378],
    'distance_from_last_transaction': [4.056965816703991],
    'ratio_to_median_purchase_price': [3.846141584961766],
    'repeat_retailer': [1],
    'used_chip': [1],
    'used_pin_number': [0],
    'online_order': [1]
})
prediction = rf_classifier.predict(legitimate_transaction)
print("\nPrediction for legitimate transaction:")
print("Fraud" if prediction[0] == 1 else "Legitimate")


# 136.3940947443769,0.16676990605090353,0.04852567092828011,1.0,0.0,0.0,1.0,1.0
fraud_transaction = pd.DataFrame({
    'distance_from_home': [136.3940947443769],
    'distance_from_last_transaction': [0.16676990605090353],
    'ratio_to_median_purchase_price': [1.04852567092828011],
    'repeat_retailer': [1],
    'used_chip': [0],
    'used_pin_number': [0],
    'online_order': [1]
})
prediction = rf_classifier.predict(fraud_transaction)
print("\nPrediction for fraud transaction:")
print("Fraud" if prediction[0] == 1 else "Legitimate")