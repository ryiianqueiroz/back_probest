import joblib
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, confusion_matrix, classification_report

# Carregar dataset
try:
    df = pd.read_csv("Wine.csv")
    print("Dataset carregado com sucesso!")
except FileNotFoundError:
    print("Erro: O arquivo Wine.csv não foi encontrado.")
    exit()

# Definir variável alvo e preditores
target_column = "class"
if target_column not in df.columns:
    print(f"Erro: A coluna '{target_column}' não foi encontrada no dataset.")
    exit()

X = df.drop(columns=[target_column])
y = df[target_column]

# Converter colunas numéricas que possam estar erradas
X = X.apply(pd.to_numeric, errors='coerce')
X.fillna(X.mean(), inplace=True)

# Divisão Treino/Teste (70% treino, 30% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treinar o modelo
model = GaussianNB()
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)

# Calcular métricas
accuracy = accuracy_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Salvar o modelo compactado
joblib.dump(model, 'modelo.pkl', compress=3)
print("Modelo salvo como 'modelo.pkl'")
