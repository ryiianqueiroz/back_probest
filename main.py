from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

app = FastAPI()

# 📌 Modelo para receber os dados do frontend
class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float
    feature5: float
    # Adicione mais atributos conforme necessário

# 📌 Dados de exemplo (simulação de dataset para treinamento)
data = pd.DataFrame({
    "feature1": np.random.rand(100),
    "feature2": np.random.rand(100),
    "feature3": np.random.rand(100),
    "feature4": np.random.rand(100),
    "feature5": np.random.rand(100),
    "class": np.random.choice(["A", "B", "C"], 100)  # Classes de exemplo
})

X_train = data.drop(columns=["class"])
y_train = data["class"]

# 📌 Treinar modelo
model = GaussianNB()
model.fit(X_train, y_train)

@app.post("/predict/")
def predict(input_data: InputData):
    # 📌 Criar array com os valores do formulário
    user_input = np.array([[input_data.feature1, input_data.feature2, input_data.feature3, input_data.feature4, input_data.feature5]])

    # 📌 Fazer previsão
    prediction = model.predict(user_input)

    return {"prediction": prediction[0]}
