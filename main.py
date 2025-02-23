from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from sklearn.naive_bayes import GaussianNB

app = FastAPI()

# 🔥 Configuração do CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 📌 Modelo para receber os dados do frontend
class InputData(BaseModel):
    Álcool: float
    Ácido_Málico: float
    Cinza: float
    Alcalinidade_das_Cinzas: float
    Magnésio: float
    Fenóis_Totais: float
    Flavonoides: float
    Fenois_não_Flavonoides: float
    Intensidade_Cor: float
    Matiz: float
    OD280_OD315_Vinhos_Diluídos: float
    Prolina: float

# 📌 Simulação de dataset para treinamento com numpy
np.random.seed(0)  # Para garantir a mesma aleatoriedade sempre que rodar

# Gerar 100 amostras de dados
num_samples = 100
X_train = np.random.rand(num_samples, 12)  # 12 características, como no seu dataset
y_train = np.random.choice(["A", "B", "C"], num_samples)  # Classes de exemplo

# 📌 Treinar modelo
model = GaussianNB()
model.fit(X_train, y_train)

@app.get("/predict/")
async def check_predict():
    return {"message": "API de previsão está funcionando. Use POST para enviar dados."}

@app.post("/predict/")
def predict(input_data: InputData):
    # 📌 Criar array com os valores do formulário
    user_input = np.array([[
        input_data.Álcool, input_data.Ácido_Málico, input_data.Cinza,
        input_data.Alcalinidade_das_Cinzas, input_data.Magnésio,
        input_data.Fenóis_Totais, input_data.Flavonoides,
        input_data.Fenois_não_Flavonoides, input_data.Intensidade_Cor,
        input_data.Matiz, input_data.OD280_OD315_Vinhos_Diluídos,
        input_data.Prolina
    ]])

    # 📌 Fazer previsão
    prediction = model.predict(user_input)

    return {"prediction": prediction[0]}
