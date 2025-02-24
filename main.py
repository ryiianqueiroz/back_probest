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
    alcool: float
    acidoMalico: float
    cinza: float
    alcalinidadeCinzas: float
    magnesio: float
    fenoisTotais: float
    flavonoides: float
    fenoisNaoFlavonoides: float
    intensidadeCor: float
    matiz: float
    vinhosDiluidos: float  # 🚀 Troquei para sem acento
    prolina: float

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
    # 📌 Criar array com os valores corretos
    user_input = np.array([[  
        input_data.alcool, 
        input_data.acidoMalico, 
        input_data.cinza,
        input_data.alcalinidadeCinzas, 
        input_data.magnesio,
        input_data.fenoisTotais, 
        input_data.flavonoides,
        input_data.fenoisNaoFlavonoides, 
        input_data.intensidadeCor,
        input_data.matiz, 
        input_data.vinhosDiluidos, 
        input_data.prolina
    ]])

    # 📌 Fazer previsão
    prediction = model.predict(user_input)

    return {"prediction": prediction[0]}
