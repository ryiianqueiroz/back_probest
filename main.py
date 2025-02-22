from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib

# Carregar modelo treinado
model = joblib.load("modelo.pkl")

app = FastAPI()

# Configuração do CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo para receber os dados do frontend
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

@app.get("/")
async def root():
    return {"message": "API funcionando no Vercel!"}

@app.post("/predict/")
def predict(input_data: InputData):
    # Criar array com os valores do formulário
    user_input = np.array([[ 
        input_data.Álcool, input_data.Ácido_Málico, input_data.Cinza,
        input_data.Alcalinidade_das_Cinzas, input_data.Magnésio,
        input_data.Fenóis_Totais, input_data.Flavonoides,
        input_data.Fenois_não_Flavonoides, input_data.Intensidade_Cor,
        input_data.Matiz, input_data.OD280_OD315_Vinhos_Diluídos,
        input_data.Prolina
    ]])

    # Fazer previsão
    prediction = model.predict(user_input)

    return {"prediction": prediction[0]}
