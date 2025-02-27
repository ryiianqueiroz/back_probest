from pydantic import BaseModel
import numpy as np
import joblib  # Carregar o modelo treinado
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI()

try:
    model = joblib.load("modelo.pkl")
    print("✅ Modelo carregado com sucesso!")
except FileNotFoundError:
    print("❌ Erro: O arquivo modelo.pkl não foi encontrado.")
    exit()

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
    vinhosDiluidos: float
    prolina: float
    proantocianinas: float

# 🔥 Configuração do CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://trabalho-probest.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.options("/{full_path:path}")  # Permite preflight requests
async def preflight_request(full_path: str):
    return JSONResponse(content={"message": "Preflight OK"}, headers={
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, GET, OPTIONS, DELETE, PUT",
        "Access-Control-Allow-Headers": "*",
    })

@app.get("/predict/")
async def check_predict():
    return JSONResponse(content={"message": "API de previsão está funcionando."}, headers={
        "Access-Control-Allow-Origin": "*",
    })

# Rota de teste para ver se a API está rodando
@app.get("/predict/")
async def check_predict():
    return {"message": "API de previsão está funcionando. Use POST para enviar dados."}

# Rota para previsão
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
        input_data.prolina,
        input_data.proantocianinas
    ]])

    # 📌 Fazer previsão usando o modelo treinado
    prediction = model.predict(user_input)

    return {"classe_predita": prediction[0]}
