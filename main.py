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

@app.post("/predict/")
async def predict(data: dict):
    try:
        # Pegando os valores e convertendo para array numpy
        input_data = np.array(list(data.values())).reshape(1, -1)

        # Fazendo a previsão
        prediction = model.predict(input_data)

        # 🔥 Convertendo a previsão para int padrão do Python
        prediction = int(prediction[0])  # Converte numpy.int64 → int

        return JSONResponse(content={"prediction": prediction})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
