from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from clarityhub import router as ch_router
from farmx import router as farmx_router
from fraud import router as fraud_router
from policysense import router as ps_router
from tts import router as tts_router
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(ch_router, prefix="/clarityhub", tags=["clarityhub"])
app.include_router(farmx_router, prefix="/farmx", tags=["farmx"])
app.include_router(fraud_router, prefix="/fraud", tags=["fraud"])
app.include_router(ps_router, prefix="/policysense", tags=["policysense"])
app.include_router(tts_router, prefix="/tts", tags=["tts"])

@app.get("/")
def read_root():
    return {"message": "Welcome to ArshSensei :3"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7070)