# Import the required packages and modules
import warnings

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from server import *

# Ignore verbose warnings
warnings.filterwarnings("ignore")

# FastAPI App definition
app = FastAPI(
    title="LLM API Server",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces with HuggingfacePipelines",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def api_home():
    return {'detail': 'Welcome to FastAPI HuggingfacePipeline Deployment'}

# Adding chain route
add_routes(
    app,
    qa_chain,
    path="/qa_chain",
)

# Adding chain route
add_routes(
    app,
    llm,
    path="/distilgpt2",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
