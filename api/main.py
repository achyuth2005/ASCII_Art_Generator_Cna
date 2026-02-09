"""
FastAPI Backend for ASCII Art Generator React Frontend
Wraps the existing ascii_gen module for HTTP access
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import sys
import base64
from io import BytesIO

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from ascii_gen.online_generator import OnlineGenerator
from ascii_gen.production_training import ProductionCNNMapper
from ascii_gen.gradient_mapper import GradientMapper, GradientConfig, RAMP_STANDARD
from ascii_gen.llm_rewriter import LLMPromptRewriter

app = FastAPI(title="ASCII Art Generator API", version="1.0.0")

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173", "http://localhost:5174", "http://127.0.0.1:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models (lazy loaded)
online_gen = None
cnn_mapper = None
rewriter = None

def load_models():
    """Load models on first request."""
    global online_gen, cnn_mapper, rewriter
    
    if rewriter is None:
        try:
            rewriter = LLMPromptRewriter()
        except Exception as e:
            print(f"Warning: Could not load rewriter: {e}")
    
    if cnn_mapper is None:
        cnn_mapper = ProductionCNNMapper()
        try:
            cnn_mapper.load("models/production_cnn.pth")
        except:
            cnn_mapper.train(epochs=50)
    
    api_key = os.getenv("HF_TOKEN")
    if api_key and online_gen is None:
        online_gen = OnlineGenerator(api_key=api_key)


class GenerateRequest(BaseModel):
    prompt: str
    width: int = 125
    quality: str = "Standard"


class GenerateResponse(BaseModel):
    ascii: str
    image_url: str = ""
    rewritten_prompt: str = ""


@app.get("/")
async def root():
    return {"status": "ok", "message": "ASCII Art Generator API"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate ASCII art from a prompt."""
    load_models()
    
    if not online_gen:
        raise HTTPException(status_code=500, detail="HF_TOKEN not configured")
    
    prompt = request.prompt
    rewritten = prompt
    
    # Step 1: Rewrite prompt with LLM
    if rewriter:
        try:
            result = rewriter.rewrite(prompt, model_type="FLUX")
            rewritten = result.rewritten  # Extract the string from RewriteResult
        except Exception as e:
            print(f"Rewrite failed: {e}")
    
    # Step 2: Generate image
    try:
        image = online_gen.generate(rewritten)
        if image is None:
            raise HTTPException(status_code=500, detail="Image generation failed: No image returned from API (Check API Key or Network)")
    except Exception as e:
        with open("server_error.log", "a") as f:
            import traceback
            traceback.print_exc(file=f)
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")
    
    # Step 3: Convert to ASCII
    try:
        config = GradientConfig(
            width=request.width,
            ramp=RAMP_STANDARD,
            invert=False
        )
        mapper = GradientMapper(config)
        ascii_art = mapper.convert(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ASCII conversion failed: {str(e)}")
    
    # Step 4: Encode image as base64 for frontend
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    image_url = f"data:image/png;base64,{img_base64}"
    
    return GenerateResponse(
        ascii=ascii_art,
        image_url=image_url,
        rewritten_prompt=rewritten
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
