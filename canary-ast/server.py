import os
import io
import json
import torch
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import soundfile as sf
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Global model variables
model = None
processor = None
device = None

def load_model():
    global model, processor, device
    
    logger.info("Loading Canary-1B Flash model...")
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    model_id = "nvidia/canary-1b-flash"
    
    # Load processor and model
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device)
    
    model.eval()
    logger.info("Canary model loaded successfully")

@app.on_event("startup")
async def startup():
    load_model()

class TranslateRequest(BaseModel):
    pcm16_hex: str
    sr: int
    src_lang: str
    tgt_lang: str

@app.post("/ast")
async def translate(request: TranslateRequest):
    try:
        # Convert hex to PCM bytes
        pcm_bytes = bytes.fromhex(request.pcm16_hex)
        if not pcm_bytes:
            return {"transcript": "", "translation": ""}
        
        # Convert PCM to numpy array
        audio_array = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Prepare the prompt for Canary based on task
        # Canary uses special tokens for different tasks
        if request.src_lang == request.tgt_lang:
            # Just transcription
            task_token = f"<|{request.src_lang}|><|transcribe|>"
        else:
            # Translation
            if request.src_lang == "es" and request.tgt_lang == "en":
                task_token = "<|es|><|translate|><|en|>"
            elif request.src_lang == "en" and request.tgt_lang == "es":
                task_token = "<|en|><|translate|><|es|>"
            else:
                task_token = f"<|{request.src_lang}|><|translate|><|{request.tgt_lang}|>"
        
        logger.info(f"Task: {task_token}")
        
        # Process audio
        inputs = processor(
            audio=audio_array,
            sampling_rate=request.sr,
            return_tensors="pt",
            truncation=False,
            padding="longest",
            return_attention_mask=True
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Add task token to the input
        # Canary expects the task token as a prompt
        inputs["prompt_ids"] = processor.tokenizer(
            task_token,
            return_tensors="pt"
        ).input_ids.to(device)
        
        # Generate translation
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                num_beams=5,
                do_sample=False
            )
        
        # Decode the output
        transcription = processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )[0]
        
        logger.info(f"Result: {transcription[:100]}...")
        
        # For Canary, the output is the translation when translate task is used
        if request.src_lang != request.tgt_lang:
            return {"transcript": "", "translation": transcription}
        else:
            return {"transcript": transcription, "translation": ""}
        
    except Exception as e:
        logger.error(f"Translation error: {e}", exc_info=True)
        return {"transcript": "", "translation": ""}

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}
