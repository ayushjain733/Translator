import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import pytesseract
from PIL import Image
import io
import torch
import langdetect

# Limit intra-op threads to prevent CPU thrashing
torch.set_num_threads(4)

app = FastAPI(title="Local Multi-Language Translator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME = "facebook/nllb-200-1.3B"

print(f"Loading {MODEL_NAME}... (This will take a moment on CPU)")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
print("Model loaded successfully!")

# User's Supported Languages
LANG_CODES = {
    "english": "eng_Latn", "spanish": "spa_Latn", "french": "fra_Latn", "german": "deu_Latn",
    "mandarin_simplified": "zho_Hans", "mandarin_traditional": "zho_Hant", "japanese": "jpn_Jpan",
    "korean": "kor_Hang", "arabic": "arb_Arab", "russian": "rus_Cyrl", "portuguese": "por_Latn",
    "italian": "ita_Latn", "turkish": "tur_Latn", "indonesian": "ind_Latn", "vietnamese": "vie_Latn",
    "assamese": "asm_Beng", "bengali": "ben_Beng", "bhojpuri": "bho_Deva", "bodo": "brx_Deva",
    "chhattisgarhi": "hne_Deva", "dogri": "dgo_Deva", "gujarati": "guj_Gujr", "hindi": "hin_Deva",
    "kannada": "kan_Knda", "kashmiri_arabic": "kas_Arab", "kashmiri_devanagari": "kas_Deva",
    "konkani": "gom_Deva", "magahi": "mag_Deva", "maithili": "mai_Deva", "malayalam": "mal_Mlym",
    "manipuri": "mni_Beng", "marathi": "mar_Deva", "nepali": "npi_Deva", "odia": "ory_Orya",
    "punjabi": "pan_Guru", "sanskrit": "san_Deva", "santali": "sat_Olck", "sindhi_arabic": "snd_Arab",
    "tamil": "tam_Taml", "telugu": "tel_Telu", "urdu": "urd_Arab"
}

# Map 2-letter ISO codes from langdetect to NLLB codes
ISO_TO_NLLB = {
    'en': 'eng_Latn', 'es': 'spa_Latn', 'fr': 'fra_Latn', 'de': 'deu_Latn',
    'zh-cn': 'zho_Hans', 'zh-tw': 'zho_Hant', 'ja': 'jpn_Jpan', 'ko': 'kor_Hang',
    'ar': 'arb_Arab', 'ru': 'rus_Cyrl', 'pt': 'por_Latn', 'it': 'ita_Latn',
    'tr': 'tur_Latn', 'id': 'ind_Latn', 'vi': 'vie_Latn', 'hi': 'hin_Deva',
    'bn': 'ben_Beng', 'gu': 'guj_Gujr', 'kn': 'kan_Knda', 'ml': 'mal_Mlym', 
    'mr': 'mar_Deva', 'ne': 'npi_Deva', 'pa': 'pan_Guru', 'ta': 'tam_Taml', 
    'te': 'tel_Telu', 'ur': 'urd_Arab'
}

translation_cache = {}

def perform_translation(text: str, src: str, tgt: str) -> dict:
    detected_iso = None
    
    # Auto-detect language logic
    if src == "auto":
        try:
            detected_iso = langdetect.detect(text)
            src_code = ISO_TO_NLLB.get(detected_iso, "eng_Latn") # Default to english if unknown
        except:
            src_code = "eng_Latn"
    else:
        src_code = LANG_CODES.get(src.lower(), "eng_Latn")

    tgt_code = LANG_CODES.get(tgt.lower(), "hin_Deva")
    
    cache_key = f"{text}_{src_code}_{tgt_code}"
    if cache_key in translation_cache:
        return {"result": translation_cache[cache_key], "detected": detected_iso}

    tokenizer.src_lang = src_code
    inputs = tokenizer(text, return_tensors="pt")
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_code)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            forced_bos_token_id=forced_bos_token_id, 
            max_length=256,
            num_beams=2
        )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    translation_cache[cache_key] = result
    
    return {"result": result, "detected": detected_iso}

@app.post("/translate/text")
async def translate_text(text: str = Form(...), src_lang: str = Form(...), tgt_lang: str = Form(...)):
    if not text.strip():
        return {"error": "Input text cannot be empty."}
    trans_data = perform_translation(text.strip(), src_lang, tgt_lang)
    return {"original": text, "translation": trans_data["result"], "detected": trans_data["detected"]}

@app.post("/translate/image")
async def translate_image(file: UploadFile = File(...), src_lang: str = Form(...), tgt_lang: str = Form(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    
    extracted_text = pytesseract.image_to_string(image).strip()
    if not extracted_text:
        return {"error": "No text detected in the image."}
        
    trans_data = perform_translation(extracted_text, src_lang, tgt_lang)
    return {"extracted_text": extracted_text, "translation": trans_data["result"], "detected": trans_data["detected"]}

# Mount the static folder to serve the frontend
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")