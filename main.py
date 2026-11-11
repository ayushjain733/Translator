import os
import io
import re
import unicodedata
import gc
from collections import OrderedDict
from difflib import SequenceMatcher
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from PIL import Image
import numpy as np
import easyocr
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
base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Compress Key/Value & Linear matrices to 8-bit (INT8) to save RAM and boost CPU speed
print("Compressing Key/Value & Linear matrices to 8-bit (INT8)...")
model = torch.ao.quantization.quantize_dynamic(
    base_model, {torch.nn.Linear}, dtype=torch.qint8
)
del base_model
gc.collect()
print("Optimized Model loaded successfully!")

# NLLB-200 Language Mapping
LANG_CODES = {
    "english": "eng_Latn", "spanish": "spa_Latn", "french": "fra_Latn",
    "german": "deu_Latn", "mandarin_simplified": "zho_Hans", "mandarin_traditional": "zho_Hant",
    "japanese": "jpn_Jpan", "korean": "kor_Hang", "arabic": "arb_Arab",
    "russian": "rus_Cyrl", "portuguese": "por_Latn", "italian": "ita_Latn",
    "turkish": "tur_Latn", "indonesian": "ind_Latn", "vietnamese": "vie_Latn",
    "assamese": "asm_Beng", "bengali": "ben_Beng", "bhojpuri": "bho_Deva",
    "bodo": "brx_Deva", "chhattisgarhi": "hne_Deva", "dogri": "dgo_Deva",
    "gujarati": "guj_Gujr", "hindi": "hin_Deva", "kannada": "kan_Knda",
    "kashmiri_arabic": "kas_Arab", "kashmiri_devanagari": "kas_Deva", "konkani": "gom_Deva",
    "magahi": "mag_Deva", "maithili": "mai_Deva", "malayalam": "mal_Mlym",
    "manipuri": "mni_Beng", "marathi": "mar_Deva", "nepali": "npi_Deva",
    "odia": "ory_Orya", "punjabi": "pan_Guru", "sanskrit": "san_Deva",
    "santali": "sat_Olck", "sindhi_arabic": "snd_Arab", "tamil": "tam_Taml",
    "telugu": "tel_Telu", "urdu": "urd_Arab"
}

# Pure single-script EasyOCR mapping
EASYOCR_LANG_CODES = {
    "english": ["en"], "spanish": ["es"], "french": ["fr"], "german": ["de"],
    "mandarin_simplified": ["ch_sim"], "mandarin_traditional": ["ch_tra"],
    "japanese": ["ja"], "korean": ["ko"], "arabic": ["ar"],
    "russian": ["ru"], "portuguese": ["pt"], "italian": ["it"],
    "turkish": ["tr"], "indonesian": ["id"], "vietnamese": ["vi"],
    "assamese": ["as"], "bengali": ["bn"], "bhojpuri": ["bho"],
    "bodo": ["hi"], "chhattisgarhi": ["hi"], "dogri": ["hi"],
    "gujarati": ["hi"], "hindi": ["hi"], "kannada": ["kn"],
    "kashmiri_arabic": ["ur"], "kashmiri_devanagari": ["hi"],
    "konkani": ["mr"], "magahi": ["mag"], "maithili": ["mai"],
    "malayalam": ["ml"], "manipuri": ["bn"], "marathi": ["mr"],
    "nepali": ["ne"], "odia": ["en"], "punjabi": ["pa"],
    "sanskrit": ["hi"], "santali": ["en"], "sindhi_arabic": ["ur"],
    "tamil": ["ta"], "telugu": ["te"], "urdu": ["ur"]
}

ISO_TO_NLLB = {
    "en": "eng_Latn", "es": "spa_Latn", "fr": "fra_Latn", "de": "deu_Latn",
    "zh-cn": "zho_Hans", "zh-tw": "zho_Hant", "ja": "jpn_Jpan", "ko": "kor_Hang",
    "ar": "arb_Arab", "ru": "rus_Cyrl", "pt": "por_Latn", "it": "ita_Latn",
    "tr": "tur_Latn", "id": "ind_Latn", "vi": "vie_Latn", "hi": "hin_Deva",
    "bn": "ben_Beng", "gu": "guj_Gujr", "kn": "kan_Knda", "ml": "mal_Mlym",
    "mr": "mar_Deva", "ne": "npi_Deva", "pa": "pan_Guru", "ta": "tam_Taml",
    "te": "tel_Telu", "ur": "urd_Arab"
}

ocr_readers = {}
MAX_CACHE_SIZE = 30
translation_cache = OrderedDict()
last_live_state = {"text": "", "translation": "", "src": "", "tgt": ""}

def get_ocr_reader(src_lang: str):
    langs = EASYOCR_LANG_CODES.get(src_lang.lower(), ["en"])
    key = "_".join(langs)
    if key not in ocr_readers:
        print(f"Loading EasyOCR neural model for {langs}... (Downloads once on first use)")
        ocr_readers[key] = easyocr.Reader(langs, gpu=torch.cuda.is_available())
    return ocr_readers[key]

def clean_ocr_text(text: str) -> str:
    text = re.sub(r'[「」『』|¦`^<>_\[\]{}]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def has_valid_script_char(text: str) -> bool:
    for ch in text:
        if unicodedata.category(ch).startswith(('L', 'M')):
            return True
    return False

def extract_scene_text(image: Image.Image, src_lang: str) -> str:
    if image.width < 500 or image.height < 500:
        image = image.resize((image.width * 2, image.height * 2), Image.Resampling.LANCZOS)

    img_np = np.array(image.convert("RGB"))
    reader = get_ocr_reader(src_lang)

    results = reader.readtext(
        img_np,
        detail=1,
        paragraph=False,
        min_size=3,
        text_threshold=0.35,
        low_text=0.25,
        link_threshold=0.3,
        mag_ratio=1.5,
        add_margin=0.15,
        contrast_ths=0.1,
        adjust_contrast=0.5
    )

    valid_boxes = []
    for (bbox, text, prob) in results:
        cleaned = clean_ocr_text(text)
        if not has_valid_script_char(cleaned):
            continue

        min_prob = 0.12 if len(cleaned) <= 4 else 0.30
        if prob >= min_prob:
            top_y = min(pt[1] for pt in bbox)
            left_x = min(pt[0] for pt in bbox)
            valid_boxes.append((top_y, left_x, cleaned))

    valid_boxes.sort(key=lambda item: (round(item[0] / 25), item[1]))
    combined_text = " ".join(item[2] for item in valid_boxes)
    return clean_ocr_text(combined_text)

def update_cache(key: str, value: str):
    if key in translation_cache:
        translation_cache.move_to_end(key)
    translation_cache[key] = value
    if len(translation_cache) > MAX_CACHE_SIZE:
        translation_cache.popitem(last=False)

def perform_translation(text: str, src: str, tgt: str) -> str:
    if src.lower() == "auto":
        try:
            detected_iso = langdetect.detect(text)
            src_code = ISO_TO_NLLB.get(detected_iso, "eng_Latn")
        except Exception:
            src_code = "eng_Latn"
    else:
        src_code = LANG_CODES.get(src.lower(), "eng_Latn")

    tgt_code = LANG_CODES.get(tgt.lower(), "eng_Latn")

    if src_code == tgt_code:
        return text

    cache_key = f"{text}_{src_code}_{tgt_code}"
    if cache_key in translation_cache:
        translation_cache.move_to_end(cache_key)
        return translation_cache[cache_key]

    tokenizer.src_lang = src_code
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    input_len = inputs["input_ids"].shape[1]

    max_new_tokens = 12 if len(text) <= 4 else min(1024, max(24, int(input_len * 2.5)))
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_code)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_new_tokens=max_new_tokens,
            num_beams=2,
            use_cache=True,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    del inputs, outputs
    update_cache(cache_key, result)
    return result

@app.post("/translate/text")
async def translate_text(text: str = Form(...), src_lang: str = Form(...), tgt_lang: str = Form(...)):
    if not text.strip():
        return {"error": "Input text cannot be empty."}
    result = perform_translation(text.strip(), src_lang, tgt_lang)
    return {"original": text, "translation": result}

@app.post("/translate/image")
async def translate_image(
    file: UploadFile = File(...),
    src_lang: str = Form(...),
    tgt_lang: str = Form(...),
    is_live: str = Form("false")
):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    extracted_text = extract_scene_text(image, src_lang.lower())

    if not extracted_text:
        return {"extracted_text": "", "translation": "", "skipped": True}

    if is_live == "true" and last_live_state["src"] == src_lang and last_live_state["tgt"] == tgt_lang:
        similarity = SequenceMatcher(None, extracted_text, last_live_state["text"]).ratio()
        
        if similarity > 0.90 and last_live_state["translation"]:
            return {
                "extracted_text": last_live_state["text"],
                "translation": last_live_state["translation"],
                "cached": True
            }
        
        if similarity < 0.20 and len(translation_cache) > 5:
            translation_cache.clear()
            gc.collect()

    result = perform_translation(extracted_text, src_lang, tgt_lang)

    if is_live == "true":
        last_live_state.update({
            "text": extracted_text,
            "translation": result,
            "src": src_lang,
            "tgt": tgt_lang
        })

    return {"extracted_text": extracted_text, "translation": result, "cached": False}

@app.get("/")
async def serve_frontend():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    elif os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    return {"error": "index.html not found in project folder"}