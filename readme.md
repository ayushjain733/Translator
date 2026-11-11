# 🌍 Local AI Multilingual Translator & Live Screen Reader

A lightweight, 100% local, AI-powered translation web application designed to run efficiently on standard consumer hardware (e.g., 16GB RAM, no GPU). It features text translation, image OCR, and a **seamless Live Screen Translation mode** with an Always-on-Top Floating HUD—perfect for reading comics, manga, webtoons, or live applications.

## ✨ Key Features

*   **100% Local & Private:** No API keys, no cloud subscriptions, and no data leaves your machine.
*   **Live Screen Translation:** Continuously captures your screen, window, or browser tab. Select a specific region (crop) to translate subtitles or comic speech bubbles in real-time.
*   **Always-on-Top HUD:** Uses the modern Document Picture-in-Picture API to detach the translation results into a floating mini-window that stays pinned above your other desktop apps.
*   **Advanced Scene OCR (EasyOCR):** Neural network-based text extraction optimized for complex backgrounds, speech bubbles, and single-character interjections (crucial for CJK languages).
*   **High-Quality Translation:** Powered by Meta's `NLLB-200-1.3B` (No Language Left Behind), supporting highly accurate translations across 40+ global and regional Indian languages.
*   **CPU-Optimized (8-bit Quantization):** Uses PyTorch dynamic INT8 quantization to compress the 1.3B model's matrix footprint by ~65%, doubling inference speed and preventing out-of-memory crashes on 16GB RAM machines.
*   **Smart LRU Caching & Deduplication:** Prevents CPU thrashing by caching recent translations and using fuzzy string matching to instantly skip static or idle screen frames.

## 🛠️ Tech Stack

*   **Backend:** Python, FastAPI, Uvicorn
*   **AI/ML:** PyTorch, Hugging Face `transformers`
*   **OCR:** EasyOCR (CRAFT Text Detector)
*   **Frontend:** Vanilla HTML, CSS, JavaScript (served directly by FastAPI)

## 🚀 Installation & Setup

### Prerequisites
*   Python 3.8 or higher installed on your system.
*   Basic C++ build tools (sometimes required by EasyOCR on Windows).

### 1. Clone the Repository
```bash
git clone https://github.com/ayushjain733/Translator.git
cd Translator
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install fastapi uvicorn transformers torch torchvision torchaudio easyocr pillow langdetect python-multipart numpy
```
*(Note: If you have a GPU and want to use it, install the CUDA version of PyTorch from the official PyTorch website before running the above command).*

### 4. Run the Application
```bash
uvicorn main:app --reload
```
*Note: The very first time you run this, it will download the NLLB-1.3B model (~5GB) and the EasyOCR language models. Subsequent startups will be nearly instant.*

### 5. Access the Web UI
Open your modern web browser (Chrome or Edge recommended for Picture-in-Picture support) and navigate to:
`http://localhost:8000`

## 📖 How to Use

### Text & Image Translation
1. Select your Source and Target languages from the dropdowns. (Use "Auto-Detect" for text if unsure).
2. Type text or upload an image.
3. Click **Translate Once**.

### Live Screen Translation (Ideal for Comics/Webtoons)
1. **Select the exact Source Language** (e.g., Korean, Japanese) from the dropdown. *Do not use Auto-Detect for Live OCR.*
2. Click **Start Live Screen Translate** and select the window or screen you want to capture.
3. **Crop (Optional):** Click and drag a cyan box on the video preview to focus the OCR on a specific region (like a chat box or speech bubble). Double-click to reset to full screen.
4. **Pop-out HUD:** Click the **Pop-Out Always-on-Top** button to spawn a floating dark-mode window containing your live translations. You can move this over your game or comic reader!

## 🧠 Under the Hood (Optimizations)

*   **INT8 Quantization:** `torch.ao.quantization.quantize_dynamic` is applied to all Linear layers. This reduces the model size in memory from ~5.2GB to ~1.5GB, making it entirely feasible to run a 1.3B parameter model smoothly on a laptop CPU.
*   **Screen-Aware Memory Management:** The backend maintains a 30-item sliding window cache. If the OCR detects a >90% similarity between frames, it returns the cached translation instantly (0ms inference). If similarity drops below 20% (indicating a page turn or scene change), it aggressively flushes the cache and forces Python Garbage Collection to free up RAM.
*   **Tuned Single-Character OCR:** The EasyOCR pipeline is configured with lowered `min_size` and `text_threshold` limits to accurately capture isolated 1-2 character speech bubbles (e.g., "Ah...", "Oh!") that default OCR settings usually ignore as noise.

## 🤝 Contributing
Feel free to open issues or submit pull requests! Suggestions for adding new local features (like Text-to-Speech or automatic manga bubble masking) are always welcome.

## 📄 License
This project is licensed under the MIT License. 
*Note: The NLLB model is subject to the CC-BY-NC 4.0 license by Meta.*