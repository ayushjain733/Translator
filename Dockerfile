FROM python:3.13

# Set up the working directory inside the container
WORKDIR /code

# Install Tesseract OCR for image translation
RUN apt-get update && apt-get install -y tesseract-ocr tesseract-ocr-all

# Copy and install python dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of the application
COPY . .

# Start the FastAPI server (Hugging Face Spaces uses port 7860 by default)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]