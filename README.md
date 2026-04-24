# Intelligent Document Verification System with AI Reasoning

A working Flask prototype for document upload, browser camera capture, OCR, OpenCV tamper detection, SQLAlchemy storage, visual highlighting, API verification, and downloadable PDF reports.

## Folder Structure

```text
HACTIVERSE/
  app.py
  requirements.txt
  README.md
  document_verifier/
    __init__.py
    config.py
    models.py
    ocr.py
    detection.py
    reasoning.py
    routes.py
    report.py
    sample_data.py
    static/
      css/styles.css
      js/app.js
    templates/
      index.html
      result.html
  tests/
    test_detection.py
    test_api.py
  instance/
    verifier.sqlite3
  storage/
    uploads/
    processed/
    samples/
```

## Prerequisites

Install Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

Install the native Tesseract OCR binary:

- Windows: install from `https://github.com/UB-Mannheim/tesseract/wiki`
- Add the install folder, often `C:\Program Files\Tesseract-OCR`, to `PATH`
- Verify with:

```powershell
tesseract --version
```

If Tesseract is not installed, the app still runs and stores a clear OCR setup error, but OCR text extraction requires the native binary.

Optional Groq reasoning:

```powershell
$env:GROQ_API_KEY="your_groq_key"
$env:GROQ_MODEL="llama-3.3-70b-versatile"
python app.py
```

The app uses Groq's OpenAI-compatible chat completions endpoint (`https://api.groq.com/openai/v1/chat/completions`) to turn OCR and computer-vision evidence into a clearer reviewer explanation. If no key is configured, or Groq is unavailable, it falls back to deterministic local reasoning.

## Dataset Context

The verifier presents public document-analysis datasets as useful domain references for evaluation and future extension:

- DocTamper: tampered text detection in document images, https://github.com/qcf-568/DocTamper
- MIDV-500: identity document recognition from mobile video, https://huggingface.co/papers/1807.05786
- RVL-CDIP: document image classification/layout reference, https://huggingface.co/datasets/aharley/rvl_cdip

Confirm each dataset's license and access terms for your use case.

## Train A Real Model

The app can use a trained classifier artifact at:

```text
models/document_tamper_model.joblib
```

Train from a labeled real dataset folder:

```powershell
python scripts/train_document_model.py --dataset-dir data/document_tamper --output models/document_tamper_model.joblib
```

Expected folder format:

```text
data/document_tamper/
  authentic/
    image_001.png
  tampered/
    image_002.png
```

Or train from a CSV manifest:

```powershell
python scripts/train_document_model.py --manifest data/document_tamper/manifest.csv --output models/document_tamper_model.joblib
```

Manifest format:

```csv
image_path,label
authentic/image_001.png,authentic
tampered/image_002.png,tampered
```

Once the model file exists, uploads automatically include a trained-model document-level tamper signal in addition to localized OpenCV findings.

## Run Locally

```powershell
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Test the Demo

Generate a sample tampered document:

```powershell
python -m document_verifier.sample_data
```

Then upload:

```text
storage/samples/sample_tampered_document.png
```

Camera testing:

1. Start the server.
2. Open the UI.
3. Click the camera button.
4. Allow browser camera permission.
5. Capture and verify.

Run automated tests:

```powershell
pytest
```

## API

Verify a file:

```powershell
curl.exe -F "document=@storage/samples/sample_tampered_document.png" http://127.0.0.1:5000/api/verify
```

Fetch a stored result:

```text
GET /api/documents/<document_id>
```

Download a PDF report:

```text
GET /documents/<document_id>/report
```
