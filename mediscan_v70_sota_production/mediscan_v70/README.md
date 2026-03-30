# MediScan AI v7.0 — Medical VLM Analysis Engine

Production-grade medical imaging intelligence pipeline with **16 open-source models** spanning 40+ medical imaging modalities.

## Architecture

**14-Stage Pipeline:**
Ingestion → Quality Assessment → Modality Detection → MONAI Preprocessing → Intelligent Routing → Parallel Model Execution → Reasoning Engine → Dynamic Fusion → Uncertainty Analysis → Self-Reflection → Clinical Safety → Governance → Report Generation → Translation

**18 Models across 5 categories:**

| Category | Models | Speciality |
|----------|--------|------------|
| Foundation VLMs | Hulu-Med (7B/14B/32B), MedGemma (4B/27B) | General medical imaging |
| Reasoning | MediX-R1 (2B/8B/30B) | Chain-of-thought medical reasoning |
| 3D Volumetric | Med3DVLM, **Merlin** (Stanford MIMI) | CT/MRI volume analysis |
| Domain Specialists | **CheXagent-2** (CXR), **PathGen** (pathology), **RETFound** (retinal), **RadFM** (radiology) | Modality-specific SOTA |
| Classifiers | BiomedCLIP | Zero-shot verification |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run analysis
python main.py --file chest_xray.jpg --question "Analyze this chest X-ray"

# Run API server
uvicorn api_server:app --host 0.0.0.0 --port 8000

# List all models
python main.py --list-models --file dummy
```

## Docker

```bash
docker build -t mediscan-ai:7.0 .
docker run --gpus all -p 8000:8000 \
  -e MEDISCAN_API_KEY=your-secret-key \
  mediscan-ai:7.0
```

## Configuration

All config files are in `./config/`:
- `model_config.yaml` — Model IDs, weights, capabilities
- `pipeline_config.yaml` — Stage settings, fusion strategy, RAG config
- `hardware_config.yaml` — GPU allocation, parallelism

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MEDISCAN_API_KEY` | API key for protected endpoints | _(none — open)_ |
| `MEDISCAN_CORS_ORIGINS` | Comma-separated allowed origins | `http://localhost:3000` |
| `MEDISCAN_PHI_SALT` | Salt for PHI de-identification | _(random per run)_ |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/analyze` | Upload image + get full analysis |
| `POST` | `/analyze/stream` | SSE streaming with real-time stage updates |
| `POST` | `/analyze/batch` | Batch-analyze multiple images in one call |
| `POST` | `/analyze/structured` | Typed Pydantic output for EHR integration |
| `POST` | `/analyze/url` | Analyze image from URL |
| `POST` | `/chat` | Conversational medical AI |
| `GET` | `/health` | Health check with model status |
| `GET` | `/models` | List all registered models |
| `POST` | `/models/{key}/load` | Load a specific model into GPU |
| `GET` | `/patients/{id}/history` | Patient history (requires API key) |
| `GET` | `/metrics` | Performance metrics and drift stats |
| `GET` | `/languages` | List supported translation languages |

## Security

- CORS restricted to configured origins (no wildcard)
- API key authentication on patient data endpoints
- PHI de-identification via HIPAA compliance module
- Path traversal protection on all file uploads
- Audit logging for all analyses

## Supported Modalities

Radiology (X-ray, CT, MRI, Mammography, Fluoroscopy, Angiography), Ultrasound (Obstetric, Cardiac, Doppler), Nuclear Medicine (PET, SPECT), Pathology (Histopathology, Cytology, IHC), Ophthalmology (Fundoscopy, OCT), Dermatology (Dermoscopy), Dental (Panoramic, Intraoral), Cardiology (ECG, Echocardiography), Endoscopy, Surgical Video, and more.

## Translation

Supports 10 Indian languages via Sarvam AI: Hindi, Tamil, Kannada, Telugu, Bengali, Marathi, Gujarati, Malayalam, Punjabi, Odia.

## License

Research use only. Not approved for clinical diagnosis.
