# IA Detector

API REST para detectar imagenes y videos generados o manipulados por Inteligencia Artificial.

## Caracteristicas

- Deteccion de imagenes generadas por IA (DALL-E, Midjourney, Stable Diffusion, etc.)
- Analisis de videos mediante extraccion de keyframes
- Soporte para videos desde URL externa con autenticacion
- Respuesta con score de veracidad y deteccion de fraude

## Instalacion

### Requisitos

- Python 3.10+
- pip

### Pasos

```bash
# Clonar repositorio
git clone <repo-url>
cd ia-detector

# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

## Uso

### Iniciar servidor

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Documentacion interactiva

Una vez iniciado el servidor, accede a:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

## API Endpoints

### Health Check

```
GET /health
```

Verifica el estado del servicio.

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "model_loaded": true
}
```

### Analizar Imagen

```
POST /analyze
Content-Type: multipart/form-data
```

Analiza una imagen para detectar si fue generada por IA.

**Request:**
```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@imagen.jpg"
```

**Response:**
```json
{
  "is_ai_generated": true,
  "confidence": 0.87,
  "label": "ai_generated",
  "processing_time_ms": 234.5
}
```

### Analizar Video (Upload)

```
POST /analyze/video
Content-Type: multipart/form-data
```

Analiza un video subido directamente.

**Request:**
```bash
curl -X POST http://localhost:8000/analyze/video \
  -F "file=@video.mp4"
```

**Response:**
```json
{
  "veracity_score": 15,
  "fraud_detected": true,
  "confidence": 87,
  "fraud_types": ["ai_generated"],
  "details": "Video analysis detected 38 of 45 frames (84.4%) as AI-generated content. High confidence (87%) in detection. The video is highly likely to be AI-generated or heavily manipulated."
}
```

### Analizar Video desde URL

```
POST /analyze/video/url
Content-Type: application/json
```

Analiza un video desde una URL externa con autenticacion opcional.

**Request:**
```bash
curl -X POST http://localhost:8000/analyze/video/url \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://api.example.com/video/12345",
    "authorization": "Bearer your-token-here"
  }'
```

**Response:**
```json
{
  "veracity_score": 15,
  "fraud_detected": true,
  "confidence": 87,
  "fraud_types": ["ai_generated"],
  "details": "Video analysis detected 38 of 45 frames (84.4%) as AI-generated content. High confidence (87%) in detection. The video is highly likely to be AI-generated or heavily manipulated."
}
```

## Formato de Respuesta (Video)

| Campo | Tipo | Descripcion |
|-------|------|-------------|
| `veracity_score` | int (0-100) | Score de veracidad. 100 = autentico, 0 = falso/IA |
| `fraud_detected` | bool | `true` si se detecta manipulacion o generacion IA |
| `confidence` | int (0-100) | Nivel de confianza en la deteccion |
| `fraud_types` | list[str] | Lista de tipos de fraude detectados |
| `details` | str | Descripcion detallada del analisis |

### Tipos de Fraude Detectados

| Tipo | Descripcion |
|------|-------------|
| `ai_generated` | Contenido generado por IA (DALL-E, Midjourney, Stable Diffusion, etc.) |
| `deepfake` | Rostros manipulados o intercambiados (face-swap) |
| `image_editing` | Edicion de imagen detectada via Error Level Analysis (Photoshop, retoques) |
| `temporal_inconsistency` | Cambios abruptos entre frames (lip-sync manipulation, video splicing) |

### Logica de Deteccion

- **fraud_detected = true**: Cuando >30% de los keyframes son detectados como IA o se detecta inconsistencia temporal
- **veracity_score**: Inverso del porcentaje de frames IA (100 - %IA)
- **confidence**: Basado en la consistencia del analisis y confianza individual de cada frame

### Metodos de Deteccion

El sistema utiliza multiples metodos de deteccion:

1. **AI Detection**: Modelo pre-entrenado de HuggingFace para detectar imagenes generadas por IA
2. **Deepfake Detection**: Modelo especializado para detectar rostros manipulados
3. **ELA (Error Level Analysis)**: Analiza inconsistencias en niveles de compresion para detectar edicion
4. **Temporal Analysis**: Detecta cambios abruptos entre frames (lip-sync, video splicing)

## Formatos Soportados

### Imagenes
- JPEG (.jpg, .jpeg)
- PNG (.png)
- WebP (.webp)
- GIF (.gif)
- BMP (.bmp)

### Videos
- MP4 (.mp4)
- WebM (.webm)
- AVI (.avi)
- MOV (.mov)
- MKV (.mkv)

## Arquitectura

```
ia-detector/
├── app/
│   ├── __init__.py          # Version del paquete
│   ├── main.py              # FastAPI app y endpoints
│   ├── detector.py          # Logica de deteccion IA (HuggingFace)
│   ├── deepfake_detector.py # Deteccion de deepfakes
│   ├── ela_analyzer.py      # Error Level Analysis
│   ├── temporal_analyzer.py # Analisis temporal (cambios abruptos)
│   ├── combined_analyzer.py # Analisis combinado de todos los metodos
│   ├── models.py            # Schemas Pydantic
│   ├── utils.py             # Utilidades para imagenes
│   └── video_processor.py   # Extraccion de keyframes (PyAV)
├── training/
│   └── fine_tuning_colab.ipynb  # Notebook para fine-tuning en Colab
├── tests/
│   ├── __init__.py
│   ├── test_api.py          # Tests de endpoints de imagen
│   └── test_video.py        # Tests de endpoints de video
├── requirements.txt
├── .env.example
└── README.md
```

## Stack Tecnologico

| Componente | Tecnologia |
|------------|------------|
| Framework | FastAPI |
| Modelo IA | HuggingFace Transformers (`umm-maybe/AI-image-detector`) |
| Modelo Deepfake | HuggingFace (`dima806/deepfake_vs_real_image_detection`) |
| Procesamiento Video | PyAV (FFmpeg bindings) |
| Procesamiento Imagen | Pillow, NumPy |
| HTTP Client | httpx |
| Server | uvicorn |

## Modelo de Deteccion

Utiliza el modelo pre-entrenado `umm-maybe/AI-image-detector` de HuggingFace, entrenado para detectar imagenes generadas por:

- DALL-E
- Midjourney
- Stable Diffusion
- Otras GANs y modelos de difusion

## Procesamiento de Video

El analisis de video funciona mediante:

1. **Extraccion de Keyframes**: Usa PyAV para extraer solo los I-frames (keyframes) del video, lo que es mucho mas eficiente que decodificar todos los frames.

2. **Analisis Secuencial**: Cada keyframe se analiza individualmente con el modelo de deteccion.

3. **Agregacion**: Los resultados se agregan para generar el veredicto final basado en el porcentaje de frames detectados como IA.

## Limites

- **Tamano maximo de video**: 100MB
- **Formatos soportados**: MP4, WebM, AVI, MOV, MKV
- **Timeout para URLs externas**: 60 segundos

## Tests

```bash
# Ejecutar todos los tests
pytest

# Ejecutar con coverage
pytest --cov=app

# Ejecutar tests especificos
pytest tests/test_video.py -v
```

## Variables de Entorno

Copia `.env.example` a `.env` y configura:

```env
HOST=0.0.0.0
PORT=8000
MODEL_NAME=umm-maybe/AI-image-detector
MAX_IMAGE_SIZE=4096
CACHE_DIR=.cache
```

## Licencia

MIT
