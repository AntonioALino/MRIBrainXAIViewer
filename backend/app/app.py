from __future__ import annotations

import base64
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from brain_regions.brain_regions import get_region_full, list_all_regions, get_region_info
from gradcam.gradcam import GradCAM, apply_colormap, blend_heatmap
from model.model import CLASS_NAMES, BrainMRIClassifier, load_model, preprocess_image


## --- Constantes --- ##

WEIGHTS_PATH = Path(__file__).parent.parent / "brain_mri_weights.pth"

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024

ACCEPTED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}

WORKING_RESOLUTION = (224, 224)

API_VERSION = "1.0.0"


## --- Estado global do servidor --- ##

_model: BrainMRIClassifier | None = None
_gradcam: GradCAM | None = None
_startup_time: float = 0.0
_device: str = "cpu"


## --- Lifecycle: startup e shutdown --- ##

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _gradcam, _startup_time, _device

    print("\n" + "─" * 60)
    print("  MRI Brain XAI Viewer — Backend")
    print("─" * 60)

    if torch.cuda.is_available():
        _device = "cuda"
        print(f"  GPU detectada: {torch.cuda.get_device_name(0)}")
    else:
        _device = "cpu"
        print("  Rodando em CPU (sem GPU detectada)")

    if not WEIGHTS_PATH.exists():
        print(f"\n  ERRO: arquivo de pesos não encontrado:")
        print(f"  {WEIGHTS_PATH.resolve()}")
        print("\n  Para gerar os pesos:")
        print("  1. Baixe o dataset: kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri")
        print("  2. Execute: python train.py")
        print("  3. Reinicie o servidor\n")
        raise FileNotFoundError(
            f"brain_mri_weights.pth não encontrado em {WEIGHTS_PATH.parent}. "
            "Execute train.py para gerar o arquivo de pesos."
        )

    t0 = time.perf_counter()
    _model  = load_model(str(WEIGHTS_PATH))
    _model  = _model.to(_device)
    _gradcam = GradCAM(_model)
    _startup_time = time.perf_counter() - t0

    print(f"\n  Modelo carregado em {_startup_time:.2f}s")
    print(f"  Classes: {CLASS_NAMES}")
    print(f"  Arquivo: {WEIGHTS_PATH.name}")
    print(f"  Device:  {_device.upper()}")
    print(f"\n  Servidor pronto em http://localhost:8000")
    print(f"  Swagger UI:  http://localhost:8000/docs")
    print("─" * 60 + "\n")

    yield

    print("\nServidor encerrado.")


## --- Aplicação FastAPI --- ##

app = FastAPI(
    title="MRI Brain XAI Viewer — API",
    description=(
        "Backend para análise de imagens de MRI cerebral com Deep Learning "
        "e Explainable AI (Grad-CAM). Classifica tumores cerebrais em 4 classes "
        "e gera heatmaps que explicam as predições."
    ),
    version=API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


## --- Schemas Pydantic --- ##

class CoordsPayload(BaseModel):

    u: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Coordenada horizontal normalizada [0.0, 1.0]. "
                    "0.0 = borda esquerda, 1.0 = borda direita.",
        examples=[0.5],
    )
    v: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Coordenada vertical normalizada [0.0, 1.0]. "
                    "0.0 = topo (frontal), 1.0 = base (cerebelo).",
        examples=[0.15],
    )


class ClassificationResult(BaseModel):

    label: str = Field(..., description="Nome da classe predita (ex: 'Glioma')")
    predicted_class: int = Field(..., description="Índice numérico da classe [0–3]")
    confidence: float = Field(..., description="Confiança da predição [0.0–1.0]")
    probabilities: dict[str, float] = Field(
        ...,
        description="Probabilidade (softmax) para cada classe.",
    )


class AnalysisResult(BaseModel):

    original: str = Field(..., description="Imagem MRI original em base64 PNG (224×224)")
    heatmap: str  = Field(..., description="Heatmap Grad-CAM colorido em base64 PNG (224×224)")
    overlay: str  = Field(..., description="Blend original + heatmap em base64 PNG (224×224)")
    classification: ClassificationResult


class CompareResult(BaseModel):

    patient1: AnalysisResult
    patient2: AnalysisResult


class HealthResponse(BaseModel):

    status: str
    version: str
    model_loaded: bool
    model_file: str
    device: str
    classes: list[str]
    startup_time_seconds: float
    uptime_seconds: float


## --- Helpers internos --- ##

def _assert_model_loaded() -> tuple[BrainMRIClassifier, GradCAM]:
    if _model is None or _gradcam is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Modelo não está carregado. "
                "Verifique se brain_mri_weights.pth existe e reinicie o servidor."
            ),
        )
    return _model, _gradcam


def _validate_image_file(file: UploadFile) -> None:

    content_type = (file.content_type or "").lower()
    if content_type not in ACCEPTED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Tipo de arquivo não suportado: '{content_type}'. "
                f"Tipos aceitos: {', '.join(sorted(ACCEPTED_MIME_TYPES))}."
            ),
        )


async def _read_and_validate_file(file: UploadFile) -> bytes:
    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Arquivo '{file.filename}' está vazio.",
        )

    if len(contents) > MAX_FILE_SIZE_BYTES:
        size_mb = len(contents) / (1024 * 1024)
        limit_mb = MAX_FILE_SIZE_BYTES / (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"Arquivo muito grande: {size_mb:.1f} MB. "
                f"Limite: {limit_mb:.0f} MB."
            ),
        )

    return contents


def _decode_image(contents: bytes, filename: str = "imagem") -> np.ndarray:
    nparr = np.frombuffer(contents, dtype=np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Não foi possível decodificar '{filename}' como imagem. "
                "Verifique se o arquivo não está corrompido e é um formato "
                "suportado (JPEG, PNG, WebP, BMP)."
            ),
        )

    return img_bgr


def _to_base64_png(img_rgb: np.ndarray) -> str:
    if img_rgb.dtype != np.uint8:
        img_rgb = np.clip(img_rgb * 255, 0, 255).astype(np.uint8)

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    success, buffer = cv2.imencode(".png", img_bgr)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Falha ao codificar imagem para PNG.",
        )

    return base64.b64encode(buffer).decode("utf-8")


def _run_analysis_pipeline(
    img_bgr: np.ndarray,
    model: BrainMRIClassifier,
    gradcam: GradCAM,
) -> AnalysisResult:
    """
    Executa o pipeline completo de análise para uma imagem MRI.

    Pipeline:
      1. Redimensiona para 224×224 (resolução de trabalho do ResNet18)
      2. Converte BGR → RGB
      3. Pré-processa para tensor PyTorch (normalização ImageNet)
      4. Forward pass → logits → softmax → classe predita + probabilidades
      5. Grad-CAM → heatmap normalizado [0,1]
      6. Aplica colormap JET ao heatmap → RGB colorido
      7. Blend original + heatmap colorido → overlay
      8. Codifica as três imagens em base64 PNG
      9. Monta e retorna AnalysisResult

    Args:
        img_bgr : imagem MRI em BGR uint8 (qualquer resolução)
        model   : BrainMRIClassifier com pesos carregados
        gradcam : GradCAM configurado com o mesmo modelo

    Returns:
        AnalysisResult com imagens em base64 e resultado de classificação
    """
    # 1–2. Resize + RGB
    img_bgr_resized = cv2.resize(img_bgr, WORKING_RESOLUTION,
                                  interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img_bgr_resized, cv2.COLOR_BGR2RGB)

    # 3. Pré-processamento
    tensor = preprocess_image(img_rgb)
    tensor = tensor.to(_device)

    # 4. Classificação
    model.eval()
    with torch.enable_grad():
        logits = model(tensor)

    probs        = torch.softmax(logits, dim=1).squeeze()
    pred_class   = int(logits.argmax(dim=1).item())
    confidence   = float(probs[pred_class].item())
    probs_dict   = {
        name: round(float(p.item()), 4)
        for name, p in zip(CLASS_NAMES, probs)
    }

    # 5. Grad-CAM
    heatmap = gradcam.compute(
        input_tensor=tensor,
        target_class=pred_class,
        output_size=WORKING_RESOLUTION,
    )

    # 6–7. Visualização
    heatmap_rgb = apply_colormap(heatmap)
    overlay_rgb = blend_heatmap(
        original_rgb=img_rgb,
        heatmap=heatmap,
        alpha=0.5,
    )

    # 8. Base64
    original_b64 = _to_base64_png(img_rgb)
    heatmap_b64  = _to_base64_png(heatmap_rgb)
    overlay_b64  = _to_base64_png(overlay_rgb)

    # 9. Resultado
    return AnalysisResult(
        original=original_b64,
        heatmap=heatmap_b64,
        overlay=overlay_b64,
        classification=ClassificationResult(
            label=CLASS_NAMES[pred_class],
            predicted_class=pred_class,
            confidence=round(confidence, 4),
            probabilities=probs_dict,
        ),
    )


## --- Endpoints --- ##

@app.post(
    "/analyze",
    response_model=AnalysisResult,
    summary="Analisa uma imagem de MRI cerebral",
    description=(
        "Recebe uma imagem de MRI, executa a classificação com ResNet18 "
        "fine-tuned e gera o heatmap Grad-CAM. Retorna as três imagens "
        "(original, heatmap, overlay) em base64 PNG e o resultado da classificação."
    ),
    tags=["Análise de MRI"],
)
async def analyze_mri(
    file: Annotated[
        UploadFile,
        File(description="Imagem de MRI cerebral (JPEG, PNG, WebP ou BMP, máx. 10 MB)"),
    ],
) -> JSONResponse:

    model, gradcam_instance = _assert_model_loaded()
    _validate_image_file(file)
    contents = await _read_and_validate_file(file)
    img_bgr  = _decode_image(contents, file.filename or "upload")

    result = _run_analysis_pipeline(img_bgr, model, gradcam_instance)
    return JSONResponse(content=result.model_dump())


@app.post(
    "/compare",
    response_model=CompareResult,
    summary="Compara dois exames de MRI lado a lado",
    description=(
        "Recebe dois arquivos de MRI e processa cada um pelo pipeline completo "
        "(classificação + Grad-CAM). Retorna os resultados paralelos para "
        "visualização comparativa no modo split-screen do frontend."
    ),
    tags=["Análise de MRI"],
)
async def compare_mri(
    file1: Annotated[
        UploadFile,
        File(description="Primeiro exame MRI (Paciente 1)"),
    ],
    file2: Annotated[
        UploadFile,
        File(description="Segundo exame MRI (Paciente 2)"),
    ],
) -> JSONResponse:

    model, gradcam_instance = _assert_model_loaded()

    results = []
    for idx, f in enumerate((file1, file2), start=1):
        _validate_image_file(f)
        contents = await _read_and_validate_file(f)
        img_bgr  = _decode_image(contents, f.filename or f"paciente{idx}")
        results.append(_run_analysis_pipeline(img_bgr, model, gradcam_instance))

    response = CompareResult(patient1=results[0], patient2=results[1])
    return JSONResponse(content=response.model_dump())


@app.post(
    "/brain-region/coords",
    summary="Retorna região cerebral por coordenadas UV",
    description=(
        "Recebe coordenadas UV normalizadas [0,1] de um clique no plano 3D "
        "(Three.js raycasting) e retorna as informações anatômicas e clínicas "
        "da região cerebral correspondente no corte axial."
    ),
    tags=["Atlas Cerebral"],
)
async def region_by_coords(payload: CoordsPayload) -> JSONResponse:
    region_id, region_info = get_region_full(payload.u, payload.v)


    response_data = {k: v for k, v in region_info.items() if k != "coords"}
    response_data["region_id"] = region_id

    return JSONResponse(content=response_data)


@app.get(
    "/brain-region/{region_id}",
    summary="Retorna informações de uma região cerebral por ID",
    description=(
        "Busca direta por ID de região. Útil quando o frontend já conhece o "
        "region_id (ex: vindo de uma lista ou de uma chamada anterior). "
        "Retorna 'unknown' silenciosamente para IDs não encontrados "
        "(sem lançar 404) para evitar erros na UI."
    ),
    tags=["Atlas Cerebral"],
)
async def region_by_id(region_id: str) -> JSONResponse:
    region_info = get_region_info(region_id)
    response_data = {k: v for k, v in region_info.items() if k != "coords"}
    response_data["region_id"] = region_id
    return JSONResponse(content=response_data)


@app.get(
    "/brain_regions",
    summary="Lista todas as regiões cerebrais do atlas",
    description=(
        "Retorna a lista completa de regiões mapeadas no atlas, com ID, "
        "nome e cor. Útil para popular dropdowns, legendas e painéis de "
        "seleção na UI. Não inclui coordenadas UV nem textos clínicos."
    ),
    tags=["Atlas Cerebral"],
)
async def list_regions() -> JSONResponse:
    return JSONResponse(content=list_all_regions())


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check do servidor",
    description=(
        "Verifica o status do servidor, do modelo e do device em uso. "
        "Use este endpoint para confirmar que o backend está pronto antes "
        "de iniciar o frontend, ou para monitoramento em produção."
    ),
    tags=["Sistema"],
)
async def health() -> JSONResponse:
    uptime = time.time() - _startup_time if _startup_time else 0.0

    return JSONResponse(
        content=HealthResponse(
            status="ok" if _model is not None else "degraded",
            version=API_VERSION,
            model_loaded=_model is not None,
            model_file=WEIGHTS_PATH.name,
            device=_device.upper(),
            classes=CLASS_NAMES,
            startup_time_seconds=round(_startup_time, 3),
            uptime_seconds=round(uptime, 1),
        ).model_dump()
    )


## --- Handlers de erro globais --- ##

@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Erro interno do servidor.",
            "type":  type(exc).__name__,
            "detail": str(exc),
        },
    )


## --- Entry point --- ##

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["./"],
        log_level="info",
        access_log=True,
    )