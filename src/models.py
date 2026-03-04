"""Request/response models"""

from typing import List, Optional
from pydantic import BaseModel, Field


class BBoxCoordinate(BaseModel):
    x1: float = Field(..., description="Top-left x (pixels)")
    y1: float = Field(..., description="Top-left y (pixels)")
    x2: float = Field(..., description="Bottom-right x (pixels)")
    y2: float = Field(..., description="Bottom-right y (pixels)")
    confidence: float = Field(..., ge=0, le=1, description="Confidence")


class AnalyzePageRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 image for SAM3 detection")
    pdf_text: str = Field(..., description="Raw text from PDF for LLM classification")
    image_height: Optional[int] = Field(None, description="Image height")
    image_width: Optional[int] = Field(None, description="Image width")


class AnalyzePageResponse(BaseModel):
    page_type: str = Field(..., description="Page type")
    table_type: str = Field(..., description="Table type")
    bboxes: List[BBoxCoordinate] = Field(default_factory=list, description="Bboxes")
    image_height: int = Field(..., description="Height")
    image_width: int = Field(..., description="Width")
    raw_image_shape: tuple = Field(..., description="Shape")
    confidence_page_type: float = Field(..., ge=0, le=1, description="Page confidence")
    confidence_table_type: float = Field(..., ge=0, le=1, description="Table confidence")
    metadata: Optional[dict] = Field(default_factory=dict, description="Metadata")


class HealthCheck(BaseModel):
    status: str = Field(..., description="Status")
    sam3_ready: bool = Field(..., description="SAM3 ready")
    ollama_ready: bool = Field(..., description="Ollama ready")
    models_dir_exists: bool = Field(..., description="Models dir exists")
