"""FastAPI application"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn

from src.config import settings
from src.models import AnalyzePageRequest, AnalyzePageResponse, HealthCheck
from src.analyzer import get_page_analyzer
from src.ollama_client import OllamaClient
from src.sam3_detector import get_sam3_detector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """App lifecycle"""
    logger.info("Starting Table Analysis Service...")
    
    # Initialize components
    analyzer = get_page_analyzer()
    if not analyzer.is_ready():
        logger.warning("Service initialized in degraded state (not all components ready)")
    else:
        logger.info("All components ready")
    
    yield
    
    logger.info("Shutting down Table Analysis Service...")


# Create FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="Financial document table detection and classification service",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check"""
    try:
        analyzer = get_page_analyzer()
        ollama = OllamaClient()
        sam3 = get_sam3_detector()
        
        ollama_ready = ollama.check_health()
        sam3_ready = sam3.is_ready()
        models_dir = settings.MODEL_BASE_PATH.exists()
        
        status = "healthy" if (ollama_ready and sam3_ready) else "degraded"
        
        return HealthCheck(
            status=status,
            sam3_ready=sam3_ready,
            ollama_ready=ollama_ready,
            models_dir_exists=models_dir
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthCheck(
            status="unhealthy",
            sam3_ready=False,
            ollama_ready=False,
            models_dir_exists=False
        )


@app.post("/api/v1/analyze-page", response_model=AnalyzePageResponse)
async def analyze_page(request: AnalyzePageRequest):
    """Analyze financial page"""
    try:
        analyzer = get_page_analyzer()
        
        if not analyzer.is_ready():
            raise HTTPException(
                status_code=503,
                detail="Service not ready - missing required components"
            )
        
        # Analyze page
        response = analyzer.analyze_page(
            image_base64=request.image_base64,
            image_height=request.image_height,
            image_width=request.image_width
        )
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analyze page request failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@app.post("/api/v1/analyze-page-file", response_model=AnalyzePageResponse)
async def analyze_page_file(file: UploadFile = File(...)):
    """Analyze image file"""
    try:
        if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
            raise HTTPException(
                status_code=400,
                detail="Only JPEG and PNG images are supported"
            )
        
        # Read file and convert to base64
        content = await file.read()
        import base64
        image_base64 = base64.b64encode(content).decode('utf-8')
        
        # Use the main analyze endpoint
        request = AnalyzePageRequest(image_base64=image_base64)
        return await analyze_page(request)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analyze page file request failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"File analysis failed: {str(e)}"
        )


@app.get("/")
async def root():
    """Root endpoint with service info"""
    return {
        "service": settings.API_TITLE,
        "version": settings.API_VERSION,
        "endpoints": {
            "health": "/health",
            "analyze_page": "/api/v1/analyze-page",
            "analyze_page_file": "/api/v1/analyze-page-file",
            "docs": "/docs"
        },
        "models": {
            "ollama": settings.OLLAMA_MODEL,
            "sam3": "sam3.1_large_ft",
            "models_path": str(settings.MODEL_BASE_PATH)
        }
    }


def run():
    """Run the FastAPI server"""
    logger.info(f"Starting server on {settings.API_HOST}:{settings.API_PORT}")
    uvicorn.run(
        app,
        host=settings.API_HOST,
        port=settings.API_PORT,
        log_level="info"
    )


# When imported as a module (for uvicorn from serve script)
# app is already instantiated above
# When run directly, execute the run function
if __name__ == "__main__":
    run()
