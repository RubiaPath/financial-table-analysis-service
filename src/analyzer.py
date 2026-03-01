"""Page analyzer orchestrator"""

import logging
import base64
from io import BytesIO
from PIL import Image
from typing import Optional, List

from src.config import settings
from src.models import AnalyzePageResponse, BBoxCoordinate
from src.ollama_client import OllamaClient
from src.sam3_detector import get_sam3_detector

logger = logging.getLogger(__name__)


class PageAnalyzer:
    """Financial page analysis"""
    
    def __init__(self):
        self.ollama = OllamaClient()
        self.sam3 = get_sam3_detector()
        self._ready = False
        self._verify_ready()
    
    def _verify_ready(self):
        """Verify all components are ready"""
        ollama_ready = self.ollama.check_health()
        sam3_ready = self.sam3.is_ready()
        
        self._ready = ollama_ready and sam3_ready
        
        if not self._ready:
            status_str = f"Ollama: {ollama_ready}, SAM3: {sam3_ready}"
            logger.error(f"PageAnalyzer not fully initialized - {status_str}")
        else:
            logger.info("PageAnalyzer initialized successfully")
    
    def is_ready(self) -> bool:
        """Check if analyzer is ready"""
        return self._ready
    
    def analyze_page(
        self, 
        image_base64: str,
        image_height: Optional[int] = None,
        image_width: Optional[int] = None
    ) -> AnalyzePageResponse:
        """Analyze financial document page"""
        try:
            # 1. Decode image
            image = self._decode_base64_image(image_base64)
            if image is None:
                return self._error_response("Failed to decode image")
            
            # Get actual dimensions
            actual_width, actual_height = image.size
            if image_height is None:
                image_height = actual_height
            if image_width is None:
                image_width = actual_width
            
            logger.info(f"Analyzing page - Size: {image_width}x{image_height}")
            
            # 2. Detect tables using SAM3
            tables = self.sam3.detect_tables(image)
            logger.info(f"SAM3 detected {len(tables)} potential tables")
            
            # 3. Classify page type
            page_type, page_confidence = self.ollama.classify_page_type(image_base64)
            if page_type is None:
                page_type = "other"
                page_confidence = 0.3
            logger.info(f"Page type: {page_type} (confidence: {page_confidence})")
            
            # 4. Classify table type
            table_type = "UNKNOWN"
            table_confidence = 0.0
            
            if tables:
                # Use first table region for classification (or could do all tables)
                table_type, table_confidence = self.ollama.classify_table_type(image_base64)
                if table_type is None:
                    table_type = "UNKNOWN"
                    table_confidence = 0.0
                logger.info(f"Table type: {table_type} (confidence: {table_confidence})")
            
            # 5. Convert table detections to response format
            bboxes = [
                BBoxCoordinate(
                    x1=float(t["bbox"][0]),
                    y1=float(t["bbox"][1]),
                    x2=float(t["bbox"][2]),
                    y2=float(t["bbox"][3]),
                    confidence=t["confidence"]
                )
                for t in tables
            ]
            
            return AnalyzePageResponse(
                page_type=page_type,
                table_type=table_type,
                bboxes=bboxes,
                image_height=actual_height,
                image_width=actual_width,
                raw_image_shape=(actual_height, actual_width),
                confidence_page_type=page_confidence,
                confidence_table_type=table_confidence,
                metadata={
                    "num_tables_detected": len(tables),
                    "ollama_model": settings.OLLAMA_MODEL,
                    "sam3_prompt": settings.SAM3_TEXT_PROMPT
                }
            )
        
        except Exception as e:
            logger.error(f"Page analysis failed: {e}", exc_info=True)
            return self._error_response(f"Analysis failed: {str(e)}")
    
    def _decode_base64_image(self, image_base64: str) -> Optional[Image.Image]:
        """Decode base64 image"""
        try:
            # Remove data URI prefix if present
            if "," in image_base64:
                image_base64 = image_base64.split(",", 1)[1]
            
            image_data = base64.b64decode(image_base64)
            image = Image.open(BytesIO(image_data))
            return image
        except Exception as e:
            logger.error(f"Failed to decode base64 image: {e}")
            return None
    
    def _error_response(self, error_message: str) -> AnalyzePageResponse:
        """Create error response"""
        return AnalyzePageResponse(
            page_type="other",
            table_type="UNKNOWN",
            bboxes=[],
            image_height=0,
            image_width=0,
            raw_image_shape=(0, 0),
            confidence_page_type=0.0,
            confidence_table_type=0.0,
            metadata={"error": error_message}
        )


# Global instance
page_analyzer: Optional[PageAnalyzer] = None


def get_page_analyzer() -> PageAnalyzer:
    """Get or create page analyzer instance"""
    global page_analyzer
    if page_analyzer is None:
        page_analyzer = PageAnalyzer()
    return page_analyzer
