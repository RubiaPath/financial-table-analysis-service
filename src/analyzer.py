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
        pdf_text: str,
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
            
            # 3. Classify page type using PDF text
            page_type, page_confidence = self.ollama.classify_page_type(pdf_text)
            if page_type is None:
                page_type = "other"
                page_confidence = 0.3
            logger.info(f"Page type: {page_type} (confidence: {page_confidence})")
            
            # 4. Classify table type using PDF text
            table_type = "UNKNOWN"
            table_confidence = 0.0
            
            if tables:
                table_type, table_confidence = self.ollama.classify_table_type(pdf_text)
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
                    "sam3_prompt": settings.SAM3_TEXT_PROMPT,
                    "pdf_text_length": len(pdf_text)
                }
            )
        
        except Exception as e:
            logger.error(f"Page analysis failed: {e}", exc_info=True)
            return self._error_response(f"Analysis failed: {str(e)}")
    
    def _decode_base64_image(self, image_base64: str) -> Optional[Image.Image]:
        """Decode base64 image, convert to RGB"""
        try:
            # Remove data URI prefix if present
            if "," in image_base64:
                image_base64 = image_base64.split(",", 1)[1]
            
            image_data = base64.b64decode(image_base64)
            image = Image.open(BytesIO(image_data))
            
            # Convert to RGB (removes alpha channel, converts grayscale, etc)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
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

    def analyze_pdf(self, pdf_bytes: bytes) -> 'AnalyzePDFResponse':
        """Analyze all pages in PDF with complete pipeline: SAM3 + per-table Ollama classification"""
        try:
            import fitz  # pymupdf
            from src.models import PDFPageResult, AnalyzePDFResponse
            
            pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            results = []
            pages_with_tables = 0
            
            logger.info(f"Analyzing PDF with {len(pdf_doc)} pages (full pipeline: SAM3 + per-table Ollama)...")
            
            for page_num in range(len(pdf_doc)):
                try:
                    page = pdf_doc[page_num]
                    
                    # 1. Extract text from page
                    pdf_text = page.get_text()
                    if not pdf_text or not pdf_text.strip():
                        pdf_text = f"Page {page_num + 1} - No text extracted"
                    
                    # 2. Render page to image
                    pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
                    image_data = pix.tobytes("ppm")
                    
                    from io import BytesIO
                    image = Image.open(BytesIO(image_data))
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    # 3. Detect tables using SAM3
                    tables = self.sam3.detect_tables(image)
                    if tables:
                        pages_with_tables += 1
                    
                    # 4-5. Only classify if tables were detected
                    page_type = "other"
                    page_confidence = 0.0
                    bboxes = []
                    
                    if tables:
                        # Classify page type using Ollama
                        page_type, page_confidence = self.ollama.classify_page_type(pdf_text)
                        if page_type is None:
                            page_type = "other"
                            page_confidence = 0.3
                        
                        # Classify table type for each table
                        for i, t in enumerate(tables):
                            table_type, table_confidence = self.ollama.classify_table_type(pdf_text)
                            if table_type is None:
                                table_type = "UNKNOWN"
                                table_confidence = 0.0
                            
                            bbox = BBoxCoordinate(
                                x1=float(t["bbox"][0]),
                                y1=float(t["bbox"][1]),
                                x2=float(t["bbox"][2]),
                                y2=float(t["bbox"][3]),
                                confidence=t["confidence"],
                                table_type=table_type,
                                confidence_table_type=table_confidence
                            )
                            bboxes.append(bbox)
                            logger.info(f"  Table {i+1} on page {page_num + 1}: type={table_type} ({table_confidence:.2%})")
                    
                    page_result = PDFPageResult(
                        page_number=page_num + 1,
                        page_type=page_type,
                        tables=bboxes,
                        image_height=pix.height,
                        image_width=pix.width,
                        confidence_page_type=page_confidence,
                        pdf_text=pdf_text[:500]  # Store first 500 chars of extracted text
                    )
                    results.append(page_result)
                    
                    logger.info(f"Page {page_num + 1}: {len(tables)} tables, "
                               f"page_type={page_type} ({page_confidence:.2%})")
                
                except Exception as e:
                    logger.error(f"Failed to process page {page_num + 1}: {e}", exc_info=True)
                    # Continue with next page even if one fails
                    results.append(PDFPageResult(
                        page_number=page_num + 1,
                        page_type="other",
                        tables=[],
                        image_height=0,
                        image_width=0,
                        confidence_page_type=0.0,
                        pdf_text=f"Error processing page: {str(e)}"
                    ))
            
            pdf_doc.close()
            
            return AnalyzePDFResponse(
                total_pages=len(results),
                pages_with_tables=pages_with_tables,
                pages=results,
                metadata={
                    "num_pages": len(results),
                    "pages_with_tables": pages_with_tables,
                    "sam3_prompt": settings.SAM3_TEXT_PROMPT,
                    "ollama_model": settings.OLLAMA_MODEL,
                    "classification_method": "Per-table individual classification"
                }
            )
        
        except Exception as e:
            logger.error(f"PDF analysis failed: {e}", exc_info=True)
            raise


# Global instance
page_analyzer: Optional[PageAnalyzer] = None


def get_page_analyzer() -> PageAnalyzer:
    """Get or create page analyzer instance"""
    global page_analyzer
    if page_analyzer is None:
        page_analyzer = PageAnalyzer()
    return page_analyzer
