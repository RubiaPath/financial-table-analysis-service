"""SAM3 table detection - directly from notebook 03_sam3_table_detection.ipynb"""

import logging
import sys
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
from PIL import Image
import torch

from src.config import settings

logger = logging.getLogger(__name__)


def to_cpu_numpy(x) -> Optional[np.ndarray]:
    """Convert torch Tensor / list / numpy to numpy."""
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().to("cpu").numpy()
    return np.asarray(x)


def mask_to_xyxy(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Convert binary mask to bounding box [x1,y1,x2,y2]."""
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return (x1, y1, x2, y2)


def normalize_masks(masks):
    """Convert masks to (N, H, W) numpy uint8 array."""
    if masks is None:
        return np.zeros((0, 1, 1), dtype=np.uint8)
    
    arr = to_cpu_numpy(masks)
    
    # (H,W) -> (1,H,W)
    if arr.ndim == 2:
        arr = arr[None, ...]
    
    # (N,1,H,W) -> (N,H,W)
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0, :, :]
    
    # Convert to binary
    arr = (arr > 0).astype(np.uint8)
    return arr


class SAM3Detector:
    """SAM3 table detection - based on notebook 03 implementation"""
    
    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self.processor = None
        self._ready = False
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize SAM3 model - exact notebook logic"""
        try:
            # Find SAM3 directory
            sam3_dir = Path("/opt/program/sam3").resolve()
            
            if not sam3_dir.exists():
                logger.error(f"SAM3 directory not found: {sam3_dir}")
                self._ready = False
                return
            
            # Add to sys.path if not already there
            if str(sam3_dir) not in sys.path:
                sys.path.insert(0, str(sam3_dir))
            
            # Import SAM3
            from sam3 import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            
            # Build BPE path exactly as notebook does
            sam3_root = sam3_dir / "sam3"
            bpe_path = str(sam3_root / "assets" / "bpe_simple_vocab_16e6.txt.gz")
            
            if not Path(bpe_path).exists():
                logger.error(f"BPE vocab file not found: {bpe_path}")
                self._ready = False
                return
            
            logger.info(f"Loading SAM3 from {sam3_dir}")
            
            # Build model exactly as notebook does
            sam_model = build_sam3_image_model(bpe_path=bpe_path)
            self.processor = Sam3Processor(sam_model, confidence_threshold=self.confidence_threshold)
            
            # GPU optimizations from notebook
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            
            self._ready = True
            logger.info("SAM3 initialized successfully")
            
        except Exception as e:
            logger.error(f"SAM3 initialization failed: {e}", exc_info=True)
            self._ready = False
    
    def is_ready(self) -> bool:
        """Check if model is ready"""
        return self._ready
    
    def sam3_text_prompt_segment(self, image: Image.Image, prompt: str) -> Dict[str, Any]:
        """Segment objects using text prompt - exact notebook implementation.
        
        Returns:
            Dict with keys: masks, boxes_xyxy, scores, n, raw
        """
        if not hasattr(self.processor, "set_image"):
            raise AttributeError("Processor missing set_image method")
        if not hasattr(self.processor, "set_text_prompt"):
            raise AttributeError("Processor missing set_text_prompt method")
        
        state = self.processor.set_image(image)
        out = self.processor.set_text_prompt(state=state, prompt=prompt)
        
        masks = normalize_masks(out.get("masks"))
        boxes = out.get("boxes", None)
        scores = out.get("scores", None)
        
        boxes_xyxy: List[Tuple[int, int, int, int]] = []
        
        # Use boxes from processor if available
        if boxes is not None and len(boxes) > 0:
            for b in boxes:
                b = list(map(int, b))
                if len(b) >= 4:
                    boxes_xyxy.append((b[0], b[1], b[2], b[3]))
        
        # Otherwise compute from masks
        if len(boxes_xyxy) == 0 and masks.shape[0] > 0:
            for i in range(masks.shape[0]):
                xyxy = mask_to_xyxy(masks[i])
                if xyxy is not None:
                    boxes_xyxy.append(xyxy)
        
        return {
            "masks": masks,
            "boxes_xyxy": boxes_xyxy,
            "scores": scores,
            "n": len(boxes_xyxy),
            "raw": out,
        }
    
    def detect_tables(self, image: Image.Image, text_prompt: Optional[str] = None) -> List[dict]:
        """Detect tables in image.
        
        Returns:
            List of tables: [{"bbox": [x1,y1,x2,y2], "confidence": float}, ...]
        """
        if not self._ready:
            logger.error("SAM3 not ready")
            return []
        
        try:
            prompt = text_prompt or settings.SAM3_TEXT_PROMPT
            
            # Use notebook logic
            res = self.sam3_text_prompt_segment(image, prompt=prompt)
            n = int(res.get("n", 0))
            boxes_xyxy = res.get("boxes_xyxy", []) or []
            scores = res.get("scores", None)
            
            tables = []
            
            # Convert scores to list
            if scores is None:
                scores_list = [self.confidence_threshold] * len(boxes_xyxy)
            elif torch.is_tensor(scores):
                scores_list = scores.detach().to("cpu").tolist()
            else:
                scores_list = np.asarray(scores).tolist() if scores is not None else []
            
            # Build result
            for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
                conf = float(scores_list[i]) if i < len(scores_list) else self.confidence_threshold
                
                if conf >= self.confidence_threshold:
                    tables.append({
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": conf
                    })
            
            logger.info(f"Detected {len(tables)} tables from {n} candidate masks")
            return tables
        
        except Exception as e:
            logger.error(f"Detection failed: {e}", exc_info=True)
            return []


# Global instance
sam3_detector: Optional[SAM3Detector] = None


def get_sam3_detector() -> SAM3Detector:
    """Get or create SAM3 detector singleton"""
    global sam3_detector
    if sam3_detector is None:
        sam3_detector = SAM3Detector(
            confidence_threshold=settings.SAM3_CONFIDENCE_THRESHOLD
        )
    return sam3_detector


