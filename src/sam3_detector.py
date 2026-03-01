"""SAM3 table detection"""

import logging
import torch
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
from PIL import Image
import base64
from io import BytesIO

from src.config import settings

logger = logging.getLogger(__name__)


class SAM3Detector:
    """SAM3 table detection"""
    
    def __init__(self):
        self.device = settings.SAM3_DEVICE
        self.model = None
        self.processor = None
        self._ready = False
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize SAM3 model from checkpoint"""
        try:
            checkpoint_dir = settings.SAM3_CHECKPOINT_DIR
            
            if not checkpoint_dir.exists():
                logger.error(f"SAM3 checkpoint directory not found: {checkpoint_dir}")
                self._ready = False
                return
            
            # List checkpoint files for debugging
            checkpoint_files = list(checkpoint_dir.glob("*.pt"))
            if not checkpoint_files:
                logger.warning(f"No .pt files found in {checkpoint_dir}")
            else:
                logger.info(f"Found checkpoint files: {[f.name for f in checkpoint_files]}")
            
            # Dynamically import SAM3 - this assumes the repo is in Python path
            # You may need to add the SAM3 repo to sys.path or install it
            try:
                from sam3 import SAM3, SAM3Processor
                
                # Find the model checkpoint - adjust pattern if needed
                model_checkpoint = list(checkpoint_dir.glob("sam3*.pt"))
                if not model_checkpoint:
                    logger.error(f"No SAM3 model checkpoint found in {checkpoint_dir}")
                    self._ready = False
                    return
                
                checkpoint_path = str(model_checkpoint[0])
                logger.info(f"Loading SAM3 from: {checkpoint_path}")
                
                self.model = SAM3.from_pretrained(checkpoint_path)
                self.processor = SAM3Processor.from_pretrained(checkpoint_path)
                
                # Move to device
                self.model.to(self.device)
                self.model.eval()
                
                self._ready = True
                logger.info("SAM3 model initialized successfully")
                
            except ImportError as e:
                logger.error(f"Failed to import SAM3: {e}")
                logger.info("Ensure SAM3 repository is installed and in Python path")
                self._ready = False
        
        except Exception as e:
            logger.error(f"SAM3 initialization failed: {e}")
            self._ready = False
    
    def is_ready(self) -> bool:
        """Check if model is ready for inference"""
        return self._ready
    
    def detect_tables(self, image: Image.Image, text_prompt: Optional[str] = None) -> List[dict]:
        """Detect tables in image"""
        if not self._ready:
            logger.error("SAM3 model not ready")
            return []
        
        try:
            prompt = text_prompt or settings.SAM3_TEXT_PROMPT
            
            # Prepare image for SAM3
            image_tensor = self._prepare_image(image)
            
            # Run SAM3 inference with text prompt
            with torch.no_grad():
                # SAM3 API may vary - adjust based on actual implementation
                # This is a template that may need modification
                if hasattr(self.processor, 'encode'):
                    encoded = self.processor.encode(image_tensor)
                    outputs = self.model.predict(
                        encoded,
                        text_prompt=prompt,
                        device=self.device
                    )
                else:
                    # Alternative: direct model call
                    outputs = self.model(
                        image_tensor,
                        text_prompt=prompt
                    )
                
                # Convert outputs to bbox format
                tables = self._process_outputs(outputs, image.size)
                
                logger.info(f"Detected {len(tables)} tables")
                return tables
        
        except Exception as e:
            logger.error(f"Table detection failed: {e}")
            return []
    
    def _prepare_image(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL image to tensor"""
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to tensor
        image_array = np.array(image)
        image_tensor = torch.from_numpy(image_array).float().to(self.device)
        
        # Normalize if needed - adjust based on SAM3 requirements
        if image_tensor.max() > 1.0:
            image_tensor = image_tensor / 255.0
        
        # Add batch dimension if needed
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor
    
    def _process_outputs(self, outputs, image_size: Tuple[int, int]) -> List[dict]:
        """Process outputs to bboxes"""
        tables = []
        
        try:
            # Extract masks and scores - adjust based on actual output format
            if hasattr(outputs, 'masks'):
                masks = outputs.masks
                scores = outputs.iou_predictions if hasattr(outputs, 'iou_predictions') else None
            else:
                # Fallback if outputs is a dict
                masks = outputs.get('masks', [])
                scores = outputs.get('scores', None)
            
            if masks is None or len(masks) == 0:
                return tables
            
            # Convert masks to bboxes
            for i, mask in enumerate(masks):
                confidence = float(scores[i]) if scores is not None else settings.SAM3_CONFIDENCE_THRESHOLD
                
                # Check confidence threshold
                if confidence < settings.SAM3_CONFIDENCE_THRESHOLD:
                    continue
                
                # Get bounding box from mask
                bbox = self._mask_to_bbox(mask, image_size)
                
                if bbox:
                    tables.append({
                        "bbox": bbox,
                        "confidence": confidence,
                        "mask": mask.cpu().numpy() if torch.is_tensor(mask) else mask
                    })
        
        except Exception as e:
            logger.error(f"Output processing failed: {e}")
        
        return tables
    
    def _mask_to_bbox(self, mask: torch.Tensor, image_size: Tuple[int, int]) -> Optional[List[float]]:
        """Convert mask to bbox"""
        try:
            if torch.is_tensor(mask):
                mask = mask.cpu().numpy()
            
            # Threshold if probability map
            if mask.dtype == np.float32:
                mask = (mask > 0.5).astype(np.uint8)
            
            # Find bounding box
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            
            if not (np.any(rows) and np.any(cols)):
                return None
            
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            
            # Ensure valid coordinates
            x_min, y_min = max(0, int(x_min)), max(0, int(y_min))
            x_max, y_max = min(image_size[0], int(x_max) + 1), min(image_size[1], int(y_max) + 1)
            
            if x_min >= x_max or y_min >= y_max:
                return None
            
            return [float(x_min), float(y_min), float(x_max), float(y_max)]
        
        except Exception as e:
            logger.error(f"Mask to bbox conversion failed: {e}")
            return None


# Global instance
sam3_detector: Optional[SAM3Detector] = None


def get_sam3_detector() -> SAM3Detector:
    """Get or create SAM3 detector instance"""
    global sam3_detector
    if sam3_detector is None:
        sam3_detector = SAM3Detector()
    return sam3_detector
