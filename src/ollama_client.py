"""Ollama LLM client"""

import logging
import requests
from typing import Optional
from src.config import settings

logger = logging.getLogger(__name__)


class OllamaClient:
    """Ollama service client"""
    
    def __init__(self, host: str = settings.OLLAMA_HOST, model: str = settings.OLLAMA_MODEL):
        self.host = host
        self.model = model
        self.timeout = settings.OLLAMA_TIMEOUT
        self._ready = False
    
    def check_health(self) -> bool:
        """Check if Ollama service is available"""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            self._ready = response.status_code == 200
            return self._ready
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            self._ready = False
            return False
    
    def classify_page_type(self, pdf_text: str) -> tuple[Optional[str], float]:
        """Classify page type based on PDF text"""
        return self._classify_with_text(
            pdf_text,
            settings.OLLAMA_PAGE_TYPE_PROMPT,
            settings.PAGE_TYPES
        )
    
    def classify_table_type(self, pdf_text: str) -> tuple[Optional[str], float]:
        """Classify table type based on PDF text"""
        return self._classify_with_text(
            pdf_text,
            settings.OLLAMA_TABLE_TYPE_PROMPT,
            settings.TABLE_TYPES
        )
    
    def _classify_with_text(
        self, 
        text: str, 
        prompt: str, 
        valid_categories: list
    ) -> tuple[Optional[str], float]:
        """Text-based classification"""
        if not self._ready:
            logger.warning("Ollama service not ready, attempting connection...")
            if not self.check_health():
                return None, 0.0
        
        try:
            # Combine prompt with context text
            full_prompt = f"{prompt}\n\nContext:\n{text}"
            
            payload = {
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                "temperature": 0.0
            }
            
            response = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                logger.error(f"Ollama request failed: {response.status_code} - {response.text}")
                return None, 0.0
            
            result = response.json()
            response_text = result.get("response", "").strip().upper()
            
            # Try to match response to valid categories
            for category in valid_categories:
                if category.upper() in response_text:
                    confidence = 0.9 if response_text == category.upper() else 0.7
                    logger.info(f"Classified as: {category} (confidence: {confidence})")
                    return category, confidence
            
            logger.warning(f"Response '{response_text}' did not match any valid category")
            return valid_categories[0], 0.3
        
        except requests.exceptions.Timeout:
            logger.error(f"Ollama request timeout (>{self.timeout}s)")
            return None, 0.0
        except Exception as e:
            logger.error(f"Ollama classification failed: {e}")
            return None, 0.0
