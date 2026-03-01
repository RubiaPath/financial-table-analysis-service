"""Configuration from YAML"""

from pathlib import Path
from typing import List
import yaml

class Settings:
    def __init__(self, config_path: str = None):
        if config_path is None:
            # Look for config.yaml in /opt/program or current directory
            paths = [
                Path("/opt/program/config.yaml"),
                Path("config.yaml"),
                Path(__file__).parent.parent / "config.yaml"
            ]
            for p in paths:
                if p.exists():
                    config_path = str(p)
                    break
            if config_path is None:
                raise FileNotFoundError("config.yaml not found in /opt/program, ., or project root")
        
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        
        api = cfg["api"]
        self.API_TITLE = api["title"]
        self.API_VERSION = api["version"]
        self.API_HOST = api["host"]
        self.API_PORT = api["port"]
        
        models = cfg["models"]
        self.MODEL_BASE_PATH = Path(models["base_path"])
        
        sam3 = models["sam3"]
        self.SAM3_CHECKPOINT_DIR = Path(sam3["checkpoint_dir"])
        self.SAM3_MODEL_NAME = sam3["model_name"]
        self.SAM3_DEVICE = sam3["device"]
        self.SAM3_CONFIDENCE_THRESHOLD = sam3["confidence_threshold"]
        self.SAM3_TEXT_PROMPT = sam3["text_prompt"]
        
        ollama = models["ollama"]
        self.OLLAMA_HOST = ollama["host"]
        self.OLLAMA_MODEL = ollama["model"]
        self.OLLAMA_TIMEOUT = ollama["timeout"]
        self.OLLAMA_MODELS_DIR = Path(models["base_path"]) / "ollama" / "models"
        
        classification = cfg["classification"]
        self.PAGE_TYPES = classification["page_types"]
        self.TABLE_TYPES = classification["table_types"]
        
        self.OLLAMA_PAGE_TYPE_PROMPT = "Classify: main, supplement, or other. Respond with ONLY the category."
        self.OLLAMA_TABLE_TYPE_PROMPT = "Classify table: BALANCE_SHEET, INCOME_STATEMENT, CASH_FLOW, EQUITY, NOTES, OTHER_FINANCIAL, NON_FINANCIAL, or UNKNOWN. Respond with ONLY the category."


settings = Settings()
