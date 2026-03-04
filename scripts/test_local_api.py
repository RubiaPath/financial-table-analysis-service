#!/usr/bin/env python3
"""
Local API test client for table analysis service
Tests /api/v1/analyze-page endpoint with image_base64 and pdf_text

Usage:
    python3 test_local_api.py <image_path> [<pdf_text_file>]
    python3 test_local_api.py /path/to/image.png /path/to/text.txt
"""

import requests
import base64
import json
import sys
from pathlib import Path
from typing import Optional

API_HOST = "http://127.0.0.1:8080"
ANALYZE_ENDPOINT = f"{API_HOST}/api/v1/analyze-page"
HEALTH_ENDPOINT = f"{API_HOST}/health"


def check_health() -> bool:
    """Check if service is healthy"""
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Health: {data['status']}")
            print(f"  - SAM3: {data['sam3_ready']}")
            print(f"  - Ollama: {data['ollama_ready']}")
            return data['sam3_ready'] and data['ollama_ready']
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Health check error: {e}")
        return False


def encode_image_to_base64(image_path: str) -> str:
    """Encode image file to base64"""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def load_pdf_text(text_file: Optional[str]) -> str:
    """Load PDF text from file or use sample text"""
    if text_file:
        with open(text_file, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        # Sample financial text
        return """
        BALANCE SHEET
        Assets
        Current Assets:
        Cash and equivalents: 1000000
        Accounts receivable: 500000
        Inventory: 750000
        
        Liabilities
        Current Liabilities:
        Accounts payable: 300000
        Short-term debt: 200000
        
        Equity
        Common stock: 1000000
        Retained earnings: 750000
        """


def analyze_page(image_base64: str, pdf_text: str) -> dict:
    """Send request to analyze-page endpoint"""
    payload = {
        "image_base64": image_base64,
        "pdf_text": pdf_text
    }
    
    try:
        response = requests.post(
            ANALYZE_ENDPOINT,
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"✗ API request failed: {response.status_code}")
            print(f"  Response: {response.text[:200]}")
            return None
    except requests.exceptions.Timeout:
        print(f"✗ Request timeout (60s)")
        return None
    except Exception as e:
        print(f"✗ Request error: {e}")
        return None


def print_results(result: dict):
    """Pretty print analysis results"""
    print("\n" + "="*60)
    print("ANALYSIS RESULTS")
    print("="*60)
    
    print(f"\nPage Type: {result.get('page_type', 'N/A')} (confidence: {result.get('confidence_page_type', 0):.2f})")
    print(f"Table Type: {result.get('table_type', 'N/A')} (confidence: {result.get('confidence_table_type', 0):.2f})")
    
    bboxes = result.get('bboxes', [])
    print(f"\nDetected Tables: {len(bboxes)}")
    for i, bbox in enumerate(bboxes):
        print(f"  [{i}] bbox: ({bbox['x1']:.0f}, {bbox['y1']:.0f}) - ({bbox['x2']:.0f}, {bbox['y2']:.0f}), confidence: {bbox['confidence']:.2f}")
    
    print(f"\nImage: {result.get('image_width')}x{result.get('image_height')}")
    
    metadata = result.get('metadata', {})
    print(f"\nMetadata:")
    for key, val in metadata.items():
        print(f"  {key}: {val}")
    
    print("="*60 + "\n")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <image_path> [<pdf_text_file>]")
        print(f"Example: {sys.argv[0]} test.png document.txt")
        sys.exit(1)
    
    image_path = sys.argv[1]
    pdf_text_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Validate image file
    if not Path(image_path).exists():
        print(f"✗ Image file not found: {image_path}")
        sys.exit(1)
    
    print("Financial Table Analysis - Local API Test")
    print("="*60)
    print()
    
    # Check health
    print("1. Checking service health...")
    if not check_health():
        print("\n✗ Service is not healthy. Make sure container is running:")
        print("  bash scripts/test-docker-local.sh")
        sys.exit(1)
    
    # Encode image
    print("\n2. Encoding image...")
    image_base64 = encode_image_to_base64(image_path)
    print(f"✓ Image encoded ({len(image_base64)} bytes)")
    
    # Load PDF text
    print("\n3. Loading PDF text...")
    pdf_text = load_pdf_text(pdf_text_file)
    print(f"✓ PDF text loaded ({len(pdf_text)} chars)")
    
    # Send request
    print("\n4. Sending analysis request...")
    result = analyze_page(image_base64, pdf_text)
    
    if result:
        print("✓ Analysis complete")
        print_results(result)
    else:
        print("\n✗ Analysis failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
