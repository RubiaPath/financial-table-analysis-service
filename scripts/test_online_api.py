#!/usr/bin/env python3
"""
Online API test client for deployed table analysis service
Tests SageMaker endpoint or any remote API endpoint

Usage:
    python3 test_online_api.py <endpoint_url> <image_path> [<pdf_text_file>]
    python3 test_online_api.py http://sagemaker-endpoint.com image.png text.txt
    python3 test_online_api.py $SAGEMAKER_ENDPOINT image.png  # uses default PDF text
"""

import requests
import base64
import json
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin


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
        CONSOLIDATED BALANCE SHEET
        As of December 31, 2024
        
        ASSETS
        Current Assets:
            Cash and cash equivalents: $50,000,000
            Marketable securities: $30,000,000
            Accounts receivable (net): $45,000,000
            Inventories: $35,000,000
            Prepaid expenses: $5,000,000
        Total Current Assets: $165,000,000
        
        Non-current Assets:
            Property, plant and equipment: $120,000,000
            Goodwill: $40,000,000
            Intangible assets: $25,000,000
            Long-term investments: $60,000,000
        Total Non-current Assets: $245,000,000
        
        TOTAL ASSETS: $410,000,000
        
        LIABILITIES
        Current Liabilities:
            Accounts payable: $20,000,000
            Accrued expenses: $15,000,000
            Current portion of long-term debt: $10,000,000
        Total Current Liabilities: $45,000,000
        
        Long-term Liabilities:
            Long-term debt: $100,000,000
            Deferred tax liabilities: $25,000,000
        Total Long-term Liabilities: $125,000,000
        
        TOTAL LIABILITIES: $170,000,000
        
        SHAREHOLDERS' EQUITY
        Common stock: $50,000,000
        Paid-in capital: $80,000,000
        Retained earnings: $110,000,000
        
        TOTAL SHAREHOLDERS' EQUITY: $240,000,000
        
        TOTAL LIABILITIES AND EQUITY: $410,000,000
        """


def check_health(endpoint_url: str) -> bool:
    """Check if remote service is healthy"""
    health_url = urljoin(endpoint_url, "/health")
    try:
        response = requests.get(health_url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Health: {data.get('status', 'healthy')}")
            return True
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Health check error: {e}")
        return False


def analyze_page_remote(endpoint_url: str, image_base64: str, pdf_text: str) -> dict:
    """Send request to remote analyze-page endpoint"""
    api_url = urljoin(endpoint_url, "/api/v1/analyze-page")
    
    payload = {
        "image_base64": image_base64,
        "pdf_text": pdf_text
    }
    
    try:
        print(f"Sending request to: {api_url}")
        start_time = time.time()
        
        response = requests.post(
            api_url,
            json=payload,
            timeout=120
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            print(f"✓ Response received in {elapsed:.1f}s")
            return response.json()
        else:
            print(f"✗ API request failed: {response.status_code}")
            error_text = response.text[:500]
            print(f"  Response: {error_text}")
            return None
    except requests.exceptions.Timeout:
        print(f"✗ Request timeout (120s)")
        return None
    except Exception as e:
        print(f"✗ Request error: {e}")
        return None


def print_results(result: dict):
    """Pretty print analysis results"""
    print("\n" + "="*60)
    print("REMOTE ANALYSIS RESULTS")
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
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <endpoint_url> <image_path> [<pdf_text_file>]")
        print()
        print("Examples:")
        print(f"  {sys.argv[0]} http://localhost:8080 test.png")
        print(f"  {sys.argv[0]} https://sagemaker.amazonaws.com/endpoint image.png document.txt")
        print()
        print("Environment variables:")
        print("  SAGEMAKER_ENDPOINT - URL of deployed endpoint")
        sys.exit(1)
    
    endpoint_url = sys.argv[1]
    image_path = sys.argv[2]
    pdf_text_file = sys.argv[3] if len(sys.argv) > 3 else None
    
    # Validate image file
    if not Path(image_path).exists():
        print(f"✗ Image file not found: {image_path}")
        sys.exit(1)
    
    print("Financial Table Analysis - Remote API Test")
    print("="*60)
    print(f"Endpoint: {endpoint_url}")
    print()
    
    # Check health
    print("1. Checking remote service health...")
    if not check_health(endpoint_url):
        print("\n✗ Remote service is not accessible")
        print(f"  Please verify endpoint URL: {endpoint_url}")
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
    result = analyze_page_remote(endpoint_url, image_base64, pdf_text)
    
    if result:
        print("✓ Analysis complete")
        print_results(result)
    else:
        print("\n✗ Analysis failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
