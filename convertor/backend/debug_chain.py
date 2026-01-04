import requests
import re
import urllib.parse
import sys

BASE_URL = "http://localhost:8000"

def test_chain():
    # 1. Fetch document
    doc_path = "sample-images/README.md"
    encoded_path = urllib.parse.quote(doc_path)
    url = f"{BASE_URL}/api/documents/{encoded_path}"
    
    print(f"Fetching document: {url}")
    try:
        res = requests.get(url)
        if res.status_code != 200:
            print(f"FAILED to fetch document: {res.status_code}")
            return
    except Exception as e:
        print(f"FAILED to connect: {e}")
        return

    data = res.json()
    html = data['content_html']
    
    print("\nExtracting images from HTML...")
    # Find img src
    matches = re.findall(r'src=["\']([^"\']+)["\']', html)
    
    if not matches:
        print("No images found in HTML!")
        return

    for src in matches:
        print(f"\nFound image src: {src}")
        
        # Handle relative URLs (if any remain, though they shouldn't)
        if not src.startswith(('http', '/')):
            print("  WARN: Relative URL found, frontend might fail to resolve it")
            full_img_url = f"{BASE_URL}/api/media/{urllib.parse.quote(src)}" # Guess
        else:
            full_img_url = f"{BASE_URL}{src}"
            
        print(f"  Fetching: {full_img_url}")
        img_res = requests.get(full_img_url)
        
        if img_res.status_code == 200:
            print(f"  SUCCESS: Image fetched ({len(img_res.content)} bytes)")
            print(f"  Content-Type: {img_res.headers.get('content-type')}")
        else:
            print(f"  FAILED: {img_res.status_code}")

if __name__ == "__main__":
    test_chain()
