"""Test media path resolution in document scanning."""
import asyncio
from pathlib import Path
from core.scanner import DocumentScanner

async def test_media_paths():
    scanner = DocumentScanner(Path('../data'))
    
    # Get the sample README with images
    doc = await scanner.get_document('sample-images/README.md')
    
    if doc:
        print("=" * 80)
        print("Document Path:", doc.info.path)
        print("Document Title:", doc.info.title)
        print("=" * 80)
        
        # Find all img tags in the HTML
        html = doc.parsed.content_html
        img_start = 0
        img_count = 0
        
        while True:
            img_start = html.find('<img', img_start)
            if img_start == -1:
                break
            
            img_end = html.find('>', img_start)
            if img_end == -1:
                break
            
            img_tag = html[img_start:img_end + 1]
            img_count += 1
            
            print(f"\nImage {img_count}:")
            print(img_tag)
            
            img_start = img_end + 1
        
        print("\n" + "=" * 80)
        print(f"Total images found: {img_count}")
        print("=" * 80)
    else:
        print("Document not found!")

if __name__ == "__main__":
    asyncio.run(test_media_paths())
