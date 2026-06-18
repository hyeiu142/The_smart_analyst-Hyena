import asyncio
import sys
import os
from backend.app.core.ingestion.image_processor import ImageProcessor

async def test_image():
    # Use a dummy pdf from uploads or similar
    pdf_path = "/home/yennguyen/Hyena/FPT_2025_7.pdf"
    if not os.path.exists(pdf_path):
        # Let's find a pdf
        import glob
        pdfs = glob.glob("/home/yennguyen/Hyena/uploads/*.pdf")
        if not pdfs:
            print("No PDF found")
            return
        pdf_path = pdfs[0]

    print(f"Testing on {pdf_path}")
    processor = ImageProcessor()
    chunks = await processor.process(pdf_path, {"company": "FPT"})
    print(f"Result: {len(chunks)} chunks")

if __name__ == "__main__":
    asyncio.run(test_image())
