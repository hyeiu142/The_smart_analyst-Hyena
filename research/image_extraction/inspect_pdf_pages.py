from pathlib import Path
import fitz

PDF_PATH = Path("/home/yennguyen/Hyena/Docs/FPT_2025_7.pdf")
OUTPUT_DIR = Path(__file__).parent / "pages"

ZOOM = 2.0


def render_pdf_pages():
    OUTPUT_DIR.mkdir(exist_ok=True)

    doc = fitz.open(PDF_PATH)

    for page_index in range(len(doc)):
        page = doc[page_index]
        matrix = fitz.Matrix(ZOOM, ZOOM)
        pix = page.get_pixmap(matrix=matrix)

        out_path = OUTPUT_DIR / f"page_{page_index + 1}.png"
        pix.save(out_path)

        print(f"saved {out_path} size={pix.width}x{pix.height}")


if __name__ == "__main__":
    render_pdf_pages()