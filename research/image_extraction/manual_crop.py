from pathlib import Path
from PIL import Image

from sample_regions import REGIONS

BASE_DIR = Path(__file__).parent
PAGES_DIR = BASE_DIR / "pages"
OUTPUT_DIR = BASE_DIR / "outputs"


def crop_regions():
    OUTPUT_DIR.mkdir(exist_ok=True)

    for region in REGIONS:
        page_num = region["page"]
        box = region["box"]
        name = region["name"]

        page_path = PAGES_DIR / f"page_{page_num}.png"
        if not page_path.exists():
            print(f"missing page image: {page_path}")
            continue

        image = Image.open(page_path)
        cropped = image.crop(box)

        out_path = OUTPUT_DIR / f"{name}.png"
        cropped.save(out_path)

        print(f"saved {out_path} box={box} size={cropped.size}")


if __name__ == "__main__":
    crop_regions()