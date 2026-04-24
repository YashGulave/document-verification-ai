from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .config import Config


def create_sample(path: Path | None = None) -> Path:
    Config.ensure_directories()
    output = path or (Config.SAMPLE_DIR / "sample_tampered_document.png")
    width, height = 900, 1180
    image = Image.new("RGB", (width, height), "#fbfbf6")
    draw = ImageDraw.Draw(image)
    font_large = _font(34)
    font = _font(24)
    font_small = _font(19)

    draw.rectangle((45, 45, width - 45, height - 45), outline="#1b1b1b", width=3)
    draw.text((80, 85), "GOVERNMENT IDENTITY RECORD", fill="#202020", font=font_large)
    rows = [
        ("Name", "AARAV MEHTA"),
        ("Document No", "HX-4920-7718"),
        ("Date of Birth", "1997-04-14"),
        ("Address", "42 Market Street, Pune"),
        ("Issue Date", "2024-09-02"),
    ]
    y = 175
    for label, value in rows:
        draw.text((90, y), f"{label}:", fill="#333333", font=font)
        draw.text((290, y), value, fill="#111111", font=font)
        y += 68

    draw.rectangle((88, 555, 812, 780), outline="#888888", width=2)
    draw.text((110, 585), "Authorized Use Only", fill="#222222", font=font)
    draw.text((110, 642), "This synthetic sample contains edited regions for testing.", fill="#333333", font=font_small)
    draw.line((90, 890, 420, 890), fill="#111111", width=2)
    draw.text((90, 905), "Signature", fill="#333333", font=font_small)

    edited = Image.new("RGB", (255, 64), "#fffdfd")
    edit_draw = ImageDraw.Draw(edited)
    edit_draw.text((12, 14), "DOB: 1993-04-14", fill="#090909", font=font)
    noisy = np.array(edited).astype(np.int16)
    noise = np.random.default_rng(7).normal(0, 22, noisy.shape)
    noisy = np.clip(noisy + noise, 0, 255).astype(np.uint8)
    image.paste(Image.fromarray(noisy), (286, 300))

    blurred_region = image.crop((520, 850, 790, 940)).filter(ImageFilter.GaussianBlur(radius=3.2))
    image.paste(blurred_region, (520, 850))

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)

    cv_image = cv2.imread(str(output))
    cv2.imwrite(str(output), cv_image)
    return output


def _font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


if __name__ == "__main__":
    print(create_sample())
