from PIL import Image

img = Image.open("symtrack_logo.png").convert("RGBA")

img.save(
    "favicon.ico",
    format="ICO",
    sizes=[
        (16, 16),
        (32, 32),
        (48, 48),
        (64, 64),
        (128, 128),
        (256, 256),
    ],
)