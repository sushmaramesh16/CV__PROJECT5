from PIL import Image, ImageDraw, ImageFont
import os

os.makedirs('my_digits', exist_ok=True)

# try different fonts that look more handwritten
fonts_to_try = [
    '/System/Library/Fonts/Supplemental/Arial.ttf',
    '/System/Library/Fonts/Supplemental/CourierNew.ttf', 
    '/System/Library/Fonts/Helvetica.ttc',
]

font = None
for f in fonts_to_try:
    try:
        font = ImageFont.truetype(f, 140)
        print(f"Using font: {f}")
        break
    except:
        continue

for i in range(10):
    img = Image.new('L', (200, 200), color=255)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), str(i), font=font)
    x = (200 - (bbox[2] - bbox[0])) // 2
    y = (200 - (bbox[3] - bbox[1])) // 2
    draw.text((x, y), str(i), fill=0, font=font)
    img.save(f'my_digits/digit_{i}.png')

print('Done!')
