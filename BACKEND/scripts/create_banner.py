# scripts/create_banner.py
"""Create a simple project banner for Kolrose Limited"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_banner():
    """Create a banner image for the project"""
    img = Image.new('RGB', (1200, 300), color='#1a5276')
    draw = ImageDraw.Draw(img)
    
    # Gradient background
    for i in range(300):
        r = int(26 + (46 - 26) * i / 300)
        g = int(82 + (134 - 82) * i / 300)
        b = int(118 + (193 - 118) * i / 300)
        draw.rectangle([(0, i), (1200, i+1)], fill=(r, g, b))
    
    # Title
    draw.text((250, 80), "🏢 Kolrose Limited", fill='white')
    draw.text((200, 150), "AI-Powered Policy Assistant", fill='#d4e6f1')
    draw.text((180, 210), "Suite 10, Bataiya Plaza, Area 2 Garki, Abuja, FCT, Nigeria", fill='#aed6f1')
    
    os.makedirs('docs/images', exist_ok=True)
    img.save('docs/images/project_banner.png')
    print("✅ Saved: docs/images/project_banner.png")

if __name__ == "__main__":
    create_banner()