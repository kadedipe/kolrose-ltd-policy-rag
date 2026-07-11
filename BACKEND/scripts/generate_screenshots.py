# scripts/generate_screenshots.py
"""
Generate placeholder screenshots for Kolrose RAG project documentation.
Run this to create images showing the app interface.
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_app_screenshot():
    """Create a simple app screenshot placeholder"""
    img = Image.new('RGB', (1200, 800), color='#ffffff')
    draw = ImageDraw.Draw(img)
    
    # Header
    draw.rectangle([(0, 0), (1200, 80)], fill='#1a5276')
    draw.text((400, 20), "🏢 Kolrose Limited - Policy Assistant", fill='white')
    
    # Sidebar
    draw.rectangle([(0, 80), (300, 800)], fill='#f8f9fa', outline='#dee2e6')
    draw.text((20, 100), "⚙️ Settings", fill='#1a5276')
    draw.text((20, 150), "OpenRouter API Key: ********", fill='#666')
    draw.text((20, 200), "📊 Retrieval", fill='#333')
    draw.text((20, 250), "Initial candidates: 20", fill='#666')
    draw.text((20, 280), "Final results: 5", fill='#666')
    draw.text((20, 320), "✅ Cross-encoder re-ranking", fill='#28a745')
    
    # Main content
    draw.text((350, 120), "💬 Ask a Policy Question", fill='#1a5276')
    draw.rectangle([(350, 160), (1150, 300)], fill='#f0f8ff', outline='#1a5276')
    draw.text((370, 200), "What is the annual leave policy for new employees?", fill='#333')
    
    draw.text((350, 340), "📋 Answer", fill='#1a5276')
    draw.rectangle([(350, 370), (1150, 550)], fill='#e8f5e9', outline='#2e7d32')
    draw.text((370, 400), "According to the Leave and Time-Off Policy [KOL-HR-002,", fill='#333')
    draw.text((370, 430), "Section 1.1], employees with 0-2 years of service receive", fill='#333')
    draw.text((370, 460), "15 working days of annual leave per year.", fill='#333')
    
    # Citations
    draw.text((370, 500), "📝 Citations: [KOL-HR-002, Section 1.1]", fill='#1a5276')
    
    # Metrics
    draw.rectangle([(370, 570), (520, 610)], fill='#e3f2fd')
    draw.text((380, 580), "⏱️ 1,250ms", fill='#1a5276')
    draw.rectangle([(540, 570), (690, 610)], fill='#e3f2fd')
    draw.text((550, 580), "📚 2 sources", fill='#1a5276')
    draw.rectangle([(710, 570), (860, 610)], fill='#e3f2fd')
    draw.text((720, 580), "📝 3 citations", fill='#1a5276')
    draw.rectangle([(880, 570), (1030, 610)], fill='#e3f2fd')
    draw.text((890, 580), "💰 FREE", fill='#1a5276')
    
    # Footer
    draw.rectangle([(0, 760), (1200, 800)], fill='#f8f9fa')
    draw.text((400, 770), "Suite 10, Bataiya Plaza, Area 2 Garki, Abuja, FCT, Nigeria", fill='#999')
    
    os.makedirs('docs/images', exist_ok=True)
    img.save('docs/images/app_screenshot.png')
    print("✅ Saved: docs/images/app_screenshot.png")

if __name__ == "__main__":
    create_app_screenshot()