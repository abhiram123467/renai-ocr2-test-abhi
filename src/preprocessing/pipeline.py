from PIL import Image, ImageEnhance, ImageFilter
import os
import json
from typing import Dict

def preprocess_image(image_path: str, output_path: str, config: Dict):
    """Apply preprocessing pipeline to single image."""
    with Image.open(image_path) as img:
        # Grayscale
        if config.get
from PIL import Image, ImageEnhance, ImageFilter
import os
import json
from typing import Dict, List

def preprocess_image(image_path: str, output_path: str, config: Dict) -> Dict:
    """Apply preprocessing pipeline to single image."""
    with Image.open(image_path) as img:
        original_size = img.size
        
        # 1. Grayscale
        if config.get('grayscale', True):
            img = img.convert('L')
        
        # 2. Resize (preserve aspect ratio)
        if config.get('target_size'):
            target_width = config['target_size'][0]
            img.thumbnail((target_width, 9999), Image.Resampling.LANCZOS)
        
        # 3. Denoising (median filter)
        if config.get('denoise', True):
            img = img.filter(ImageFilter.MedianFilter(size=3))
        
        # 4. Contrast enhancement
        if config.get('enhance_contrast', True):
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.2)
        
        # 5. Binarization (optional)
        if config.get('binarize', False):
            img = img.point(lambda p: 255 if p > 128 else 0)
        
        img.save(output_path, 'JPEG', quality=95)
    
    return {
        'input': image_path,
        'output': output_path,
        'original_size': original_size,
        'final_size': Image.open(output_path).size,
        'config_used': config
    }

def batch_preprocess(input_dir: str, output_dir: str, config: Dict = None) -> List[Dict]:
    """Process all images in input directory."""
    if config is None:
        config = {}
    
    os.makedirs(output_dir, exist_ok=True)
    results = []
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.bmp')):
            input_path = os.path.join(input_dir, filename)
            output_filename = f"proc_{filename}"
            output_path = os.path.join(output_dir, output_filename)
            
            try:
                result = preprocess_image(input_path, output_path, config)
                results.append(result)
                print(f"✅ Processed: {filename}")
            except Exception as e:
                print(f"❌ Failed {filename}: {e}")
    
    # Save metadata
    metadata_path = os.path.join(output_dir, 'preprocessing_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"🎉 Processed {len(results)} images")
    print(f"📄 Metadata saved: {metadata_path}")
    return results