#!/usr/bin/env python3
import argparse
import yaml
import sys
from src.preprocessing.pipeline import batch_preprocess

def main():
    parser = argparse.ArgumentParser(description="Historical OCR preprocessing pipeline")
    parser.add_argument("input_dir", help="Input image directory")
    parser.add_argument("output_dir", help="Output directory") 
    parser.add_argument("--config", default="config.yaml", help="Config YAML file")
    
    args = parser.parse_args()
    
    # Load config
    config = {
        'grayscale': True,
        'target_size': [1024, None],
        'denoise': True,
        'enhance_contrast': True,
        'binarize': False
    }
    
    try:
        with open(args.config, 'r') as f:
            user_config = yaml.safe_load(f)
            config.update(user_config)
        print("📋 Loaded config:", args.config)
    except FileNotFoundError:
        print("⚠️  No config file, using defaults")
    
    batch_preprocess(args.input_dir, args.output_dir, config)

if __name__ == "__main__":
    main()