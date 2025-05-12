# main_blog_generator.py

"""
Gerador de imagens para posts de blog sobre o BICNet

Este script gera todas as imagens necessárias para os posts de blog
sobre o BICNet (Biologically-Inspired Complex Neural Network).

Uso:
    python main_blog_generator.py
"""

import os
import argparse
from export_blog_images import (
    export_standard_assembly_images,
    export_enhanced_assembly_images,
    export_demonstration_images
)

def main():
    parser = argparse.ArgumentParser(description='Generate blog images for BICNet')
    parser.add_argument('--output', default='./blog_images',
                        help='Output directory for blog images')
    parser.add_argument('--standard', action='store_true',
                        help='Generate only standard assembly images')
    parser.add_argument('--enhanced', action='store_true',
                        help='Generate only enhanced assembly images')
    parser.add_argument('--demos', action='store_true',
                        help='Generate only demonstration images')
    parser.add_argument('--all', action='store_true',
                        help='Generate all images (default)')
    args = parser.parse_args()
    
    # Se nenhuma opção específica for selecionada, gera todas as imagens
    if not (args.standard or args.enhanced or args.demos):
        args.all = True
    
    # Cria o diretório de saída se não existir
    os.makedirs(args.output, exist_ok=True)
    
    print(f"Generating blog images in {args.output}")
    
    if args.standard or args.all:
        export_standard_assembly_images()
    
    if args.enhanced or args.all:
        export_enhanced_assembly_images()
    
    if args.demos or args.all:
        export_demonstration_images()
    
    print(f"\nAll requested images have been generated in {args.output}")
    print("These images can now be used in your blog posts!")

if __name__ == "__main__":
    main()