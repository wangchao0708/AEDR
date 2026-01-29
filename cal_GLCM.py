import cv2
import numpy as np
import argparse
import os
import re
from skimage.feature import graycomatrix, graycoprops

# 设置参数解析器
# Set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', type=str, required=True, help='Path to images')
parser.add_argument('--output_file', type=str, required=True, help='Output text file')
args = parser.parse_args()


# 定义支持的图像扩展名
# Define supported image extensions
valid_extensions = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')

# 获取文件夹内所有符合条件的图像文件
# Get all valid image files from the directory
image_files = [f for f in os.listdir(args.image_dir) if f.endswith(valid_extensions)]
image_files.sort(key=lambda f: int(re.sub('\D', '', f)) if re.sub('\D', '', f) else f)

# 打开文件准备写入
# Open file for writing
with open(args.output_file, 'w') as f:
    total_files = len(image_files)
    print(f"Total images found: {total_files}")
    
    for index, filename in enumerate(image_files):
        img_path = os.path.join(args.image_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"[{index+1}/{total_files}] Failed to read: {filename}")
            f.write("NaN\n")
            continue

        if (index + 1) % 50 == 0:
            print(f"Processed {index + 1}/{total_files} images...")

        # 降低灰度等级
        # Reduce grayscale levels
        img_reduced = (img / 8).astype(np.uint8)

        # 计算GLCM（方向为0度，距离为1）
        # Calculate GLCM (angle=0, distance=1)
        glcm = graycomatrix(img_reduced,
                            distances=[1],
                            angles=[0],
                            levels=32,
                            symmetric=True,
                            normed=True)

        # 提取同质性特征
        # Extract homogeneity property
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        f.write(f"{homogeneity:.6f}\n")
        
    print(f"Processing complete! Results saved to: {args.output_file}")

# python cal_GLCM.py --image_dir Imgs/FLUX --output_file GLCM/FLUX.txt
# python cal_GLCM.py --image_dir Imgs/KD2.1 --output_file GLCM/KD2.1.txt
# python cal_GLCM.py --image_dir Imgs/Real --output_file GLCM/Real.txt
# python cal_GLCM.py --image_dir Imgs/SD1.5 --output_file GLCM/SD1.5.txt
# python cal_GLCM.py --image_dir Imgs/SD2.1 --output_file GLCM/SD2.1.txt
# python cal_GLCM.py --image_dir Imgs/SD2base --output_file GLCM/SD2base.txt
# python cal_GLCM.py --image_dir Imgs/SD3.5 --output_file GLCM/SD3.5.txt
# python cal_GLCM.py --image_dir Imgs/SDXL --output_file GLCM/SDXL.txt
# python cal_GLCM.py --image_dir Imgs/VQDM --output_file GLCM/VQDM.txt