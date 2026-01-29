import numpy as np
from scipy.stats import gaussian_kde
import argparse

def compute_threshold_kde(fileA, fileB, alpha=0.05):

    with open(fileA, 'r', encoding='utf-8') as fA, open(fileB, 'r', encoding='utf-8') as fB:
        linesA = fA.readlines()
        linesB = fB.readlines()
    
    values = []
    for i in range(500):
        partsA = linesA[i].strip().split()
        a = float(partsA[0])
        b = float(partsA[1])
        c = float(linesB[i].strip())
        val = a * c / b
        values.append(val)
    values = np.array(values)
    
    # 利用KDE计算阈值
    # Using KDE to calculate the threshold
    sigma = np.std(values, ddof=1)
    kde = gaussian_kde(values)
    grid_min = values.min() - sigma
    grid_max = values.max() + sigma
    grid = np.linspace(grid_min, grid_max, 10000)
    density = kde(grid)
    cdf = np.cumsum(density)
    cdf = cdf / cdf[-1]
    N = 500
    prob = 1 - alpha
    threshold = np.interp(prob, cdf, grid)
    
    return threshold


# 设置参数解析器
# Set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--Loss_ratio', type=str, required=True, help='Path to loss ratio')
parser.add_argument('--GLCM', type=str, required=True, help='Path to GLCM')
parser.add_argument('--alpha', type=float, default=0.05, help='Significance level')
args = parser.parse_args()

threshold = compute_threshold_kde(args.Loss_ratio, args.GLCM, alpha=args.alpha)
print(f"threshold = {threshold}")


# python cal_threshold.py --Loss_ratio Result/FLUX_FLUX_l2.txt --GLCM GLCM/FLUX.txt
# python cal_threshold.py --Loss_ratio Result/KD2.1_KD2.1_l2.txt --GLCM GLCM/KD2.1.txt
# python cal_threshold.py --Loss_ratio Result/SD1.5_SD1.5_l2.txt --GLCM GLCM/SD1.5.txt
# python cal_threshold.py --Loss_ratio Result/SD2.1_SD2.1_l2.txt --GLCM GLCM/SD2.1.txt
# python cal_threshold.py --Loss_ratio Result/SD2base_SD2base_l2.txt --GLCM GLCM/SD2base.txt
# python cal_threshold.py --Loss_ratio Result/SD3.5_SD3.5_l2.txt --GLCM GLCM/SD3.5.txt
# python cal_threshold.py --Loss_ratio Result/SDXL_SDXL_l2.txt --GLCM GLCM/SDXL.txt
# python cal_threshold.py --Loss_ratio Result/VQDM_VQDM_l2.txt --GLCM GLCM/VQDM.txt