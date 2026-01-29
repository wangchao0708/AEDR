import argparse

def compute_acc(fileA, fileB, threshold):

    with open(fileA, 'r', encoding='utf-8') as fA, open(fileB, 'r', encoding='utf-8') as fB:
        linesA = fA.readlines()
        linesB = fB.readlines()
    
    values = []
    for i in range(len(linesA)):
        partsA = linesA[i].strip().split()
        a = float(partsA[0])
        b = float(partsA[1])
        c = float(linesB[i].strip())
        val = a * c / b
        values.append(val)
    
    belonging = 0
    for i in values:
        if i < threshold:
            belonging += 1
    
    return belonging, len(linesA)-belonging


# 设置参数解析器
# Set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--Loss_ratio', type=str, required=True, help='Path to loss ratio')
parser.add_argument('--GLCM', type=str, required=True, help='Path to GLCM')
parser.add_argument('--threshold', type=float, required=True, help='Threshold of the target model')
args = parser.parse_args()

belonging, non_belonging = compute_acc(args.Loss_ratio, args.GLCM, args.threshold)
print(f"belonging, non_belonging =", belonging, non_belonging)


# python cal_acc.py --Loss_ratio Result/SD1.5_SD2.1_l2.txt --GLCM GLCM/SD2.1.txt --threshold 1.0364583773111364