import os
import re
import torch
import torch.nn as nn
import argparse
from PIL import Image
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import retrieve_latents
import torchvision.transforms as T
from diffusers import (
    StableDiffusionPipeline, StableDiffusionXLPipeline,
    AutoPipelineForText2Image, FluxPipeline, StableDiffusion3Pipeline
)


def extract_numbers(filename):
    return tuple(map(int, re.findall(r'\d+', filename)))


# 设置命令行参数
# Set up command line arguments
parser = argparse.ArgumentParser(description='calculate attribution signal')
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--filePath', type=str, required=True)
parser.add_argument('--distance_metric', type=str, required=True, choices=['l1', 'l2', 'ssim', 'psnr', 'lpips'])
args = parser.parse_args()


# 模型配置
# Model configuration
model_config = {
    "FLUX": {
        "class": FluxPipeline,
        "params": {
            "pretrained_model_name_or_path": "black-forest-labs/FLUX.1-dev",
            "torch_dtype": torch.bfloat16
        }
    },
    "KD2.1": {
        "class": AutoPipelineForText2Image,
        "params": {
            "pretrained_model_or_path": "kandinsky-community/kandinsky-2-1",
            "torch_dtype": torch.float32
        }
    },
    "SD1.5": {
        "class": StableDiffusionPipeline,
        "params": {
            "pretrained_model_name_or_path": "runwayml/stable-diffusion-v1-5",
            "torch_dtype": torch.float32
        }
    },
    "SD2.1": {
        "class": StableDiffusionPipeline,
        "params": {
            "pretrained_model_name_or_path": "stabilityai/stable-diffusion-2-1",
            "torch_dtype": torch.float32
        }
    },
    "SD2base": {
        "class": StableDiffusionPipeline,
        "params": {
            "pretrained_model_name_or_path": "stabilityai/stable-diffusion-2-base",
            "torch_dtype": torch.float32
        }
    },
    "SD3.5": {
        "class": StableDiffusion3Pipeline,
        "params": {
            "pretrained_model_name_or_path": "stabilityai/stable-diffusion-3.5-large",
            "torch_dtype": torch.bfloat16
        }
    },
    "SDXL": {
        "class": StableDiffusionXLPipeline,
        "params": {
            "pretrained_model_name_or_path": "stabilityai/stable-diffusion-xl-base-1.0",
            "torch_dtype": torch.float32
        }
    },
}


# 加载模型
# Load the model
generator = torch.Generator().manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = model_config.get(args.model_name)
pipe = config["class"].from_pretrained(**config["params"])
pipe = pipe.to(device)
pipe.enable_model_cpu_offload()


# 获取模型自编码器
# Get the model's autoencoder
if hasattr(pipe, "vae"):
    ae = pipe.vae
    if hasattr(pipe, "upcast_vae"):
        pipe.upcast_vae()
elif hasattr(pipe, "movq"):
    ae = pipe.movq
elif hasattr(pipe, "vqvae"):
    ae = pipe.vqvae
ae.to(device)
ae = torch.compile(ae)
decode_dtype = ae.dtype


# 选择损失计算函数
# Select the loss calculation function
if args.distance_metric == 'l1':
    criterion = nn.L1Loss()
elif args.distance_metric == 'l2':
    criterion = nn.MSELoss()
elif args.distance_metric == 'ssim':
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    ssim = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
    criterion = lambda x, y: 1 - ssim((x + 1)/2, (y + 1)/2)
elif args.distance_metric == 'psnr':
    from torchmetrics.image import PeakSignalNoiseRatio
    psnr = PeakSignalNoiseRatio(data_range=2.0).to(device)
    criterion = lambda x, y: -psnr((x + 1)/2, (y + 1)/2)
elif args.distance_metric == 'lpips':
    import lpips
    lpips_model = lpips.LPIPS(net='alex').to(device)
    criterion = lambda x, y: lpips_model(x, y).mean()
else:
    raise ValueError(f"Unsupported metric: {args.distance_metric}")

# 定义输出文件
# Define the output file
image_files = sorted(
    [f for f in os.listdir(args.filePath) if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
    key=extract_numbers)
base_name = f"{args.model_name}_{os.path.basename(args.filePath)}_{args.distance_metric}"


metrics_file = open(f"Result/{base_name}.txt", 'a')
for filename in image_files:
    
    # 图像预处理
    # Image preprocessing
    img_path = os.path.join(args.filePath, filename)
    image = Image.open(img_path).convert('RGB')
    width, height = image.size
    if args.model_name == "KD2.1":
        image = image.resize((512, 512))
    elif width % 8 != 0 or height % 8 != 0:
        new_width = int(round(width / 8)) * 8
        new_height = int(round(height / 8)) * 8
        image = image.resize((new_width, new_height))
    transform = T.Compose([T.ToTensor(),])
    orig_image_tensor = transform(image).unsqueeze(0)
    current_input = (orig_image_tensor.to(device) * 2.0 - 1.0).to(ae.dtype)
    orig_image_tensor = current_input
    
    # 双重编解码
    # Double reconstruction
    metric_values, total_values = [], []
    for i in range(2):
        with torch.no_grad():
            latents = retrieve_latents(ae.encode(current_input), generator=generator)
            reconstruction = ae.decode(latents.to(decode_dtype), return_dict=False)[0]
            mse_loss = criterion(reconstruction, current_input)
        
        metric_val = mse_loss.item()
        total_val = criterion(reconstruction, orig_image_tensor).item()
        metric_values.append(f"{metric_val:.8f}")
        total_values.append(f"{total_val:.8f}")
        current_input = reconstruction.detach()

    metrics_file.write(f"{' '.join(metric_values)}\n")
    metrics_file.flush()

metrics_file.close()

# salloc -N 1 -c 5 --mem 30G -p gpu5 --gres gpu:1
# conda activate AEDR

# python cal_loss_ratio.py --model_name FLUX --filePath Imgs/FLUX --distance_metric l2
# python cal_loss_ratio.py --model_name KD2.1 --filePath Imgs/KD2.1 --distance_metric l2
# python cal_loss_ratio.py --model_name SD1.5 --filePath Imgs/SD1.5 --distance_metric l2
# python cal_loss_ratio.py --model_name SD2.1 --filePath Imgs/SD2.1 --distance_metric l2
# python cal_loss_ratio.py --model_name SD2base --filePath Imgs/SD2base --distance_metric l2
# python cal_loss_ratio.py --model_name SD3.5 --filePath Imgs/SD3.5 --distance_metric l2
# python cal_loss_ratio.py --model_name SDXL --filePath Imgs/SDXL --distance_metric l2

# srun python cal_loss_ratio.py --model_name SD1.5 --filePath Imgs/SD1.5 --distance_metric l1
# srun python cal_loss_ratio.py --model_name SD1.5 --filePath Imgs/SD1.5 --distance_metric l2
# srun python cal_loss_ratio.py --model_name SD1.5 --filePath Imgs/SD1.5 --distance_metric ssim
# srun python cal_loss_ratio.py --model_name SD1.5 --filePath Imgs/SD1.5 --distance_metric psnr
# srun python cal_loss_ratio.py --model_name SD1.5 --filePath Imgs/SD1.5 --distance_metric lpips

# python cal_loss_ratio.py --model_name SD1.5 --filePath Imgs/FLUX --distance_metric l2
# python cal_loss_ratio.py --model_name SD1.5 --filePath Imgs/KD2.1 --distance_metric l2
# python cal_loss_ratio.py --model_name SD1.5 --filePath Imgs/Real --distance_metric l2
# python cal_loss_ratio.py --model_name SD1.5 --filePath Imgs/SD1.5 --distance_metric l2
# python cal_loss_ratio.py --model_name SD1.5 --filePath Imgs/SD2.1 --distance_metric l2
# python cal_loss_ratio.py --model_name SD1.5 --filePath Imgs/SD2base --distance_metric l2
# python cal_loss_ratio.py --model_name SD1.5 --filePath Imgs/SD3.5 --distance_metric l2
# python cal_loss_ratio.py --model_name SD1.5 --filePath Imgs/SDXL --distance_metric l2