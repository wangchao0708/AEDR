import os
import torch
import argparse
from diffusers import (
    StableDiffusionPipeline, StableDiffusionXLPipeline,
    AutoPipelineForText2Image, FluxPipeline,
    StableDiffusion3Pipeline
)


# 读取并解析提示词文件
# Read and parse prompts from file
def parse_prompts(file_path):
    prompts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if ': ' in line:
                num_part, prompt = line.split(': ', 1)
                num = num_part.split()[0]
                prompts.append((num.zfill(3), prompt))
    return prompts


# 图像生成函数
# Image generation functio
def generate_images(prompts):
    for num_str, prompt in prompts:
        print(f"Processing prompt {num_str}: {prompt[:100]}...")
        try:
            # 根据模型调整参数
            # Set inference parameters based on model
            if args.model in ["SD3.5", "FLUX"]:
                num_inference_steps, guidance_scale = 35, 7.5
            else:
                num_inference_steps, guidance_scale = 50, 7.5

            output = pipeline(
                prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                output_type="pil"
            )
            image = output.images[0]
            filename = f"{num_str}.png"
            save_path = os.path.join(save_folder, filename)
            image.save(save_path)
            print(f"Saved: {save_path}")
        except Exception as e:
            print(f"Error processing {num_str}: {str(e)}")


# 设置命令行参数
# Set up command line arguments
parser = argparse.ArgumentParser(description='image generation script')
parser.add_argument('--model', type=str, required=True, help='model name to generate images')
parser.add_argument('--prompt_file', type=str, required=True, help='path to prompt file')
parser.add_argument('--save_dir', type=str, default=None, help='directory to save generated images')
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
device = "cuda" if torch.cuda.is_available() else "cpu"
config = model_config.get(args.model)
if not config:
    raise ValueError(f"Unsupported model: {args.model}")
print(f"Loading model: {args.model}")
pipeline = config["class"].from_pretrained(**config["params"])
pipeline = pipeline.to(device)


# 验证并创建保存目录
# Create save directory if it doesn't exist
save_folder = args.save_dir if args.save_dir else f"Imgs/{args.model}"
os.makedirs(save_folder, exist_ok=True)


# 执行生成
# Execute image generation
print(f"\nLoading prompts from {args.prompt_file}")
prompts = parse_prompts(args.prompt_file)
print(f"Loaded {len(prompts)} prompts")
generate_images(prompts)
print("Image generation complete. Saved at:", os.path.abspath(save_folder))


# python gen_img.py --model FLUX --prompt_file Prompt.txt
# python gen_img.py --model KD2.1 --prompt_file Prompt.txt
# python gen_img.py --model SD1.5 --prompt_file Prompt.txt
# python gen_img.py --model SD2.1 --prompt_file Prompt.txt
# python gen_img.py --model SD2base --prompt_file Prompt.txt
# python gen_img.py --model SD3.5 --prompt_file Prompt.txt
# python gen_img.py --model SDXL --prompt_file Prompt.txt
# python gen_img.py --model VQDM --prompt_file Prompt.txt