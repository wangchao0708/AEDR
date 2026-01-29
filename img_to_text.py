import argparse
import os
import torch
from clip_interrogator import Config, Interrogator, list_clip_models
from PIL import Image
from tqdm import tqdm


# 三种模式可自行切换 默认：classic
# Three modes are available
def inference(ci, image, mode):
    image = image.convert("RGB")
    if mode == "best":
        return ci.interrogate(image)
    elif mode == "classic":
        return ci.interrogate_classic(image)
    elif mode == "fast":
        return ci.interrogate_fast(image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--clip",
        default="ViT-L-14/openai",
        choices=["ViT-L-14/openai", "ViT-H-14/laion2b_s32b_b79k"],
        help="name of CLIP model to use")
    parser.add_argument("-d", "--device", default="auto", help="device to use (auto, cuda or cpu)")
    parser.add_argument("-f", "--folder", required=True, help="path to folder of images")
    parser.add_argument("-m", "--mode", default="best", help="best, classic, or fast")
    parser.add_argument("-o", "--output", required=True, help="output txt file")
    parser.add_argument("--lowvram", action="store_true", help="optimize settings for low VRAM")
    args = parser.parse_args()

    # 定义输出路径
    # Define output path
    txt_path = args.output
    if os.path.exists(txt_path):
        raise FileExistsError(f"The output file {txt_path} already exists!")

    # 验证模型是否可用
    # Validate clip model name
    models = list_clip_models()
    if args.clip not in models:
        print(f"Could not find CLIP model {args.clip}!")
        print(f"available models: {models}")
        exit(1)

    # 选择设备
    # Select device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            print("CUDA is not available, using CPU. Warning: this will be very slow!")
    else:
        device = torch.device(args.device)

    # 初始化 CLIP Interrogator
    # Initialize configuration
    config = Config(device=device, clip_model_name=args.clip, quiet=True)
    if args.lowvram:
        config.apply_low_vram_defaults()
    ci = Interrogator(config)

    # 处理文件夹中的图像
    # Process folder of images
    if not os.path.exists(args.folder):
        print(f"The folder {args.folder} does not exist!")
        exit(1)
    num_img = len([f for f in os.listdir(args.folder) if f.lower().endswith('.jpg')])
    files = [f"{i:03d}.jpg" for i in range(1, num_img+1)]

    # 批量迭代处理图像
    # Iterate through the files and generate prompts
    prompts = []
    for file in tqdm(files):
        image_path = os.path.join(args.folder, file)
        if not os.path.exists(image_path):
            print(f"Image {image_path} does not exist!")
            continue
        image = Image.open(image_path).convert("RGB")
        prompt = inference(ci, image, args.mode)
        prompts.append(prompt)

    # 将结果保存到文件
    # Write the generated prompts into the output text file
    if len(prompts):
        with open(txt_path, "w", encoding="utf-8") as f:
            for file, prompt in zip(files, prompts):
                f.write(f"{os.path.splitext(file)[0]}: {prompt}\n")
        print(f"\n\n\n\nGenerated {len(prompts)} prompts and saved to {txt_path}, enjoy!")


if __name__ == "__main__":
    main()


# python img_to_text.py -f Imgs/Real -o Prompt.txt -c ViT-H-14/laion2b_s32b_b79k -m classic