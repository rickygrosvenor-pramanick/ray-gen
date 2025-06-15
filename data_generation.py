"""

Use a base Stable Diffusion checkpoint plus a LoRA adapter to generate synthetic RSNA-style chest X-rays.

Usage:
  python data_generation.py \
    --base_model_dir models/monai-sd-cxr \
    --adapter_dir adapters/xray-lora \
    --output_dir data/synth \
    --labels normal pneumonia \
    --prompts "frontal chest X-ray of a healthy lung" "frontal chest X-ray showing pneumonia" \
    --num_per_class 1000 \
    --guidance_scale 7.5 \
    --num_steps 30
"""
import os
import argparse
import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic chest X-rays with LoRA-adapted diffusion"
    )
    parser.add_argument(
        "--base_model_dir", type=str, required=True,
        help="Local path to the base Stable Diffusion model"
    )
    parser.add_argument(
        "--adapter_dir", type=str, required=True,
        help="Directory where the LoRA adapter is saved"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Root directory to save generated images"
    )
    parser.add_argument(
        "--labels", nargs='+', required=True,
        help="List of class labels (must match prompts)"
    )
    parser.add_argument(
        "--prompts", nargs='+', required=True,
        help="List of text prompts (one per label)"
    )
    parser.add_argument(
        "--num_per_class", type=int, default=100,
        help="Number of images to generate per class"
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=7.5,
        help="Classifier-free guidance scale"
    )
    parser.add_argument(
        "--num_steps", type=int, default=50,
        help="Number of inference steps"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()

    # Load the base pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        args.base_model_dir,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    # Attach LoRA adapter
    pipe.unet = PeftModel.from_pretrained(pipe.unet, args.adapter_dir)
    pipe.to(device)
    # Memory-efficient attention
    try:
        pipe.unet.enable_xformers_memory_efficient_attention()
    except:
        pass

    # Create output directories
    for label in args.labels:
        os.makedirs(os.path.join(args.output_dir, label), exist_ok=True)

    # Generation loop
    for label, prompt in zip(args.labels, args.prompts):
        class_dir = os.path.join(args.output_dir, label)
        print(f"Generating {args.num_per_class} images for '{label}' with prompt: {prompt}")
        for i in range(args.num_per_class):
            image = pipe(
                prompt,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_steps,
            ).images[0]
            filename = f"{label}_{i:04d}.png"
            image.save(os.path.join(class_dir, filename))
    print("Generation complete.")

if __name__ == '__main__':
    main()
