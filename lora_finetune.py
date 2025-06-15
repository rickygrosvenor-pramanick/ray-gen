"""

Usage:
  python lora_finetune_hf.py \
    --repo_id linaqruf/monai-sd-cxr \
    --cache_dir models/monai-sd-cxr \
    --data_dir data/real/train \
    --prompt "frontal chest X-ray showing pneumonia" \
    --output_dir adapters/xray-lora \
    --epochs 3 --batch_size 4 --lr 1e-4 --r 4 --alpha 16 --dropout 0.05
"""

import os
import argparse
import torch
from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model
from huggingface_hub import snapshot_download
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def download_base_model(repo_id: str, cache_dir: str, token: str = None) -> str:
    """
    Download the model repository from Hugging Face Hub to a local cache.
    Returns the local path.
    """
    return snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=cache_dir,
        local_dir_use_symlinks=False,
        token=token,
    )

def train_lora(
    model_dir:  str,
    data_dir:   str,
    prompt:     str,
    output_dir: str,
    epochs:     int,
    batch_size: int,
    lr:         float,
    r:          int,
    alpha:      int,
    dropout:    float,
):
    device = get_device()
    # Load pipeline from local cache
    pipe = StableDiffusionPipeline.from_pretrained(
        model_dir,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    pipe.to(device)
    # Enable memory-efficient attention if available
    try:
        pipe.unet.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    # Attach LoRA
    peft_config = LoraConfig(
        task_type="UNET",
        inference_mode=False,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
    )
    pipe.unet = get_peft_model(pipe.unet, peft_config)

    # Prepare dataset
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    optimizer = optim.AdamW(pipe.unet.parameters(), lr=lr)
    mse = nn.MSELoss()

    # Training loop
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for imgs, _ in loader:
            imgs = imgs.to(device)
            # Encode images to latents
            latents = pipe.vae.encode(imgs * 2 - 1).latent_dist.sample()
            latents = latents * pipe.vae.config.scaling_factor
            # Sample random noise and timesteps
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                pipe.scheduler.num_train_timesteps,
                (latents.shape[0],),
                device=device,
            )
            noisy = pipe.scheduler.add_noise(latents, noise, timesteps)

            # Text embeddings
            inputs = pipe.tokenizer(
                [prompt] * imgs.size(0),
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(device)
            embeds = pipe.text_encoder(inputs.input_ids)[0]

            # Predict and compute loss
            pred = pipe.unet(noisy, timesteps, encoder_hidden_states=embeds).sample
            loss = mse(pred, noise)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg = total_loss / len(loader)
        print(f"Epoch {epoch}/{epochs} — avg loss: {avg:.4f}")

    # Save LoRA adapter only
    os.makedirs(output_dir, exist_ok=True)
    pipe.unet.save_pretrained(output_dir)
    print("LoRA adapter saved to", output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo_id", type=str, required=True,
        help="Hugging Face model repo (e.g. linaqruf/monai-sd-cxr)"
    )
    parser.add_argument(
        "--cache_dir", type=str, required=True,
        help="Local directory to store the downloaded model"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Folder of real images for fine-tuning (ImageFolder format)"
    )
    parser.add_argument(
        "--prompt", type=str, required=True,
        help="Text prompt describing your images"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Where to save the LoRA adapter"
    )
    parser.add_argument("--token", type=str, default=None,
                        help="HF_TOKEN if needed for private repos or rate limits")
    parser.add_argument("--epochs",    type=int,   default=3)
    parser.add_argument("--batch_size",type=int,   default=4)
    parser.add_argument("--lr",        type=float, default=1e-4)
    parser.add_argument("--r",         type=int,   default=4)
    parser.add_argument("--alpha",     type=int,   default=16)
    parser.add_argument("--dropout",   type=float, default=0.05)
    args = parser.parse_args()

    # Download base model locally
    model_dir = download_base_model(args.repo_id, args.cache_dir, args.token)
    # LoRA fine-tune
    train_lora(
        model_dir=model_dir,
        data_dir=args.data_dir,
        prompt=args.prompt,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        r=args.r,
        alpha=args.alpha,
        dropout=args.dropout,
    )

if __name__ == '__main__':
    main()
