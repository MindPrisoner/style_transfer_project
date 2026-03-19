from PIL import Image
import torch
import torchvision.transforms as transforms


def load_image(image_path, max_size=512, device="cpu"):
    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize(max_size),
        transforms.ToTensor()
    ])

    image = transform(image).unsqueeze(0)
    return image.to(device)


def save_image(tensor, path):
    image = tensor.clone().detach().cpu().squeeze(0)
    image = transforms.ToPILImage()(image)
    image.save(path)
