from PIL import Image
import torch
import torchvision.transforms as transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_image(image_path, max_size=512, device="cpu"):
    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize(max_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    image = transform(image).unsqueeze(0)
    return image.to(device)


def denormalize(tensor):
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1).to(tensor.device)
    return tensor * std + mean


def save_image(tensor, path):
    image = tensor.clone().detach()
    image = denormalize(image)
    image = image.clamp(0, 1)
    image = image.cpu().squeeze(0)
    image = transforms.ToPILImage()(image)
    image.save(path)