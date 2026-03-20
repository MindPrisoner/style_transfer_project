import os
import torch
import torch.optim as optim

from models.style_transfer import VGGFeatures, gram_matrix
from utils.image_utils import load_image, save_image


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    content_path = "assets/content/content.jpg"
    style_path = "assets/style/style.jpg"
    output_path = "outputs/stylized_version2.jpg"

    os.makedirs("outputs", exist_ok=True)

    content_image = load_image(content_path, device=device)
    style_image = load_image(style_path, device=device)

    generated = content_image.clone().requires_grad_(True).to(device)

    model = VGGFeatures().to(device)

    with torch.no_grad():
        content_features = model(content_image)
        style_features = model(style_image)
        style_grams = {
            layer: gram_matrix(style_features[layer]).detach()
            for layer in style_features
        }

    optimizer = optim.Adam([generated], lr=0.02)

    content_weight = 1e2
    style_weight = 1e7
    tv_weight = 0.0
    for step in range(800):
        gen_features = model(generated)


        content_loss = torch.mean(
            (gen_features["conv4_2"] - content_features["conv4_2"]) ** 2
        )

        style_loss = 0.0
        for layer in style_grams:
            gen_gram = gram_matrix(gen_features[layer])
            style_gram = style_grams[layer]
            style_loss += torch.mean((gen_gram - style_gram) ** 2)

        # tv_loss = total_variation_loss(generated)
        tv_loss = torch.tensor(0.0, device=device)
        # total_loss = content_weight * content_loss + style_weight * style_loss
        total_loss = (
                content_weight * content_loss
                + style_weight * style_loss
                + tv_weight * tv_loss
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # with torch.no_grad():
        #     generated.clamp_(0, 1)
        with torch.no_grad():
            generated.clamp_(-3, 3)

        if step % 100 == 0:
            # print(
            #     f"Step [{step}/300], "
            #     f"Content Loss: {content_loss.item():.6f}, "
            #     f"Style Loss: {style_loss.item():.6f}, "
            #     f"Total Loss: {total_loss.item():.6f}"
            # )
            print(
                f"Step [{step}/800], "
                f"Content Loss: {content_loss.item():.6f}, "
                f"Style Loss: {style_loss.item():.6f}, "
                f"TV Loss: {tv_loss.item():.6f}, "
                f"Total Loss: {total_loss.item():.6f}"
            )
            save_image(generated, f"outputs/step_{step}.jpg")

    save_image(generated, output_path)
    print(f"Stylized image saved to {output_path}")

def total_variation_loss(image):
    loss = torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :])) + \
           torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]))
    return loss


if __name__ == "__main__":
    main()
