import matplotlib.pyplot as plt
import torch
from PIL import Image
import io

def test_convert_image_to_PIL():
    image_tensor = torch.randn((1, 128, 256))
    if image_tensor.shape[0] == 1:
        image_tensor = image_tensor[0]

    image_tensor = image_tensor.T

    image_tensor = image_tensor.detach()
    image_tensor = image_tensor.cpu()
    image_tensor = image_tensor.numpy()

    fig, ax = plt.subplots(dpi=60)
    ax.imshow(image_tensor, origin="lower", interpolation=None)

    buffer = io.BytesIO()
    plt.savefig(buffer)
    buffer.seek(0)
    image = Image.open(buffer)
    plt.close(fig)
    return image