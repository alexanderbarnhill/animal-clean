import io
from PIL import Image
import matplotlib.pyplot as plt


def save_image_tensor(image_tensor, output, transpose=True):
    if image_tensor.shape[0] == 1:
        image_tensor = image_tensor[0]
    if transpose:
        image_tensor = image_tensor.T

    image_tensor = image_tensor.detach()
    image_tensor = image_tensor.cpu()
    image_tensor = image_tensor.numpy()

    fig, ax = plt.subplots(dpi=60)
    ax.imshow(image_tensor, origin="lower", interpolation=None)
    plt.axis("off")
    plt.savefig(output, bbox_inches="tight")


def convert_tensor_to_PIL(image_tensor, transpose=True):
    if image_tensor.shape[0] == 1:
        image_tensor = image_tensor[0]
    if transpose:
        image_tensor = image_tensor.T

    image_tensor = image_tensor.detach()
    image_tensor = image_tensor.cpu()
    image_tensor = image_tensor.numpy()

    fig, ax = plt.subplots(dpi=60, figsize=(5, 10))
    ax.imshow(image_tensor, origin="lower", interpolation=None)
    plt.axis("off")
    plt.box(False)
    plt.margins(x=0, y=0)
    buffer = io.BytesIO()
    plt.savefig(buffer, bbox_inches="tight", pad_inches=0)
    buffer.seek(0)
    image = Image.open(buffer)
    image = resize_image(image)
    plt.close(fig)
    return image

def resize_image(image, base_width=512):
    wpercent = (base_width / float(image.size[0]))
    hsize = int((float(image.size[1]) * float(wpercent)))
    image = image.resize((base_width, hsize), Image.LANCZOS)
    return image