import matplotlib.pyplot as plt
import torch
from PIL import Image
import io
from utilities.viewing import convert_tensor_to_PIL

def test_convert_image_to_PIL():
    image_tensor = torch.randn((1, 128, 256))
    image = convert_tensor_to_PIL(image_tensor)
    return image