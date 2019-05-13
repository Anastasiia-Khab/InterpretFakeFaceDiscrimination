import torch
from torchvision.utils import make_grid

from PIL import Image

from stylegan_generator.model import StyledGenerator

def generate_fakes(num_images):
    generator = StyledGenerator(512).cuda()
    generator.load_state_dict(torch.load('stylegan_generator/checkpoint/weights.model')['generator'])

    mean_style = None
    step = 6
    shape = 4 * 2 ** step

    for i in range(10):
        style = generator.mean_style(torch.randn(1024, 512).cuda())

        if mean_style is None:
            mean_style = style
        else:
            mean_style += style

    mean_style /= 10

    with torch.no_grad():
        images = generator(
            torch.randn(num_images, 512).cuda(),
            step=step,
            alpha=1,
            mean_style=mean_style,
            style_weight=0.7,
        )

    images = postprocess_images(images)

    return images

def postprocess_images(images):
    processed_images = []
    for image in images:
        processed_image = make_grid(image, nrow=1, normalize=True, range=(-1, 1))
        processed_image = processed_image.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        processed_image = Image.fromarray(processed_image)
        processed_images.append(processed_image)

    return processed_images
