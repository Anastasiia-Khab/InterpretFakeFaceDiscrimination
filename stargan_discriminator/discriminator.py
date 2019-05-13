import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from stargan_discriminator.model import Discriminator, ModifiedDiscriminator

transform = Compose([
    Resize(256),
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def change_key(d, old, new):
    for _ in range(len(d)):
        k, v = d.popitem(False)
        d[new if old == k else k] = v

def get_discriminator():
    discrim = Discriminator(image_size=256).cuda()
    discrim.load_state_dict(torch.load('stargan_discriminator/checkpoint/weights.model'))
    return discrim

def get_modified_discriminator():
    discrim = ModifiedDiscriminator(image_size=256).cuda()
    discrim.load_state_dict(torch.load('stargan_discriminator/checkpoint/weights.model'))

    change_key(discrim.__dict__['_modules'], 'main', 'features')
    discrim._modules['features'].add_module('12', discrim._modules['conv1'])
    del discrim._modules['conv1']
    del discrim._modules['conv2']
    
    return discrim

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

def predict(x, threshold=0.58):
    return (sigmoid(x.view(x.shape[0], -1).mean(-1)).detach().cpu().numpy() > threshold).astype(int)

def discriminate(images, discrim):

    processed_images = []
    for image in images:
        processed_image = transform(image)
        processed_images.append(processed_image)
    processed_images = torch.stack(processed_images).cuda()

    preds, _ = discrim(processed_images)
    preds = predict(preds)

    return preds
