import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import random

class LightTransform:
    def __init__(self, is_train=True):
        self.is_train = is_train
        self.base_ops = T.Compose([
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __call__(self, img):
        if self.is_train and random.random() < 0.8:
            brightness = random.uniform(0.2, 1.8)
            img = F.adjust_brightness(img, brightness)
        return self.base_ops(img)