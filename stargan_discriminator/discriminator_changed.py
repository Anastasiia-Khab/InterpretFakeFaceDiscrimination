import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Predictor(nn.Module):
    
    def __init__(self):
        super(Predictor, self).__init__()
                
    def forward(self, x):
        
        x = x.view(x.size(0), -1)
        
        real_scores = x.mean(dim=1).view(-1, 1) / 2
        fake_scores = -real_scores
        
        x = torch.cat((fake_scores, real_scores), dim=1)
        
        return x
        

class Discriminator(nn.Module):

    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
        
        self.classifier = Predictor()
        
    def forward(self, x):
                
        x = self.features(x)
        x = self.classifier(x)
        
        return x