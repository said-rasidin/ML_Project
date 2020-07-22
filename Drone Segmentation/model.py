pip install segmentation-models-pytorch
import segmentation_models_pytorch as smp
from torchvision import models
import torch.nn as nn

def deeplabv3_resnet101(n_class=23, pretrained=True):
    if pretrained:
        model = models.segmentation.deeplabv3_resnet101(pretrained=True)
    else:
        model = models.segmentation.deeplabv3_resnet101(pretrained=False)

    model.classifier[4] = nn.Conv2d(in_channels=256, out_channels=n_class, kernel_size=1, stride=1)
    model.aux_classifier[4] = nn.Conv2d(in_channels=256, out_channels=n_class, kernel_size=1, stride=1)

    return model

def U-Net(n_class=23, encoder_name='efficientnet-b3', activation=None):
    model = smp.Unet(encoder_name, encoder_weights='imagenet', classes=n_class, activation=activation, 
                    encoder_depth=5, decoder_channels=[256, 128, 64, 32, 16])
    return model

                        