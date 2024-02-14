import os
from torchvision import models

def resnet18(pretrained=True, **kwargs):
    model = models.resnet18(weights='DEFAULT')
    return model


def resnet34(pretrained=True, **kwargs):
    model = models.resnet34(weights='DEFAULT')
    return model


def resnet50(pretrained=True, **kwargs):
    model = models.resnet50(weights='DEFAULT')
    return model


def resnet101(pretrained=True, **kwargs):
    model = models.resnet101(weights='DEFAULT')
    return model


def resnet152(pretrained=True, **kwargs):
    model = models.resnet152(weights='DEFAULT')
    return model
