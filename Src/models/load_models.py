from torchvision.models.resnet import BasicBlock, model_urls, ResNet
import torch.utils.model_zoo as model_zoo

__all__ = ['resnet18']

def resnet18(pretrained=False, model_dir=None, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir=model_dir))
    return model