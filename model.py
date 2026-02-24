"""
Lightweight segmentation model for PASCAL VOC 2012.
Uses LR-ASPP with MobileNetV3-Large backbone (pretrained on ImageNet).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import (
    lraspp_mobilenet_v3_large,
    LRASPP_MobileNet_V3_Large_Weights,
)

from utils import NUM_CLASSES


def get_model(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """
    Build and return an LR-ASPP MobileNetV3-Large segmentation model.

    The model is pretrained on COCO (which includes VOC classes),
    so we get a strong starting point and just fine-tune.
    """
    if pretrained:
        weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT
        model = lraspp_mobilenet_v3_large(weights=weights)
        # Pretrained model already has 21-class heads for VOC
        # Only replace if a different num_classes is requested
        if num_classes != 21:
            in_channels_low = model.classifier.low_classifier.in_channels
            in_channels_high = model.classifier.high_classifier.in_channels
            model.classifier.low_classifier = nn.Conv2d(
                in_channels_low, num_classes, kernel_size=1
            )
            model.classifier.high_classifier = nn.Conv2d(
                in_channels_high, num_classes, kernel_size=1
            )
    else:
        model = lraspp_mobilenet_v3_large(weights=None, num_classes=num_classes)

    return model


class SegmentationModel(nn.Module):
    """
    Wrapper that ensures output is resized to the input spatial dimensions.
    torchvision segmentation models return an OrderedDict with key 'out'.
    """

    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()
        self.model = get_model(num_classes, pretrained)

    def forward(self, x):
        """
        x: (B, 3, H, W)
        returns: (B, num_classes, H, W)  — logits at input resolution
        """
        h, w = x.shape[2], x.shape[3]
        out = self.model(x)["out"]  # may be lower resolution
        if out.shape[2] != h or out.shape[3] != w:
            out = F.interpolate(out, size=(h, w), mode="bilinear",
                                align_corners=False)
        return out


def count_flops(model, input_size=(1, 3, 300, 300), device="cpu"):
    """Count FLOPs for a single forward pass using thop."""
    try:
        from thop import profile
        inp = torch.randn(*input_size).to(device)
        model_dev = model.to(device)
        model_dev.eval()
        flops, params = profile(model_dev, inputs=(inp,), verbose=False)
        return flops, params
    except ImportError:
        print("Warning: 'thop' not installed. Cannot compute FLOPs.")
        return None, None


if __name__ == "__main__":
    # Quick sanity check with pretrained=False (no download)
    print("Testing with pretrained=False...")
    model = SegmentationModel(num_classes=NUM_CLASSES, pretrained=False)
    x = torch.randn(1, 3, 300, 300)
    y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    assert y.shape == (1, NUM_CLASSES, 300, 300), f"Unexpected shape: {y.shape}"
    print("✓ Model output shape is correct!")

    flops, params = count_flops(model)
    if flops is not None:
        print(f"FLOPs:  {flops / 1e9:.3f} GFLOPs")
        print(f"Params: {params / 1e6:.2f} M")
