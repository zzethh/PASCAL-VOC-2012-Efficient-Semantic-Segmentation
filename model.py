import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.models._utils import IntermediateLayerGetter
from utils import NUM_CLASSES

class MicroMultiClassModel(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT).features
        self.backbone = IntermediateLayerGetter(backbone, return_layers={"1": "low", "12": "high"})
        self.high_conv = nn.Conv2d(in_channels=576, out_channels=21, kernel_size=1)
        self.low_conv = nn.Conv2d(in_channels=16, out_channels=21, kernel_size=1)

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        high_logits = self.high_conv(features['high'])
        low_logits = self.low_conv(features['low'])
        high_logits_up = F.interpolate(high_logits, size=low_logits.shape[-2:], mode='bilinear', align_corners=False)
        out = F.interpolate(high_logits_up + low_logits, size=input_shape, mode='bilinear', align_corners=False)
        return {"out": out}

class SegmentationModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True, model_type="micro_multiclass"):
        super().__init__()
        if model_type == "micro_multiclass": self.model = MicroMultiClassModel()
        self.internal_size = 192

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        x_in = F.interpolate(x, size=(self.internal_size, self.internal_size), mode="bilinear", align_corners=False) if (h != self.internal_size or w != self.internal_size) else x
        out = self.model(x_in)["out"]
        return F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False) if (out.shape[2] != h or out.shape[3] != w) else out

def count_flops(model, input_size=(1, 3, 300, 300), device="cpu"):
    try:
        from thop import profile
        inp = torch.randn(*input_size).to(device)
        model_dev = model.to(device).eval()
        flops, params = profile(model_dev, inputs=(inp,), verbose=False)
        return flops, params
    except ImportError: return None, None
