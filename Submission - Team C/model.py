import torch
import torch.nn as nn
from torchvision import models


def get_model(num_classes=2, pretrained=True, freeze_backbone=True, dropout=0.5):
  """Returns a transfer-learning ResNet34 based model suitable for binary classification.

  Design choices:
  - ResNet34 pre-trained on ImageNet: good trade-off between capacity and overfitting risk.
  - Freeze backbone by default to avoid overfitting on small dataset; can unfreeze later.
  - Custom classifier head with dropout and a small intermediate layer.

  Args:
    num_classes (int): number of output classes (should be 2 for this competition).
    pretrained (bool): load ImageNet weights if True.
    freeze_backbone (bool): whether to freeze backbone parameters initially.
    dropout (float): dropout probability in classifier head.

  Returns:
    model (nn.Module): PyTorch model with logits output of shape (batch, num_classes).
  """

  backbone = models.resnet34(pretrained=pretrained)

  # Optionally freeze backbone parameters
  if freeze_backbone:
    for param in backbone.parameters():
      param.requires_grad = False

  # Capture feature dimension from the original fc
  feat_dim = backbone.fc.in_features

  # Small classifier head: [feat_dim] -> [512] -> [num_classes]
  classifier = nn.Sequential(
    nn.Dropout(p=dropout),
    nn.Linear(feat_dim, 512),
    nn.ReLU(inplace=True),
    nn.BatchNorm1d(512),
    nn.Dropout(p=dropout * 0.5),
    nn.Linear(512, num_classes),
  )


  class GlaucomaModel(nn.Module):
    def __init__(self, backbone, head):
      super().__init__()
      self.backbone = backbone
      self.head = head

    def forward(self, x):
      # Run ResNet layers up to avgpool (skip original fc)
      x = self.backbone.conv1(x)
      x = self.backbone.bn1(x)
      x = self.backbone.relu(x)
      x = self.backbone.maxpool(x)
      x = self.backbone.layer1(x)
      x = self.backbone.layer2(x)
      x = self.backbone.layer3(x)
      x = self.backbone.layer4(x)
      x = self.backbone.avgpool(x)
      x = torch.flatten(x, 1)
      x = self.head(x)
      return x

  model = GlaucomaModel(backbone, classifier)
  return model


def logits_to_probs(logits):
  """Convert model logits (shape [N, C]) to probability percentages.

  For multi-class (C>=2) outputs use softmax; for single-logit binary use sigmoid.

  Args:
    logits (torch.Tensor): raw model outputs (no activation).

  Returns:
    probs (torch.Tensor): probabilities in percent (0-100), same shape as logits.
  """
  if logits.dim() == 1:
    logits = logits.unsqueeze(0)
  if logits.size(1) == 1:
    probs = torch.sigmoid(logits)
    probs = torch.cat([1 - probs, probs], dim=1)  # [neg, pos]
  else:
    probs = torch.softmax(logits, dim=1)

  return probs * 100.0


if __name__ == "__main__":
  # Quick sanity check
  model = get_model()
  x = torch.randn(2, 3, 224, 224)
  logits = model(x)
  probs = logits_to_probs(logits)
  print("Logits shape:", logits.shape)
  print("Probs (percent):", probs)
