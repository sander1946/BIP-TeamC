import torch.nn as nn
from torchvision import models


def get_model(num_classes=2):
  """
  This function must return your model architecture.
  Args:
    num_classes (int): Number of output classes (always 2 for this competition)
  Returns:
    model: Your PyTorch model
  """
  model = models.resnet50(pretrained=False) # TODO: Choose appropriate base model architecture
  model.fc = nn.Linear(model.fc.in_features, num_classes) 
  return model


def main():
  """
  This function will be used for local testing. When running this file directly, this code will run
  If it's imported it will not run.
  """
  model = get_model()
  print(model)


# For quick testing
if __name__ == "__main__":
  main()
