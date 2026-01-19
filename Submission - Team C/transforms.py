from torchvision import transforms as T

# ImageNet normalization (mean, std)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(img_size=224):
    """Training transforms with conservative augmentations for medical images.

    - RandomResizedCrop: small scale variation
    - Horizontal flip: plausible for eye images (be cautious if laterality matters)
    - Small rotation and color jitter: simulate acquisition variability
    - Normalize with ImageNet stats (model pretrained on ImageNet)
    """
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.08),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_valid_transforms(img_size=224):
    """Validation / Test transforms: deterministic resizing and center crop."""
    return T.Compose([
        T.Resize(int(img_size * 1.14)),  # 256 for 224 crop
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


if __name__ == "__main__":
    # Quick check
    tr = get_train_transforms()
    te = get_valid_transforms()
    print(tr)
    print(te)
from torchvision import transforms

def get_test_transform():
  """
  This function must return the transforms to apply to test images.
  IMPORTANT: These should match the transforms you used during training
  (without data augmentation).
  Returns:
    transform: torchvision.transforms.Compose object
  """
  return transforms.Compose([]) # TODO: Implement appropriate transforms for the dataset