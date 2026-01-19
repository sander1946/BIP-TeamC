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