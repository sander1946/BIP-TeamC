# BIP-TeamC — Project README

Short overview of the repository structure and purpose of key files.

## Project structure

- BIP-TeamC
  - Datasets/
    - Glaucoma/
      - Train/ — training images for glaucoma class
      - Test/ — test images for glaucoma class
    - Normal/
      - Train/ — training images for normal class
      - Test/ — test images for normal class
  - Documents/
    - train_RIMONE_original.csv — training CSV (original)
    - train_RIMONE_original_extended.csv — training CSV (extended)
    - test_RIMONE_original.csv — test CSV (original)
    - test_RIMONE_original_extended.csv — test CSV (extended)
  - Submission - Team C/
    - info.json — submission metadata
    - model.py — model definition / inference script
    - transforms.py — augmentation / preprocessing functions used by model
  - Testing/
    - model.ipynb — exploratory notebook for the model
    - transforms.ipynb — notebook for transform/augmentation experiments
  - README.md — (this file)

## Quick notes

- Datasets are organized by class (Glaucoma / Normal) and split (Train / Test).
- CSVs in Documents correspond to train/test splits and variations (original vs extended).
- Submission - Team C contains the deliverable model code; adapt or import these for training/inference.
- Notebooks in Testing are for development and experimentation.

For usage, run notebooks or import Submission - Team C/model.py and Submission - Team C/transforms.py in your training or inference pipeline.
