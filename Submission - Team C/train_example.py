"""
Example training loop for small glaucoma dataset.

Notes:
- Uses `Submission - Team C/model.py` -> `get_model`
- Uses transforms defined in `transforms.py` in the same folder
- Assumes dataset organized for `torchvision.datasets.ImageFolder` or custom Dataset
"""
import os
import copy
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from PIL import Image
import csv
from typing import List, Tuple, Optional

from model import get_model, logits_to_probs
from transforms import get_train_transforms, get_valid_transforms


def train(
    data_dir,
    epochs=30,
    batch_size=16,
    lr=2e-4,
    weight_decay=1e-4,
    patience=7,
    device=None,
    unfreeze_after_epoch=3,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_dir = os.path.join(data_dir, "Train")
    val_dir = os.path.join(data_dir, "Test")

    # Helper to detect ImageFolder-style class subdirectories
    def has_class_folders(path: str) -> bool:
        if not os.path.isdir(path):
            return False
        for entry in os.listdir(path):
            p = os.path.join(path, entry)
            if os.path.isdir(p):
                # check for any image files inside
                for f in os.listdir(p):
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
                        return True
        return False

    # CSV-based dataset fallback
    class CSVDataset(torch.utils.data.Dataset):
        def __init__(self, csv_path: str, img_root: str, transform=None):
            self.samples: List[Tuple[str, int]] = []
            self.transform = transform
            self.img_root = img_root
            # Read CSV expecting columns: filename,label
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            if len(rows) == 0:
                raise ValueError(f"Labels CSV {csv_path} is empty")

            # Collect labels and map to integers if necessary
            labels = [r.get("label") or r.get("labels") or list(r.values())[-1] for r in rows]
            unique = sorted(list({l for l in labels}))
            label_to_idx = {lab: i for i, lab in enumerate(unique)}

            for r in rows:
                filename = r.get("filename") or r.get("file") or list(r.values())[0]
                label = r.get("label") or r.get("labels") or list(r.values())[-1]
                img_path = filename if os.path.isabs(filename) else os.path.join(img_root, filename)
                if not os.path.exists(img_path):
                    # try relative to csv location
                    img_path = os.path.join(os.path.dirname(csv_path), filename)
                if not os.path.exists(img_path):
                    raise FileNotFoundError(f"Image file {filename} referenced in {csv_path} not found")
                self.samples.append((img_path, label_to_idx[label]))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            path, label = self.samples[idx]
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, label

    # Choose dataset loader depending on directory layout
    if has_class_folders(train_dir) and has_class_folders(val_dir):
        train_ds = datasets.ImageFolder(train_dir, transform=get_train_transforms())
        val_ds = datasets.ImageFolder(val_dir, transform=get_valid_transforms())
    else:
        # Try to find a CSV file listing filenames and labels inside data_dir
        def find_label_csv(root: str) -> Optional[str]:
            candidates = []
            for fname in os.listdir(root):
                if fname.lower().endswith(".csv") and "label" in fname.lower():
                    candidates.append(os.path.join(root, fname))
            return candidates[0] if candidates else None

        train_csv = find_label_csv(train_dir) or find_label_csv(data_dir)
        val_csv = find_label_csv(val_dir) or find_label_csv(data_dir)

        if train_csv and val_csv:
            train_ds = CSVDataset(train_csv, img_root=train_dir, transform=get_train_transforms())
            val_ds = CSVDataset(val_csv, img_root=val_dir, transform=get_valid_transforms())
        else:
            raise FileNotFoundError(
                f"Couldn't find ImageFolder-style class subfolders in {train_dir} and {val_dir} nor label CSVs.\n"
                "Organize images as: Train/<class_name>/*.jpg and Test/<class_name>/*.jpg OR provide CSV files named like '*labels*.csv' with columns 'filename,label'."
            )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    model = get_model(num_classes=2, pretrained=True, freeze_backbone=True)
    model = model.to(device)

    # Only train head params at first
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    criterion = nn.CrossEntropyLoss()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_score = 0.0
    epochs_no_improve = 0

    try:
        from sklearn.metrics import balanced_accuracy_score
    except Exception:
        balanced_accuracy_score = None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        epoch_loss = running_loss / len(train_ds)

        # Validation
        model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                logits = model(imgs)
                preds = torch.argmax(logits, dim=1)
                y_true.extend(labels.cpu().numpy().tolist())
                y_pred.extend(preds.cpu().numpy().tolist())

        if balanced_accuracy_score is not None:
            val_bacc = balanced_accuracy_score(y_true, y_pred)
        else:
            # fallback: compute per-class recall average
            from collections import Counter

            def recalls(y_true, y_pred):
                classes = sorted(set(y_true))
                recs = []
                for c in classes:
                    tp = sum(int(t == c and p == c) for t, p in zip(y_true, y_pred))
                    fn = sum(int(t == c and p != c) for t, p in zip(y_true, y_pred))
                    recs.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
                return sum(recs) / len(recs)

            val_bacc = recalls(y_true, y_pred)

        print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f} - val_bacc: {val_bacc:.4f}")

        scheduler.step(val_bacc)

        # Early stopping & save best
        if val_bacc > best_score:
            best_score = val_bacc
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            torch.save(model.state_dict(), "model.pth")
            print("Saved best model to model.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered")
                break

        # Optionally unfreeze backbone after a few epochs
        if epoch + 1 == unfreeze_after_epoch:
            print("Unfreezing backbone for fine-tuning")
            for p in model.backbone.parameters():
                p.requires_grad = True
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(params, lr=lr * 0.2, weight_decay=weight_decay)

    # Load best weights before returning
    model.load_state_dict(best_model_wts)
    return model


if __name__ == "__main__":
    # Example usage: adjust `data_dir` to point to `Datasets/Complete`
    data_dir = os.path.join("Datasets", "Complete")
    print("Training with data_dir:", data_dir)
    trained = train(data_dir, epochs=25)
    print("Done")
