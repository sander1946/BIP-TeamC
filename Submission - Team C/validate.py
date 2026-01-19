import os
import csv
import json
import torch
import numpy as np
from torchvision import datasets
from torchvision import transforms as T

from model import get_model
from transforms import get_valid_transforms


def find_label_csv(root: str):
    if not os.path.isdir(root):
        return None
    for fname in os.listdir(root):
        if fname.lower().endswith(".csv") and "label" in fname.lower():
            return os.path.join(root, fname)
    return None


def load_dataset(data_dir: str, img_size=224):
    test_dir = os.path.join(data_dir, "Test")
    transform = get_valid_transforms(img_size)

    # Prefer ImageFolder layout
    if os.path.isdir(test_dir):
        # detect subfolders with images
        has_sub = any(
            os.path.isdir(os.path.join(test_dir, d))
            for d in os.listdir(test_dir)
        )
        if has_sub:
            ds = datasets.ImageFolder(test_dir, transform=transform)
            return ds, ds.class_to_idx

    # Fallback: CSV listing
    csv_path = find_label_csv(test_dir) or find_label_csv(data_dir)
    if csv_path:
        samples = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if len(rows) == 0:
            raise ValueError(f"Label CSV {csv_path} is empty")
        # Infer columns
        first = rows[0]
        keys = list(first.keys())
        # guess filename and label columns
        fn_key = None
        lbl_key = None
        for k in keys:
            lk = k.lower()
            if "file" in lk or "image" in lk or "filename" in lk:
                fn_key = k
            if "label" in lk or "class" in lk:
                lbl_key = k
        if fn_key is None:
            fn_key = keys[0]
        if lbl_key is None:
            lbl_key = keys[-1]

        labels_set = sorted({r[lbl_key] for r in rows})
        class_to_idx = {c: i for i, c in enumerate(labels_set)}

        for r in rows:
            fname = r[fn_key]
            path = fname if os.path.isabs(fname) else os.path.join(os.path.dirname(csv_path), fname)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Referenced image {path} not found")
            lbl = class_to_idx[r[lbl_key]]
            samples.append((path, lbl))

        class CSVTestDataset(torch.utils.data.Dataset):
            def __init__(self, samples, transform):
                self.samples = samples
                self.transform = transform

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                path, lbl = self.samples[idx]
                from PIL import Image

                img = Image.open(path).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                return img, lbl, path

        ds = CSVTestDataset(samples, transform)
        return ds, class_to_idx

    raise FileNotFoundError(
        f"Could not find test data in {data_dir}. Provide Test/<class>/* or a labels CSV in {data_dir} or {test_dir}"
    )


def evaluate(data_dir: str, model_path: str = "model.pth", batch_size: int = 16, device=None, out_prefix: str = "predictions"):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    ds, class_to_idx = load_dataset(data_dir)

    # If using ImageFolder, dataset __getitem__ returns (img,label)
    is_imagefolder = isinstance(ds, datasets.ImageFolder)

    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)

    # Prepare model and load weights
    model = get_model(num_classes=len(class_to_idx), pretrained=False, freeze_backbone=False)
    model = model.to(device)
    state = torch.load(model_path, map_location=device)
    try:
        model.load_state_dict(state)
    except Exception:
        # Attempt to load partial/compatible keys
        model.load_state_dict(state, strict=False)

    model.eval()

    # Reverse mapping
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Determine which class index corresponds to glaucoma (positive class)
    pos_idx = None
    for name, idx in class_to_idx.items():
        if "glauc" in name.lower():
            pos_idx = idx
            break
    if pos_idx is None:
        # prefer a label that is not 'normal' if possible
        for name, idx in class_to_idx.items():
            if "norm" not in name.lower() and "healthy" not in name.lower():
                pos_idx = idx
                break
    if pos_idx is None:
        # fallback: use index 1 if exists, else 0
        pos_idx = 1 if len(class_to_idx) > 1 else 0

    y_true = []
    y_pred = []
    y_prob = []
    files = []

    with torch.no_grad():
        for batch in loader:
            if is_imagefolder:
                imgs, labels = batch
                paths = [p for p, _ in loader.dataset.samples[len(files): len(files)+len(labels)]] if hasattr(loader.dataset, 'samples') else [""] * len(labels)
            else:
                imgs, labels, paths = batch

            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.tolist())
            # probability of positive class (assume class index 1 corresponds to glaucoma)
            if probs.shape[1] >= 2:
                pos_probs = probs[:, 1].tolist()
            else:
                pos_probs = probs[:, 0].tolist()

            y_prob.extend(pos_probs)
            files.extend(paths)

    # Metrics
    try:
        from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix, classification_report, roc_auc_score

        bacc = balanced_accuracy_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred).tolist()
        report = classification_report(y_true, y_pred, output_dict=True)
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            auc = None
    except Exception:
        # minimal fallbacks
        def recalls(y_true, y_pred):
            classes = sorted(set(y_true))
            recs = []
            for c in classes:
                tp = sum(int(t == c and p == c) for t, p in zip(y_true, y_pred))
                fn = sum(int(t == c and p != c) for t, p in zip(y_true, y_pred))
                recs.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
            return sum(recs) / len(recs)

        bacc = recalls(y_true, y_pred)
        acc = sum(int(t == p) for t, p in zip(y_true, y_pred)) / len(y_true)
        cm = []
        report = {}
        auc = None

    # Save CSV of predictions
    out_csv = f"{out_prefix}.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "true_label", "pred_label", "prob_positive"])
        for fp, t, p, prob in zip(files, y_true, y_pred, y_prob):
            writer.writerow([fp, idx_to_class.get(t, t), idx_to_class.get(p, p), float(prob)])

    results = {
        "balanced_accuracy": float(bacc) if bacc is not None else None,
        "accuracy": float(acc) if acc is not None else None,
        "roc_auc": float(auc) if auc is not None else None,
        "confusion_matrix": cm,
        "classification_report": report,
        "n_samples": len(y_true),
        "class_to_idx": class_to_idx,
        "predictions_csv": out_csv,
    }

    out_json = f"{out_prefix}_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print("Evaluation finished")
    print(json.dumps({k: results[k] for k in ["balanced_accuracy", "accuracy", "roc_auc", "n_samples"]}, indent=2))
    print(f"Predictions saved to {out_csv}, results to {out_json}")

    return results


if __name__ == "__main__":
    data_dir = os.path.join("Datasets", "Complete")
    model_path = "model.pth"
    evaluate(data_dir, model_path)
