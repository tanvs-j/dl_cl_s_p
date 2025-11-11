from __future__ import annotations
import os
import csv
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from app.io_utils import load_recording
from app.preprocess import preprocess_for_model
from models.network import EEGNet1D

class EEGDataset(Dataset):
    def __init__(self, data_dir: Path, labels_csv: Path):
        self.items = []
        with open(labels_csv, 'r', newline='') as f:
            r = csv.DictReader(f)
            for row in r:
                fpath = data_dir / row['file']
                y = int(row['label'])
                self.items.append((fpath, y))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fpath, y = self.items[idx]
        with open(fpath, 'rb') as fb:
            rec = load_recording(fb, fpath.name)
        X, sfreq = rec['data'], rec['sfreq']
        wins = preprocess_for_model(X, sfreq)
        if wins.size == 0:
            # fabricate empty
            wins = np.zeros((1, X.shape[0], min(1000, X.shape[1])), dtype=np.float32)
        return wins, y


def collate_batch(batch):
    # batch: list of (wins, y) with variable number of windows, stack windows and repeat labels
    xs, ys = [], []
    for wins, y in batch:
        xs.append(torch.from_numpy(wins).float())
        ys.append(torch.full((wins.shape[0],), y, dtype=torch.long))
    x = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)
    return x, y


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_dir = Path(args.data_dir)
    labels_csv = Path(args.labels_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = EEGDataset(data_dir, labels_csv)
    dl = DataLoader(ds, batch_size=args.batch_files, shuffle=True, collate_fn=collate_batch)

    # Peek to get channel count
    x0, y0 = next(iter(dl))
    in_ch = x0.shape[1]

    model = EEGNet1D(in_channels=in_ch, num_classes=2).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    best = 0.0
    for epoch in range(args.epochs):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in dl:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item()) * x.size(0)
            pred = logits.argmax(1)
            correct += int((pred == y).sum().item())
            total += int(x.size(0))
        acc = correct / max(1, total)
        print(f"Epoch {epoch+1}/{args.epochs} loss={loss_sum/total:.4f} acc={acc:.3f}")
        if acc > best:
            best = acc
            torch.save({
                'state_dict': model.state_dict(),
                'model_kwargs': {'in_channels': in_ch, 'num_classes': 2}
            }, out_dir / 'best.pt')
            print("Saved best.pt")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--labels_csv', required=True)
    ap.add_argument('--out_dir', default='models/checkpoints')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch_files', type=int, default=2, help='Number of recordings per batch (windows are stacked)')
    ap.add_argument('--lr', type=float, default=1e-3)
    args = ap.parse_args()
    train(args)
