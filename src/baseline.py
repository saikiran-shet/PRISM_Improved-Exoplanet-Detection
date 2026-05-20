import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
import logging

from src.data.dataset import get_dataloaders

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


class BaselineCNN(nn.Module):
    """
    Vanilla 1D CNN that classifies raw X(t) directly.
    No decomposition. No physics. No MINE.
    Single encoder → single latent vector → binary classifier.

    This is what PRISM is compared against in the paper.
    Architecturally similar to Shallue & Vanderburg (2018).

    Why keep it simple?
        The baseline should be a fair comparison — same data,
        similar capacity, same training procedure.
        If the baseline is too weak, PRISM looks good for the
        wrong reasons. Match encoder depth to PRISM's encoder.

    Input : (B, 1, 1024)
    Output: (B, 1) probability
    """

    def __init__(self, in_channels=1, latent_dim=64, dropout_p=0.3):
        super().__init__()

        # Same backbone depth as PRISMEncoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        flat_size = self._get_flat_size(in_channels)

        self.classifier = nn.Sequential(
            nn.Linear(flat_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def _get_flat_size(self, in_channels):
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 1024)
            return self.encoder(dummy).view(1, -1).shape[1]

    def forward(self, x):
        feat = self.encoder(x).view(x.size(0), -1)
        return self.classifier(feat)


def train_baseline(proc_dir="data/processed", n_injections=500,
                   epochs=50, patience=8, lr=1e-4, batch_size=32,
                   ckpt_dir="outputs/checkpoints"):
    """
    Trains the baseline CNN with the same data and procedure as PRISM.
    Uses identical: optimizer, LR, early stopping, batch size.
    Only difference: no decomposition, no MINE, no physics loss.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = get_dataloaders(
        proc_dir=proc_dir, n_injections=n_injections,
        batch_size=batch_size,
    )

    model    = BaselineCNN().to(device)
    bce      = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    best_val_loss = float('inf')
    patience_ctr  = 0
    os.makedirs(ckpt_dir, exist_ok=True)

    log.info("Training baseline CNN...")
    for epoch in range(1, epochs + 1):

        # Train
        model.train()
        train_loss = 0.0
        for x, labels in tqdm(train_loader, desc=f"Baseline epoch {epoch}",
                               leave=False):
            x, labels = x.to(device), labels.to(device).view(-1, 1)
            optimizer.zero_grad()
            prob = model(x)
            loss = bce(prob, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validate
        model.eval()
        val_loss = 0.0
        correct = total = 0
        with torch.no_grad():
            for x, labels in val_loader:
                x, labels = x.to(device), labels.to(device).view(-1, 1)
                prob = model(x)
                val_loss += bce(prob, labels).item()
                preds = (prob >= 0.5).float()
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

        val_loss /= len(val_loader)
        val_acc   = correct / total
        scheduler.step(val_loss)

        log.info(f"Epoch {epoch:3d} | train={train_loss/len(train_loader):.4f} "
                 f"val={val_loss:.4f} acc={val_acc:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ctr  = 0
            torch.save(model.state_dict(),
                       os.path.join(ckpt_dir, "baseline_best.pt"))
            log.info(f"  Checkpoint saved (val_loss={best_val_loss:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                log.info(f"Early stopping at epoch {epoch}")
                break

    # Final evaluation on test set
    model.load_state_dict(
        torch.load(os.path.join(ckpt_dir, "baseline_best.pt"),
                   map_location=device)
    )
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, labels in test_loader:
            prob = model(x.to(device)).squeeze(1)
            all_probs.append(prob.cpu().numpy())
            all_labels.append(labels.numpy())

    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds  = (probs >= 0.5).astype(int)

    auc_roc = roc_auc_score(labels, probs)
    auc_pr  = average_precision_score(labels, probs)
    acc     = (preds == labels).mean()

    log.info("\n" + "="*50)
    log.info("BASELINE CNN — TEST RESULTS")
    log.info("="*50)
    log.info(f"  Accuracy : {acc:.4f}")
    log.info(f"  AUC-ROC  : {auc_roc:.4f}")
    log.info(f"  AUC-PR   : {auc_pr:.4f}")
    log.info("="*50)

    os.makedirs("outputs/results", exist_ok=True)
    np.save("outputs/results/baseline_probs.npy",  probs)
    np.save("outputs/results/baseline_labels.npy", labels)
    np.save("outputs/results/baseline_metrics.npy",
            dict(acc=acc, auc_roc=auc_roc, auc_pr=auc_pr))

    return dict(acc=acc, auc_roc=auc_roc, auc_pr=auc_pr)


if __name__ == "__main__":
    train_baseline()