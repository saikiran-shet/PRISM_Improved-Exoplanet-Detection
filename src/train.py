import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import logging

from src.models       import build_prism
from src.losses       import build_losses
from src.data.dataset import get_dataloaders

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


DEFAULT_CFG = dict(
    latent_dim   = 64,
    seq_len      = 1024,
    dropout_p    = 0.3,
    n_injections = 500,
    batch_size   = 32,
    val_split    = 0.15,
    test_split   = 0.15,
    epochs       = 150,
    patience     = 15,
    lr_prism     = 1e-4,
    lr_mine      = 1e-4,
    lambda_recon = 1.0,
    lambda_mi    = 0.1,
    lambda_phys  = 0.1,
    lambda_cls   = 1.0,
    mine_steps   = 1,
    ckpt_dir     = "outputs/checkpoints",
    proc_dir     = "data/processed",
)


def compute_pos_weight(train_loader):
    """
    Computes BCE pos_weight from the training DataLoader.

    train_loader.dataset is a Subset (from random_split).
    Subset wraps PRISMDataset — access via .dataset to get samples.

    pos_weight = n_negative / n_positive
    Passed into ClassifierLoss to down-weight majority planet class.
    Called ONCE before training — never inside the batch loop.
    """
    # Unwrap Subset → PRISMDataset → samples list
    full_dataset = train_loader.dataset.dataset
    all_samples  = full_dataset.samples

    # Only count samples that are actually in this split
    # using the Subset indices
    subset_indices = train_loader.dataset.indices
    subset_labels  = [all_samples[i][1] for i in subset_indices]

    n_pos = sum(1 for l in subset_labels if l == 1.0)
    n_neg = sum(1 for l in subset_labels if l == 0.0)

    if n_pos == 0:
        log.warning("No positive samples in training set — pos_weight set to 1.0")
        return 1.0

    weight = n_neg / n_pos
    log.info(f"Class balance — planet: {n_pos}, "
             f"no-planet: {n_neg}, pos_weight: {weight:.4f}")
    return weight


def train_one_epoch(models, losses, prism_opt, mine_opt,
                    loader, device, cfg, epoch):
    enc   = models['encoder'].train()
    s_dec = models['stellar_decoder'].train()
    t_dec = models['transit_decoder'].train()
    clf   = models['classifier'].train()

    l_recon_sum = l_mi_sum = l_phys_sum = l_cls_sum = l_total_sum = 0.0
    n_batches   = 0

    for x, labels in tqdm(loader, desc=f"Epoch {epoch} train", leave=False):
        x      = x.to(device)
        labels = labels.to(device)

        # Step 1 — Update MINENetwork (maximize MI bound)
        for _ in range(cfg['mine_steps']):
            mine_opt.zero_grad()
            with torch.no_grad():
                z_s, z_t = enc(x)
            mi_est    = losses['mine'](z_s.detach(), z_t.detach())
            mine_loss = -mi_est
            mine_loss.backward()
            mine_opt.step()

        # Step 2 — Update PRISM (minimize total loss)
        prism_opt.zero_grad()

        z_s, z_t           = enc(x)
        S_t                = s_dec(z_s)
        T_t                = t_dec(z_t)
        prob               = clf(z_t)

        l_recon            = losses['recon'](x, S_t, T_t)
        l_mi               = losses['mine'](z_s, z_t)
        l_phys, l_ma, l_tv = losses['physics'](S_t, T_t)
        l_cls              = losses['classify'](prob, labels)

        total = (cfg['lambda_recon'] * l_recon
               + cfg['lambda_mi']    * l_mi
               + cfg['lambda_phys']  * l_phys
               + cfg['lambda_cls']   * l_cls)

        total.backward()

        nn.utils.clip_grad_norm_(
            list(enc.parameters())
          + list(s_dec.parameters())
          + list(t_dec.parameters())
          + list(clf.parameters()),
            max_norm=1.0
        )

        prism_opt.step()

        l_recon_sum += l_recon.item()
        l_mi_sum    += l_mi.item()
        l_phys_sum  += l_phys.item()
        l_cls_sum   += l_cls.item()
        l_total_sum += total.item()
        n_batches   += 1

    return dict(
        recon = l_recon_sum / n_batches,
        mi    = l_mi_sum    / n_batches,
        phys  = l_phys_sum  / n_batches,
        cls   = l_cls_sum   / n_batches,
        total = l_total_sum / n_batches,
    )


@torch.no_grad()
def validate(models, losses, loader, device, cfg):
    enc   = models['encoder'].eval()
    s_dec = models['stellar_decoder'].eval()
    t_dec = models['transit_decoder'].eval()
    clf   = models['classifier'].eval()

    l_total_sum = 0.0
    correct = total = 0
    n_batches = 0

    for x, labels in loader:
        x      = x.to(device)
        labels = labels.to(device)

        z_s, z_t          = enc(x)
        S_t               = s_dec(z_s)
        T_t               = t_dec(z_t)
        prob              = clf(z_t)

        l_recon           = losses['recon'](x, S_t, T_t)
        l_mi              = losses['mine'](z_s, z_t)
        l_phys, _, _      = losses['physics'](S_t, T_t)
        l_cls             = losses['classify'](prob, labels)

        total_loss = (cfg['lambda_recon'] * l_recon
                    + cfg['lambda_mi']    * l_mi
                    + cfg['lambda_phys']  * l_phys
                    + cfg['lambda_cls']   * l_cls)

        l_total_sum += total_loss.item()
        n_batches   += 1

        preds   = (prob.squeeze(1) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    return dict(
        total    = l_total_sum / n_batches,
        accuracy = correct / total if total > 0 else 0.0,
    )


def save_checkpoint(models, losses, epoch, val_loss, cfg, tag="best"):
    os.makedirs(cfg['ckpt_dir'], exist_ok=True)
    path = os.path.join(cfg['ckpt_dir'], f"prism_{tag}.pt")
    torch.save(dict(
        epoch           = epoch,
        val_loss        = val_loss,
        cfg             = cfg,
        encoder         = models['encoder'].state_dict(),
        stellar_decoder = models['stellar_decoder'].state_dict(),
        transit_decoder = models['transit_decoder'].state_dict(),
        classifier      = models['classifier'].state_dict(),
        mine            = losses['mine'].state_dict(),
    ), path)
    log.info(f"Checkpoint saved → {path}  (val_loss={val_loss:.4f})")


def load_checkpoint(path, device='cpu'):
    ckpt   = torch.load(path, map_location=device)
    cfg    = ckpt['cfg']
    models = build_prism(latent_dim=cfg['latent_dim'],
                         seq_len=cfg['seq_len'],
                         dropout_p=cfg['dropout_p'])
    models['encoder'].load_state_dict(ckpt['encoder'])
    models['stellar_decoder'].load_state_dict(ckpt['stellar_decoder'])
    models['transit_decoder'].load_state_dict(ckpt['transit_decoder'])
    models['classifier'].load_state_dict(ckpt['classifier'])

    losses = build_losses(latent_dim=cfg['latent_dim'],
                          seq_len=cfg['seq_len'])
    losses['mine'].load_state_dict(ckpt['mine'])

    log.info(f"Loaded checkpoint from {path} (epoch {ckpt['epoch']})")
    return models, losses, cfg


def train(cfg=None):
    if cfg is None:
        cfg = DEFAULT_CFG

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, _ = get_dataloaders(
        proc_dir     = cfg['proc_dir'],
        n_injections = cfg['n_injections'],
        batch_size   = cfg['batch_size'],
        val_split    = cfg['val_split'],
        test_split   = cfg['test_split'],
    )

    # ── pos_weight — computed from actual training split indices ──────────────
    pos_weight = compute_pos_weight(train_loader)

    # ── Models + Losses ───────────────────────────────────────────────────────
    models = build_prism(latent_dim = cfg['latent_dim'],
                         seq_len    = cfg['seq_len'],
                         dropout_p  = cfg['dropout_p'])

    losses = build_losses(latent_dim = cfg['latent_dim'],
                          seq_len    = cfg['seq_len'],
                          pos_weight = pos_weight)

    for m in models.values():
        m.to(device)
    for l in losses.values():
        if hasattr(l, 'parameters'):
            l.to(device)

    # ── Two optimizers ────────────────────────────────────────────────────────
    prism_params = (list(models['encoder'].parameters())
                  + list(models['stellar_decoder'].parameters())
                  + list(models['transit_decoder'].parameters())
                  + list(models['classifier'].parameters()))
    prism_opt = torch.optim.Adam(prism_params, lr=cfg['lr_prism'])

    mine_opt  = torch.optim.Adam(
        losses['mine'].get_mine_parameters(), lr=cfg['lr_mine']
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        prism_opt, mode='min', factor=0.5, patience=3
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float('inf')
    patience_ctr  = 0
    history       = []

    log.info("=" * 60)
    log.info("Starting PRISM training")
    log.info(f"  Epochs      : {cfg['epochs']}")
    log.info(f"  Patience    : {cfg['patience']}")
    log.info(f"  Batch size  : {cfg['batch_size']}")
    log.info(f"  pos_weight  : {pos_weight:.4f}")
    log.info(f"  lr_prism    : {cfg['lr_prism']}")
    log.info(f"  lr_mine     : {cfg['lr_mine']}")
    log.info(f"  λ_recon     : {cfg['lambda_recon']}")
    log.info(f"  λ_mi        : {cfg['lambda_mi']}")
    log.info(f"  λ_phys      : {cfg['lambda_phys']}")
    log.info(f"  λ_cls       : {cfg['lambda_cls']}")
    log.info("=" * 60)

    for epoch in range(1, cfg['epochs'] + 1):

        train_metrics = train_one_epoch(
            models, losses, prism_opt, mine_opt,
            train_loader, device, cfg, epoch
        )

        val_metrics = validate(models, losses, val_loader, device, cfg)

        scheduler.step(val_metrics['total'])

        log.info(
            f"Epoch {epoch:3d}/{cfg['epochs']} | "
            f"train={train_metrics['total']:.4f} "
            f"(recon={train_metrics['recon']:.4f} "
            f"mi={train_metrics['mi']:.4f} "
            f"phys={train_metrics['phys']:.4f} "
            f"cls={train_metrics['cls']:.4f}) | "
            f"val={val_metrics['total']:.4f} "
            f"acc={val_metrics['accuracy']:.3f}"
        )

        history.append(dict(
            epoch = epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}":   v for k, v in val_metrics.items()},
        ))

        if val_metrics['total'] < best_val_loss:
            best_val_loss = val_metrics['total']
            patience_ctr  = 0
            save_checkpoint(models, losses, epoch,
                            best_val_loss, cfg, tag="best")
        else:
            patience_ctr += 1
            log.info(f"  No improvement ({patience_ctr}/{cfg['patience']})")

        if patience_ctr >= cfg['patience']:
            log.info(f"Early stopping at epoch {epoch}.")
            break

    save_checkpoint(models, losses, epoch,
                    val_metrics['total'], cfg, tag="final")

    os.makedirs("outputs/results", exist_ok=True)
    np.save("outputs/results/train_history.npy", history)
    log.info("Training complete. History → outputs/results/train_history.npy")

    return models, losses, history


if __name__ == "__main__":
    train()
