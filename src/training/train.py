import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.training.evaluate import ndcg_at_k, recall_at_k

logger = logging.getLogger(__name__)


def encode_all_items(
    model,
    img_emb: np.ndarray,
    txt_emb: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    """Pass all precomputed CLIP+SBERT embeddings through ItemTower -> (M, 256) numpy array."""
    model.eval()
    all_embs = []
    M = len(img_emb)
    with torch.no_grad():
        for start in range(0, M, batch_size):
            img_b = torch.from_numpy(img_emb[start : start + batch_size]).to(device)
            txt_b = torch.from_numpy(txt_emb[start : start + batch_size]).to(device)
            emb = model.item_tower(img_b, txt_b)
            all_embs.append(emb.cpu().numpy())
    return np.concatenate(all_embs, axis=0)


def _collect_user_embs(
    model,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect user embeddings and true item indices for a full DataLoader."""
    model.eval()
    all_user_embs = []
    all_true_idx  = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="  user embs", leave=False):
            B, N, _ = batch["user_seq_img"].shape
            seq_img  = batch["user_seq_img"].view(B * N, -1).to(device)
            seq_txt  = batch["user_seq_txt"].view(B * N, -1).to(device)
            seq_item = model.item_tower(seq_img, seq_txt).view(B, N, -1)
            user_emb = model.user_tower(seq_item, batch["user_mask"].to(device))
            all_user_embs.append(user_emb.cpu().numpy())
            all_true_idx.append(batch["target_idx"].numpy())
    return np.concatenate(all_user_embs, axis=0), np.concatenate(all_true_idx, axis=0)


def _log_collapse_stats(
    model,
    img_emb: np.ndarray,
    txt_emb: np.ndarray,
    device: torch.device,
    global_step: int,
    current_loss: float,
    current_lr: float,
) -> float:
    """
    Sample 256 random items, run through ItemTower, report:
      - output norm mean/std (sanity; always ~1.0 since we normalize)
      - mean pairwise cosine of 64 items (healthy < 0.5; collapse > 0.95)
    Returns mean pairwise cosine so caller can decide to stop.
    """
    model.eval()
    rng = np.random.default_rng(global_step)      # reproducible per step
    idx256 = rng.choice(len(img_emb), 256, replace=False)

    with torch.no_grad():
        img_b = torch.from_numpy(img_emb[idx256]).to(device)
        txt_b = torch.from_numpy(txt_emb[idx256]).to(device)
        embs  = model.item_tower(img_b, txt_b)    # (256, 256) already L2-normalised

        norms     = embs.norm(dim=-1)
        norm_mean = norms.mean().item()
        norm_std  = norms.std().item()

        # Mean pairwise cosine on first 64
        e64  = embs[:64]                           # (64, 256)
        sim  = e64 @ e64.T                         # (64, 64)
        mask = ~torch.eye(64, dtype=torch.bool, device=device)
        mean_cos = sim[mask].mean().item()

    model.train()
    print(
        f"  [step {global_step:>5}] loss={current_loss:.4f} | "
        f"lr={current_lr:.2e} | "
        f"item_norm mean={norm_mean:.4f} std={norm_std:.6f} | "
        f"mean_pairwise_cos={mean_cos:.4f}"
    )
    return mean_cos


def run_sanity_check(model, train_dataset, device: torch.device) -> None:
    """
    One forward+backward on 8 samples before the main loop.
    Expected at random init:
      loss   ~= log(batch_size) = log(8) ~= 2.08
      logits roughly in [-10, 10]
      grad norm > 0 and finite
    Raises RuntimeError if any check fails.
    """
    print("\n--- Sanity check (batch_size=8) ---")
    n = min(8, len(train_dataset))
    loader = DataLoader(Subset(train_dataset, list(range(n))), batch_size=n, shuffle=False)
    batch  = next(iter(loader))

    model.train()
    model.zero_grad()

    B      = batch["target_img"].shape[0]
    logits = model(
        batch["user_seq_img"].to(device),
        batch["user_seq_txt"].to(device),
        batch["user_mask"].to(device),
        batch["target_img"].to(device),
        batch["target_txt"].to(device),
    )
    labels = torch.arange(B, device=device)
    loss   = F.cross_entropy(logits, labels)
    loss.backward()

    grad_norm = sum(
        p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None
    ) ** 0.5

    expected = math.log(B)
    print(f"  Loss:        {loss.item():.4f}  (expected ~{expected:.4f})")
    print(f"  Logit range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
    print(f"  Grad norm:   {grad_norm:.6f}")

    model.zero_grad()

    failed = False
    if not torch.isfinite(loss):
        print("  ERROR: loss is NaN or Inf")
        failed = True
    if loss.item() < 1.0 or loss.item() > 5.0:
        print(f"  WARNING: loss {loss.item():.4f} outside expected range [1.0, 5.0]")
        failed = True
    if not np.isfinite(grad_norm) or grad_norm == 0.0:
        print(f"  ERROR: grad norm is {grad_norm}")
        failed = True

    if failed:
        raise RuntimeError("Sanity check failed -- see messages above.")

    print("  Sanity check PASSED -- proceeding to full training.\n")


def train(
    config: dict,
    model,
    train_dataset,
    val_dataset,
    all_img_emb: np.ndarray,
    all_txt_emb: np.ndarray,
    device: torch.device,
    test_dataset=None,
    checkpoint_path: str = "checkpoints/best.pt",
) -> dict:
    """
    Full training loop:
    - AdamW optimiser with linear LR warmup then constant
    - Mixed precision via torch.amp
    - Grad clipping (max_norm=1.0)
    - Collapse diagnostics every 100 global steps
    - Val loss + full retrieval eval (Recall@10, NDCG@10) each epoch
    - Early stopping on val loss (patience=2)
    - Best checkpoint -> checkpoints/best.pt
    """
    tcfg         = config["training"]
    bs           = tcfg["batch_size"]
    num_epochs   = tcfg["num_epochs"]
    warmup_steps = tcfg.get("warmup_steps", 0)
    patience     = 2

    train_loader = DataLoader(
        train_dataset, batch_size=bs, shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=bs, shuffle=False, num_workers=0, pin_memory=True
    )
    test_loader = (
        DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=0, pin_memory=True)
        if test_dataset is not None else None
    )

    optimiser = torch.optim.AdamW(
        model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"]
    )

    # Linear warmup then constant LR
    def lr_lambda(step: int) -> float:
        if warmup_steps <= 0:
            return 1.0
        return min(1.0, (step + 1) / warmup_steps)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda=lr_lambda)
    scaler    = torch.amp.GradScaler("cuda")

    Path("checkpoints").mkdir(exist_ok=True)

    best_val_loss  = float("inf")
    no_improve     = 0
    best_metrics: dict = {}
    global_step    = 0

    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        train_loss_sum = 0.0
        n_train        = 0
        t0             = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [train]", leave=True)
        for step, batch in enumerate(pbar):
            optimiser.zero_grad()
            with torch.amp.autocast("cuda"):
                logits = model(
                    batch["user_seq_img"].to(device),
                    batch["user_seq_txt"].to(device),
                    batch["user_mask"].to(device),
                    batch["target_img"].to(device),
                    batch["target_txt"].to(device),
                )
                B    = logits.shape[0]
                loss = F.cross_entropy(logits, torch.arange(B, device=device))

            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimiser)
            scaler.update()
            scheduler.step()
            global_step += 1

            train_loss_sum += loss.item()
            n_train        += 1

            if step % 50 == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Collapse diagnostic every 100 global steps
            if global_step % 100 == 0:
                current_lr = scheduler.get_last_lr()[0]
                mean_cos   = _log_collapse_stats(
                    model, all_img_emb, all_txt_emb, device,
                    global_step, loss.item(), current_lr,
                )
                if mean_cos > 0.9:
                    print(
                        f"  WARNING: mean_pairwise_cos={mean_cos:.4f} > 0.9 "
                        "-- possible collapse. Monitoring closely."
                    )
                model.train()

        avg_train_loss = train_loss_sum / max(n_train, 1)
        elapsed        = time.time() - t0

        # Val loss
        model.eval()
        val_loss_sum = 0.0
        n_val        = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [val loss]", leave=False):
                with torch.amp.autocast("cuda"):
                    logits = model(
                        batch["user_seq_img"].to(device),
                        batch["user_seq_txt"].to(device),
                        batch["user_mask"].to(device),
                        batch["target_img"].to(device),
                        batch["target_txt"].to(device),
                    )
                    B    = logits.shape[0]
                    loss = F.cross_entropy(logits, torch.arange(B, device=device))
                val_loss_sum += loss.item()
                n_val        += 1
        avg_val_loss = val_loss_sum / max(n_val, 1)

        # Full retrieval eval — encode all items once, reuse for both metrics
        print(f"  Encoding all {len(all_img_emb):,} items for retrieval eval...")
        all_item_embs          = encode_all_items(model, all_img_emb, all_txt_emb, device)
        val_user_embs, val_idx = _collect_user_embs(model, val_loader, device)
        val_recall = recall_at_k(val_user_embs, all_item_embs, val_idx, k=10)
        val_ndcg   = ndcg_at_k(val_user_embs,   all_item_embs, val_idx, k=10)

        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_loss={avg_val_loss:.4f} | "
            f"val_Recall@10={val_recall:.4f} | "
            f"val_NDCG@10={val_ndcg:.4f} | "
            f"time={elapsed:.0f}s"
        )

        # Checkpoint + early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve    = 0
            best_metrics  = {
                "epoch":           epoch,
                "train_loss":      avg_train_loss,
                "val_loss":        avg_val_loss,
                "val_recall_at_10": val_recall,
                "val_ndcg_at_10":  val_ndcg,
            }
            torch.save(
                {
                    "epoch":               epoch,
                    "model_state_dict":    model.state_dict(),
                    "optimiser_state_dict": optimiser.state_dict(),
                    "metrics":             best_metrics,
                    "config":              config,
                },
                checkpoint_path,
            )
            print(f"  [saved] Checkpoint saved (best val_loss={best_val_loss:.4f})")
        else:
            no_improve += 1
            print(f"  No improvement for {no_improve}/{patience} epochs.")
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch}.")
                break

    # Optional test evaluation using best checkpoint
    if test_loader is not None and best_metrics:
        print("\n  Loading best checkpoint for test evaluation...")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        all_item_embs             = encode_all_items(model, all_img_emb, all_txt_emb, device)
        test_user_embs, test_idx  = _collect_user_embs(model, test_loader, device)
        test_recall = recall_at_k(test_user_embs, all_item_embs, test_idx, k=10)
        test_ndcg   = ndcg_at_k(test_user_embs,   all_item_embs, test_idx, k=10)
        best_metrics["test_recall_at_10"] = test_recall
        best_metrics["test_ndcg_at_10"]   = test_ndcg
        print(f"  Test | Recall@10={test_recall:.4f} | NDCG@10={test_ndcg:.4f}")

    print(f"\nTraining complete. Best: {best_metrics}")
    return best_metrics
