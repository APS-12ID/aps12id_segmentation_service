"""Round 1 SAM3 fine-tune: freeze both encoders, train the decoder head.

Ming's approved recipe: leave ``backbone.vision_backbone.*`` and
``backbone.language_backbone.*`` frozen; update the transformer decoder,
segmentation head, and prompt/geometry heads on our GCP hole/sample data.

The trainer reuses Meta's own data + loss modules (``Sam3ImageDataset``,
``collate_fn_api``, ``Sam3LossWrapper``, ``HungarianMatcher``, ``Boxes`` /
``IABCEMdetr`` / ``Masks``) so the ``BatchedDatapoint`` construction and
DETR-style matching are exactly what the model expects; the training loop,
freezing, eval, and checkpointing are ours.

Run on sentosa H200; batch size defaults to 1; bf16 autocast (matches inference).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

# Reuse the sys.path shim from the main package so the vendored ``sam3``
# submodule beats any global install.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
from aps12id_segmentation_service.runtime import _prefer_submodule_sam3_package  # noqa: E402
_prefer_submodule_sam3_package()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from scripts.finetune.coco_schema import CATEGORIES  # noqa: E402
from scripts.finetune.dataset import (  # noqa: E402
    GcpCocoDataset,
    split_coco_by_image_id,
)


FROZEN_PREFIXES = ("backbone.vision_backbone.", "backbone.language_backbone.")
COCO_CATEGORY_NAMES = tuple(category["name"] for category in CATEGORIES)
DEFAULT_RESOLUTION = 1008  # SAM3's vision backbone is built at 1008x1008 (see model_builder.py)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _load_config(parser: argparse.ArgumentParser, config_path: Path) -> None:
    with config_path.open() as f:
        config = yaml.safe_load(f)

    if config is None:
        config = {}
    if not isinstance(config, dict):
        parser.error(f"config file must contain a mapping: {config_path}")

    actions_by_key = {}
    for action in parser._actions:
        actions_by_key[action.dest] = action
        actions_by_key.update(
            (option.removeprefix("--"), action)
            for option in action.option_strings
            if option.startswith("--")
        )

    unknown = sorted(set(config) - set(actions_by_key))
    if unknown:
        parser.error(f"unknown config field(s): {', '.join(unknown)}")

    defaults = {}
    configured_actions = set()
    for key, value in config.items():
        action = actions_by_key[key]
        if action.dest == "config":
            parser.error("the config field cannot set --config")
        if action in configured_actions:
            parser.error(f"config contains multiple fields for {action.dest}")
        configured_actions.add(action)

        if isinstance(action, argparse._StoreTrueAction):
            if not isinstance(value, bool):
                parser.error(f"config field {key} must be a boolean")
        elif value is not None and action.type is not None:
            try:
                value = action.type(value)
            except (TypeError, ValueError) as exc:
                parser.error(f"invalid value for config field {key}: {exc}")

        if action.required and value is None:
            parser.error(f"config field {key} cannot be null")
        action.required = False
        defaults[action.dest] = value

    parser.set_defaults(**defaults)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    config_args, _ = config_parser.parse_known_args(argv)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="YAML file containing argument defaults")
    parser.add_argument("--coco", type=Path, required=True, help="path to canonical GCP coco.json")
    parser.add_argument("--image-root", type=Path, required=True, help="image root the coco file_names are relative to")
    parser.add_argument("--base-checkpoint", type=Path, required=True, help="path to base sam3.pt")
    parser.add_argument("--out-dir", type=Path, required=True, help="where to write splits/checkpoints/metrics.jsonl")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--max-ann-per-img", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--eval-only", action="store_true", help="skip training; just eval --resume (or --base-checkpoint) on val split")
    parser.add_argument("--resume", type=Path, default=None, help="checkpoint to load before training/eval")
    parser.add_argument("--enable-mlflow", action="store_true")
    parser.add_argument("--mlflow-base-uri", default=None, help="MLflow tracking server URI")
    parser.add_argument("--mlflow-experiment-name", default=None)
    parser.add_argument("--mlflow-run-name", default=None, help="defaults to the current timestamp")
    if config_args.config is not None:
        _load_config(parser, config_args.config)
    return parser.parse_args(argv)


def _build_transforms(training: bool, resolution: int):
    """Minimal transform chain that ``Sam3ImageDataset`` can consume.

    Segmentation-only case, no crop/flip augmentation for round 1 to keep the
    initial run's behavior simple to diagnose. Every stage is Meta's, imported
    from the vendored sam3 tree.
    """
    from sam3.train.transforms.basic_for_api import (
        ComposeAPI,
        NormalizeAPI,
        PadToSizeAPI,
        RandomResizeAPI,
        ToTensorAPI,
    )
    from sam3.train.transforms.filter_query_transforms import (
        FilterEmptyTargets,
        FlexibleFilterFindGetQueries,
    )
    from sam3.train.transforms.segmentation import DecodeRle

    return [
        ComposeAPI(
            transforms=[
                DecodeRle(),
                RandomResizeAPI(
                    sizes=[resolution],
                    max_size=resolution,
                    square=True,
                    consistent_transform=True,
                ),
                PadToSizeAPI(size=resolution, consistent_transform=True, bottom_right=True),
                ToTensorAPI(),
                NormalizeAPI(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
            ]
        ),
        FlexibleFilterFindGetQueries(query_filter=FilterEmptyTargets()),
    ]


def _build_dataset(coco_path: Path, image_root: Path, resolution: int, training: bool, max_ann_per_img: int):
    from sam3.train.data.sam3_image_dataset import Sam3ImageDataset

    return Sam3ImageDataset(
        img_folder=str(image_root),
        ann_file=str(coco_path),
        transforms=_build_transforms(training=training, resolution=resolution),
        max_ann_per_img=max_ann_per_img,
        multiplier=1,
        training=training,
        load_segmentation=True,
        max_train_queries=len(COCO_CATEGORY_NAMES),
        max_val_queries=len(COCO_CATEGORY_NAMES),
        use_caching=False,
    )


def _build_dataloader(dataset, batch_size: int, num_workers: int, dict_key: str):
    from sam3.train.data.collator import collate_fn_api

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=isinstance(dataset, torch.utils.data.Dataset) and getattr(dataset, "training", False),
        num_workers=num_workers,
        collate_fn=partial(collate_fn_api, dict_key=dict_key, with_seg_masks=True),
        drop_last=False,
        pin_memory=True,
    )


def _build_model(base_checkpoint: Path, device: str):
    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model(
        checkpoint_path=str(base_checkpoint),
        device=device,
        load_from_HF=False,
    )
    return model


def _freeze_encoders(model: torch.nn.Module) -> tuple[int, int]:
    total = 0
    trainable = 0
    for name, p in model.named_parameters():
        total += p.numel()
        if name.startswith(FROZEN_PREFIXES):
            p.requires_grad_(False)
        else:
            p.requires_grad_(True)
            trainable += p.numel()

    # The ViTDet backbone's fused MLP kernel (sam3/perflib/fused.py) refuses to
    # run when grad is globally enabled -- it only works in inference. Since we
    # freeze the whole backbone, wrapping its two entry points in no_grad is
    # equivalent to `requires_grad=False` for gradient flow and lets the fused
    # path run.
    backbone = model.backbone
    orig_forward_image = backbone.forward_image
    orig_forward_text = backbone.forward_text

    def _forward_image_no_grad(*args, **kwargs):
        with torch.no_grad():
            return orig_forward_image(*args, **kwargs)

    def _forward_text_no_grad(*args, **kwargs):
        with torch.no_grad():
            return orig_forward_text(*args, **kwargs)

    backbone.forward_image = _forward_image_no_grad
    backbone.forward_text = _forward_text_no_grad

    return total, trainable


def _build_matcher():
    from sam3.train.matcher import BinaryHungarianMatcherV2

    return BinaryHungarianMatcherV2(focal=True, cost_class=2.0, cost_bbox=5.0, cost_giou=2.0)


def _disable_triton_focal_loss() -> None:
    """Force PyTorch fallback for Meta's sigmoid_focal_loss.

    The Triton kernel in ``sigmoid_focal_loss.py`` segfaults during compile on
    our H200/Blackwell driver stack. The function has a ``triton=False`` branch
    that runs pure PyTorch; we bind the default to False so every existing call
    site (``Boxes``, ``IABCEMdetr``, ``Masks``) uses it.
    """
    import functools

    from sam3.train.loss import loss_fns

    original = loss_fns.sigmoid_focal_loss
    if getattr(original, "_triton_disabled", False):
        return

    @functools.wraps(original)
    def _no_triton(*args, triton: bool = False, **kwargs):
        return original(*args, triton=False, **kwargs)

    _no_triton._triton_disabled = True  # type: ignore[attr-defined]
    loss_fns.sigmoid_focal_loss = _no_triton


def _build_loss(device: str, matcher):
    _disable_triton_focal_loss()
    from sam3.train.loss.loss_fns import Boxes, IABCEMdetr, Masks
    from sam3.train.loss.sam3_loss import Sam3LossWrapper
    from sam3.train.matcher import BinaryOneToManyMatcher

    o2m_matcher = BinaryOneToManyMatcher(alpha=0.3, threshold=0.4, topk=4)
    loss_fns = [
        Boxes(weight_dict={"loss_bbox": 5.0, "loss_giou": 2.0}, compute_aux=True),
        IABCEMdetr(
            weight_dict={"loss_ce": 20.0, "presence_loss": 20.0},
            compute_aux=True,
            pos_weight=10.0,
            alpha=0.25,
            gamma=2,
            weak_loss=False,
            use_presence=True,
            pos_focal=False,
            pad_n_queries=200,
        ),
        Masks(
            weight_dict={"loss_mask": 200.0, "loss_dice": 10.0},
            compute_aux=False,
            focal_alpha=0.25,
            focal_gamma=2.0,
            num_sample_points=None,  # full-loss path — avoids detectron2 point_sample CUDA kernel
        ),
    ]
    wrapper = Sam3LossWrapper(
        loss_fns_find=loss_fns,
        matcher=matcher,
        o2m_matcher=o2m_matcher,
        o2m_weight=2.0,
        use_o2m_matcher_on_o2m_aux=False,
        normalization="none",
        normalize_by_valid_object_num=True,
        normalize_by_stage_num=False,
    )
    return wrapper.to(device)


def _to_device(batch, device):
    """Move BatchedDatapoint / nested containers of tensors to ``device``."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, list):
        return [_to_device(x, device) for x in batch]
    if isinstance(batch, tuple):
        return tuple(_to_device(x, device) for x in batch)
    if isinstance(batch, dict):
        return {k: _to_device(v, device) for k, v in batch.items()}
    if hasattr(batch, "__dict__"):
        for name in list(vars(batch)):
            setattr(batch, name, _to_device(getattr(batch, name), device))
        return batch
    return batch


KEY_COMPONENTS = ("loss_bbox", "loss_giou", "loss_ce", "presence_loss", "loss_mask", "loss_dice")


def _forward_step(model, batch_dict, loss_wrapper, device):
    """One forward + loss step. Returns (core_loss_tensor, key_component_dict).

    ``key_component_dict`` is only the top-level (non-aux, non-o2m) loss
    components. Aux and o2m variants exist but would drown metrics.jsonl.
    """
    from sam3.train.loss.loss_fns import CORE_LOSS_KEY

    ((_key, batch),) = batch_dict.items()
    batch = _to_device(batch, device)
    find_stages = model(batch)
    find_targets = [model.back_convert(x) for x in batch.find_targets]
    loss_dict = loss_wrapper(find_stages, find_targets)
    components = {
        k: float(loss_dict[k].detach().cpu())
        for k in KEY_COMPONENTS
        if k in loss_dict and isinstance(loss_dict[k], torch.Tensor)
    }
    core = loss_dict[CORE_LOSS_KEY]
    return core, components


def train_one_epoch(model, loader, optimizer, loss_wrapper, device, autocast_dtype):
    model.train()
    running = 0.0
    n = 0
    for step, batch_dict in enumerate(loader):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device, dtype=autocast_dtype):
            core, _ = _forward_step(model, batch_dict, loss_wrapper, device)
        core.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=1.0,
        )
        optimizer.step()
        running += float(core.detach().cpu())
        n += 1
    return running / max(n, 1)


@torch.no_grad()
def validate(model, loader, loss_wrapper, device, autocast_dtype):
    # Keep model in train mode so its Hungarian matcher runs inside forward and
    # populates out["indices"] for Sam3LossWrapper. no_grad prevents updates.
    # The frozen encoders don't contain BN/Dropout that would misbehave here.
    model.train()
    running = 0.0
    n = 0
    for batch_dict in loader:
        with torch.autocast(device_type=device, dtype=autocast_dtype):
            core, _ = _forward_step(model, batch_dict, loss_wrapper, device)
        running += float(core.detach().cpu())
        n += 1
    return running / max(n, 1)


@torch.no_grad()
def per_category_iou(
    model: torch.nn.Module,
    val_coco_path: Path,
    image_root: Path,
    device: str,
    confidence_threshold: float = 0.5,
) -> dict[str, float]:
    """Post-hoc per-category IoU using the same processor path production uses.

    Uses ``Sam3Processor`` (the inference wrapper) rather than the training
    forward so the number we report matches how the model behaves in the
    deployed server.
    """
    from sam3.model.sam3_image_processor import Sam3Processor

    # Sam3Processor drives the deployed inference path — needs the model in eval
    # mode. validate() leaves it in train mode so its Hungarian matcher runs;
    # flip it back here.
    model.eval()
    processor = Sam3Processor(model, device=device, confidence_threshold=confidence_threshold)

    per_cat_iou: dict[str, list[float]] = {name: [] for name in COCO_CATEGORY_NAMES}
    dataset = GcpCocoDataset(
        coco_path=val_coco_path,
        image_root=image_root,
        categories=list(COCO_CATEGORY_NAMES),
        split="train",  # whole file — we already split externally
        val_fraction=0.0,
    )
    for sample in dataset:
        image = sample.load_image()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            state = processor.set_image(image)
            state = processor.set_text_prompt(state=state, prompt=sample.category_name)
        pred_masks = state["masks"].detach().cpu().numpy()  # [N, 1, H, W]
        pred_bin = (pred_masks.squeeze(1) > 0).astype(np.uint8) if pred_masks.ndim == 4 else (pred_masks > 0).astype(np.uint8)
        gt_union = np.zeros((sample.image_height, sample.image_width), dtype=np.uint8)
        for inst in sample.instances:
            gt_union |= inst.mask
        pred_union = np.zeros_like(gt_union)
        for m in pred_bin:
            if m.shape != gt_union.shape:
                # simple nearest-neighbor rescale via PIL
                from PIL import Image as PILImage
                m_img = PILImage.fromarray(m * 255).resize(
                    (sample.image_width, sample.image_height), resample=PILImage.NEAREST
                )
                m = (np.array(m_img) > 0).astype(np.uint8)
            pred_union |= m
        inter = int(np.logical_and(pred_union, gt_union).sum())
        union = int(np.logical_or(pred_union, gt_union).sum())
        iou = inter / union if union > 0 else 0.0
        per_cat_iou[sample.category_name].append(iou)

    return {
        name: float(np.mean(scores)) if scores else 0.0
        for name, scores in per_cat_iou.items()
    }


def _write_split_files(coco_path: Path, out_dir: Path, val_fraction: float) -> tuple[Path, Path, dict[str, Any]]:
    coco = json.loads(coco_path.read_text())
    train_coco, val_coco = split_coco_by_image_id(coco, val_fraction)
    train_path = out_dir / "train.coco.json"
    val_path = out_dir / "val.coco.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path.write_text(json.dumps(train_coco))
    val_path.write_text(json.dumps(val_coco))
    summary = {
        "train_images": len(train_coco["images"]),
        "train_annotations": len(train_coco["annotations"]),
        "val_images": len(val_coco["images"]),
        "val_annotations": len(val_coco["annotations"]),
        "val_fraction": val_fraction,
    }
    return train_path, val_path, summary


def _append_metrics(out_dir: Path, entry: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "metrics.jsonl").open("a") as f:
        f.write(json.dumps(entry) + "\n")


@contextmanager
def _mlflow_run(args: argparse.Namespace):
    if not args.enable_mlflow:
        yield
        return

    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError(
            "MLflow is enabled but not installed; run `uv sync --extra mlflow`"
        ) from exc

    if args.mlflow_base_uri is not None:
        mlflow.set_tracking_uri(args.mlflow_base_uri)
    if args.mlflow_experiment_name is not None:
        mlflow.set_experiment(args.mlflow_experiment_name)

    run_name = args.mlflow_run_name or datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    settings = {key: str(value) for key, value in vars(args).items()}
    settings["mlflow_run_name"] = run_name
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(settings)
        yield


def _train(args: argparse.Namespace, log: logging.Logger) -> None:

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    autocast_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_coco_path, val_coco_path, split_summary = _write_split_files(
        args.coco, args.out_dir, args.val_fraction
    )
    log.info("split summary: %s", split_summary)

    log.info("loading base checkpoint: %s", args.base_checkpoint)
    model = _build_model(args.base_checkpoint, device)
    if args.resume is not None:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"] if "model" in state else state, strict=False)
        log.info("resumed from %s", args.resume)

    total, trainable = _freeze_encoders(model)
    log.info("params: total=%d trainable=%d (%.2f%%)", total, trainable, 100.0 * trainable / max(total, 1))

    matcher = _build_matcher()
    model.matcher = matcher  # Sam3Image.forward_grounding calls self.matcher directly.

    val_dataset = _build_dataset(val_coco_path, args.image_root, args.resolution, training=False, max_ann_per_img=args.max_ann_per_img)
    val_loader = _build_dataloader(val_dataset, args.batch_size, args.num_workers, dict_key="gcp")

    loss_wrapper = _build_loss(device, matcher)

    if args.eval_only:
        val_loss = validate(model, val_loader, loss_wrapper, device, autocast_dtype)
        ious = per_category_iou(model, val_coco_path, args.image_root, device)
        entry = {"mode": "eval_only", "val_loss": val_loss, "per_category_iou": ious}
        log.info("eval: %s", entry)
        _append_metrics(args.out_dir, entry)
        return

    train_dataset = _build_dataset(train_coco_path, args.image_root, args.resolution, training=True, max_ann_per_img=args.max_ann_per_img)
    train_loader = _build_dataloader(train_dataset, args.batch_size, args.num_workers, dict_key="gcp")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val = float("inf")
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_wrapper, device, autocast_dtype)
        val_loss = validate(model, val_loader, loss_wrapper, device, autocast_dtype)
        elapsed = time.time() - t0
        entry = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "elapsed_sec": elapsed}
        log.info("epoch %d/%d train=%.4f val=%.4f (%.1fs)", epoch + 1, args.epochs, train_loss, val_loss, elapsed)
        _append_metrics(args.out_dir, entry)

        torch.save({"model": model.state_dict(), "epoch": epoch}, args.out_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss}, args.out_dir / "best.pt")
            log.info("  saved best.pt (val_loss=%.4f)", val_loss)

    log.info("final per-category IoU eval on val split...")
    ious = per_category_iou(model, val_coco_path, args.image_root, device)
    _append_metrics(args.out_dir, {"mode": "final_iou", "per_category_iou": ious})
    log.info("final IoU: %s", ious)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("train")
    with _mlflow_run(args):
        _train(args, log)


if __name__ == "__main__":
    main()
