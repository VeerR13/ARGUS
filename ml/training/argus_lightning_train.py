"""
ARGUS — YOLO12x Training Script for Lightning AI
Run inside a Lightning AI Studio terminal:
    tmux new -s argus
    python3 argus_lightning_train.py
    Ctrl+B, D  (detach — safe to close browser)

Resume after interruption: re-run the same command — last.pt is auto-detected.
"""

import os, glob, shutil, random, subprocess
from pathlib import Path
from collections import Counter

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = Path('/teamspace/studios/this_studio')
BDD_DIR   = BASE / 'data' / 'bdd100k'
OUT_DIR   = BASE / 'argus_dataset'
MODEL_OUT = BASE / 'models'

# ── Training config ───────────────────────────────────────────────────────────
EPOCHS  = 45
BATCH   = 8
IMGSZ   = 640
WORKERS = 4
MODEL   = 'yolo12x.pt'

# 35k BDD + ×3 moto resamp ≈ 37k images → ~4,662 steps/epoch
# 45 epochs × ~77 min/epoch ≈ 58 hrs — fits within 62-hr credit budget
BDD_MAX_TRAIN = 35_000
BDD_MAX_VAL   = 10_000
MOTO_MULT     = 3

CLASS_NAMES = ['car', 'motorcycle', 'bus', 'truck', 'bicycle']

random.seed(42)

for split in ('train', 'valid'):
    (OUT_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
    (OUT_DIR / split / 'labels').mkdir(parents=True, exist_ok=True)
MODEL_OUT.mkdir(parents=True, exist_ok=True)

# ── Fast-path: skip dataset assembly if already built ─────────────────────────
n_existing = sum(1 for _ in (OUT_DIR / 'train' / 'images').glob('*'))
dataset_ready = n_existing >= 35_000
if dataset_ready:
    print(f'Dataset already built ({n_existing:,} train images). Skipping copy steps.')


# ── Download extra datasets via Kaggle ───────────────────────────────────────
def kaggle_download(ref, dest):
    dest = Path(dest)
    if dest.exists() and any(dest.rglob('*.jpg')):
        print(f'  {ref}: already present, skipping download')
        return
    dest.mkdir(parents=True, exist_ok=True)
    print(f'  Downloading {ref}...')
    subprocess.run(
        ['kaggle', 'datasets', 'download', '-d', ref, '-p', str(dest), '--unzip'],
        check=True
    )

if not dataset_ready:
    print('=== Downloading extra datasets ===')
    kaggle_download('dataclusterlabs/indian-vehicle-dataset',
                    BASE / 'data' / 'indian_vehicles')


# ── BDD class mapping ─────────────────────────────────────────────────────────
def discover_bdd_classes(bdd_root):
    import yaml
    for yf in sorted(glob.glob(f'{bdd_root}/**/*.yaml', recursive=True)):
        try:
            d = yaml.safe_load(open(yf))
            if isinstance(d, dict) and 'names' in d:
                names = d['names']
                if isinstance(names, list) and any('car' in n.lower() for n in names):
                    return {n.lower().strip(): i for i, n in enumerate(names)}, names
        except Exception:
            pass
    return None, None

bdd_name_to_id, bdd_class_list = discover_bdd_classes(BDD_DIR)
if bdd_class_list:
    print(f'BDD classes: {bdd_class_list}')
else:
    bdd_name_to_id = {'car': 2, 'truck': 3, 'bus': 4, 'motorcycle': 6, 'bicycle': 0}
    print(f'BDD: yaml not found — using hardcoded fallback')

_SYNONYMS = {
    'car':        ['car', 'vehicle'],
    'motorcycle': ['motorcycle', 'motor', 'motorbike'],
    'bus':        ['bus', 'minibus'],
    'truck':      ['truck', 'lorry'],
    'bicycle':    ['bicycle', 'bike'],
}
BDD_TO_ARGUS = {}
for our_id, our_name in enumerate(CLASS_NAMES):
    for syn in _SYNONYMS.get(our_name, [our_name]):
        if syn in bdd_name_to_id:
            BDD_TO_ARGUS[bdd_name_to_id[syn]] = our_id
            break
print(f'BDD_TO_ARGUS: {BDD_TO_ARGUS}')


# ── Generic YOLO-format copy with class remapping ────────────────────────────
def copy_yolo_split(src_img_dir, src_lbl_dir, dst_img, dst_lbl,
                    class_remap, max_n, prefix):
    os.makedirs(dst_img, exist_ok=True)
    os.makedirs(dst_lbl, exist_ok=True)
    imgs = [f for f in os.listdir(src_img_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(imgs)
    kept, stats = 0, Counter()
    for img_f in imgs:
        if kept >= max_n:
            break
        stem = os.path.splitext(img_f)[0]
        lbl_src = os.path.join(src_lbl_dir, stem + '.txt')
        if not os.path.exists(lbl_src):
            continue
        lines = []
        for row in open(lbl_src).read().strip().splitlines():
            parts = row.split()
            if not parts:
                continue
            try:
                src_cls = int(parts[0])
            except ValueError:
                continue
            dst_cls = class_remap.get(src_cls)
            if dst_cls is not None:
                lines.append(f'{dst_cls} {" ".join(parts[1:])}')
                stats[CLASS_NAMES[dst_cls]] += 1
        if not lines:
            continue
        shutil.copy(os.path.join(src_img_dir, img_f),
                    os.path.join(dst_img, prefix + img_f))
        with open(os.path.join(dst_lbl, prefix + stem + '.txt'), 'w') as wf:
            wf.write('\n'.join(lines) + '\n')
        kept += 1
    print(f'  kept={kept:,}  {dict(stats)}')
    return kept


def find_split_dirs(dataset_root, split):
    """Try common YOLO directory layouts."""
    candidates = [
        (f'{dataset_root}/{split}/images', f'{dataset_root}/{split}/labels'),
        (f'{dataset_root}/images/{split}', f'{dataset_root}/labels/{split}'),
        (f'{dataset_root}/{split}',         f'{dataset_root}/{split}'),
    ]
    for si, sl in candidates:
        if os.path.isdir(si) and os.path.isdir(sl):
            return si, sl
    return None, None


# ── BDD100K (70k, remapped) ───────────────────────────────────────────────────
n_bdd_train = n_bdd_val = 0

if not dataset_ready:
    print('\n=== BDD100K (full 70k) ===')
    bdd_tr_img, bdd_tr_lbl = find_split_dirs(BDD_DIR, 'train')
    bdd_va_img, bdd_va_lbl = find_split_dirs(BDD_DIR, 'val')

    if bdd_tr_img:
        print(f'Train (cap {BDD_MAX_TRAIN:,}):')
        n_bdd_train = copy_yolo_split(
            bdd_tr_img, bdd_tr_lbl,
            OUT_DIR/'train'/'images', OUT_DIR/'train'/'labels',
            BDD_TO_ARGUS, BDD_MAX_TRAIN, 'bdd_')
    else:
        print('BDD train not found')

    if bdd_va_img:
        print(f'Val (cap {BDD_MAX_VAL:,}):')
        n_bdd_val = copy_yolo_split(
            bdd_va_img, bdd_va_lbl,
            OUT_DIR/'valid'/'images', OUT_DIR/'valid'/'labels',
            BDD_TO_ARGUS, BDD_MAX_VAL, 'bdd_')
    else:
        print('BDD val not found')

    bad = sum(
        1 for lbl in glob.glob(str(OUT_DIR/'train'/'labels'/'bdd_*.txt'))
        for row in open(lbl).read().splitlines()
        if row.strip() and int(row.split()[0]) >= len(CLASS_NAMES)
    )
    print(f'Sanity — out-of-range BDD labels: {bad} (must be 0)')

n_b5_train = n_b5_val = 0


# ── Indian Vehicle Dataset ────────────────────────────────────────────────────
n_ivd = 0

if not dataset_ready:
    print('\n=== Indian Vehicle Dataset ===')
    ivd_root = BASE / 'data' / 'indian_vehicles'

    def process_flat_yolo(dataset_root, dst_img, dst_lbl, class_remap, prefix='ivd_'):
        os.makedirs(dst_img, exist_ok=True)
        os.makedirs(dst_lbl, exist_ok=True)
        all_imgs = list(Path(dataset_root).rglob('*.jpg')) + \
                   list(Path(dataset_root).rglob('*.jpeg')) + \
                   list(Path(dataset_root).rglob('*.png'))
        random.shuffle(all_imgs)
        kept, stats = 0, Counter()
        for img_path in all_imgs:
            stem = img_path.stem
            lbl_candidates = [
                img_path.parent / (stem + '.txt'),
                img_path.parent.parent / 'labels' / (stem + '.txt'),
            ]
            lbl_src = next((p for p in lbl_candidates if p.exists()), None)
            if not lbl_src:
                continue
            lines = []
            for row in open(lbl_src).read().strip().splitlines():
                parts = row.split()
                if not parts:
                    continue
                try:
                    src_cls = int(parts[0])
                except ValueError:
                    continue
                dst_cls = class_remap.get(src_cls)
                if dst_cls is not None:
                    lines.append(f'{dst_cls} {" ".join(parts[1:])}')
                    stats[CLASS_NAMES[dst_cls]] += 1
            if not lines:
                continue
            shutil.copy(str(img_path), os.path.join(dst_img, prefix + img_path.name))
            with open(os.path.join(dst_lbl, prefix + stem + '.txt'), 'w') as wf:
                wf.write('\n'.join(lines) + '\n')
            kept += 1
        print(f'  kept={kept:,}  {dict(stats)}')
        return kept

    ivd_tr_img, ivd_tr_lbl = find_split_dirs(ivd_root, 'train')
    if ivd_tr_img:
        import yaml
        ivd_remap = {}
        for yf in sorted(glob.glob(f'{ivd_root}/**/*.yaml', recursive=True)):
            try:
                d = yaml.safe_load(open(yf))
                if isinstance(d, dict) and 'names' in d:
                    for i, n in enumerate(d['names']):
                        for our_id, our_name in enumerate(CLASS_NAMES):
                            if any(s in n.lower() for s in _SYNONYMS.get(our_name, [our_name])):
                                ivd_remap[i] = our_id
                    break
            except Exception:
                pass
        print(f'  IVD remap: {ivd_remap}')
        print('Train:')
        n_ivd = copy_yolo_split(
            ivd_tr_img, ivd_tr_lbl,
            OUT_DIR/'train'/'images', OUT_DIR/'train'/'labels',
            ivd_remap, 20_000, 'ivd_')
    else:
        print('  No standard split — trying flat layout...')
        ivd_remap = {0: 0, 1: 3, 2: 2, 3: 1, 4: 4}
        n_ivd = process_flat_yolo(ivd_root,
                                  OUT_DIR/'train'/'images',
                                  OUT_DIR/'train'/'labels',
                                  ivd_remap)


# ── Motorcycle resampling ─────────────────────────────────────────────────────
n_moto = 0

if not dataset_ready:
    print('\n=== Motorcycle resampling ===')
    def resample_class(dataset_dir, cls_id, multiplier):
        img_dir = dataset_dir / 'train' / 'images'
        lbl_dir = dataset_dir / 'train' / 'labels'
        added = 0
        for lbl in sorted(lbl_dir.glob('*.txt')):
            if lbl.name.startswith('resamp_'):
                continue
            has_cls = any(
                row.split()[0] == str(cls_id)
                for row in lbl.read_text().splitlines() if row.strip()
            )
            if not has_cls:
                continue
            for ext in ('.jpg', '.jpeg', '.png'):
                img = img_dir / (lbl.stem + ext)
                if img.exists():
                    for k in range(multiplier - 1):
                        stem = f'resamp_{k}_{lbl.stem}'
                        shutil.copy(img, img_dir / (stem + ext))
                        shutil.copy(str(lbl), lbl_dir / (stem + '.txt'))
                        added += 1
                    break
        return added

    n_moto = resample_class(OUT_DIR, cls_id=1, multiplier=MOTO_MULT)
    print(f'Motorcycle copies added: {n_moto:,} (×{MOTO_MULT})')


# ── Summary + data.yaml ───────────────────────────────────────────────────────
if dataset_ready:
    n_train = n_existing
    n_val   = sum(1 for _ in (OUT_DIR / 'valid' / 'images').glob('*'))
else:
    n_train = n_bdd_train + n_b5_train + n_ivd + n_moto
    n_val   = n_bdd_val   + n_b5_val

print(f'\n=== Dataset summary ===')
print(f'Train: {n_train:,}  Val: {n_val:,}')

yaml_path = OUT_DIR / 'data.yaml'
yaml_path.write_text(f"""path: {OUT_DIR.resolve()}
train: train/images
val:   valid/images
nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
""")
print(f'data.yaml written.')


# ── Train (auto-resumes from last.pt if interrupted) ─────────────────────────
from ultralytics import YOLO

last_pt = MODEL_OUT / 'yolo12x_bdd_idd' / 'weights' / 'last.pt'
resuming = last_pt.exists()

if resuming:
    print(f'\nResuming from checkpoint: {last_pt}')
    model = YOLO(str(last_pt))
    results = model.train(resume=True)
else:
    print(f'\nStarting fresh training for {EPOCHS} epochs.')
    model = YOLO(MODEL)
    results = model.train(
        data      = str(yaml_path),
        epochs    = EPOCHS,
        batch     = BATCH,
        imgsz     = IMGSZ,
        device    = '0',
        workers   = WORKERS,
        project   = str(MODEL_OUT),
        name      = 'yolo12x_bdd_idd',
        exist_ok  = True,
        optimizer     = 'AdamW',
        lr0           = 5e-4,
        lrf           = 0.01,
        weight_decay  = 5e-4,
        warmup_epochs = 5,
        hsv_h    = 0.015,
        hsv_s    = 0.7,
        hsv_v    = 0.4,
        degrees  = 0.0,
        translate= 0.1,
        scale    = 0.5,
        fliplr   = 0.5,
        mosaic   = 1.0,
        mixup    = 0.15,
        copy_paste = 0.1,
        patience    = 20,
        save        = True,
        save_period = 5,
        val         = True,
        plots       = True,
        verbose     = True,
    )


# ── Eval + export ─────────────────────────────────────────────────────────────
from ultralytics import YOLO

best_pt  = MODEL_OUT / 'yolo12x_bdd_idd' / 'weights' / 'best.pt'
final_pt = BASE / 'argus_yolo12x_best.pt'
shutil.copy(best_pt, final_pt)
print(f'\nSaved: {final_pt}  ({final_pt.stat().st_size/1e6:.1f} MB)')

model = YOLO(str(best_pt))
metrics = model.val(data=str(yaml_path), imgsz=IMGSZ, verbose=True)
print(f'\nmAP50:     {metrics.box.map50:.4f}')
print(f'mAP50-95:  {metrics.box.map:.4f}')
if hasattr(metrics.box, 'ap_class_index'):
    for i, cls_idx in enumerate(metrics.box.ap_class_index):
        print(f'  {CLASS_NAMES[cls_idx]:12s} AP50={metrics.box.ap50[i]:.3f}')

model.export(format='onnx', imgsz=IMGSZ, simplify=True)
print('ONNX exported. Download argus_yolo12x_best.pt from Studio file browser.')
