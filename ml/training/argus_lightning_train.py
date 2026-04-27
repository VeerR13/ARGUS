"""
ARGUS — YOLO12x Training Script for Lightning AI
Run inside a Lightning AI Studio terminal with:
    tmux new -s argus
    python argus_lightning_train.py
    Ctrl+B, D  (detach — close browser safely)
"""

import os, glob, json, shutil, random
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter

# ── Paths (Lightning AI Studio) ───────────────────────────────────────────────
BASE      = Path('/teamspace/studios/this_studio')
DATA_DIR  = BASE / 'data'
BDD100K_DIR = DATA_DIR / 'bdd100k'
IDD_DIR     = DATA_DIR / 'idd' / 'IDD_Detection'
OUT_DIR   = BASE / 'argus_dataset'
MODEL_OUT = BASE / 'models'

# ── Training config ───────────────────────────────────────────────────────────
EPOCHS  = 60
BATCH   = 8       # T4 16GB single GPU — use 16 if you have A10G/A100
IMGSZ   = 640
WORKERS = 4
MODEL   = 'yolo12x.pt'

BDD_MAX_TRAIN = 15_000
BDD_MAX_VAL   =  2_500
MOTO_MULT     = 2

CLASS_NAMES = ['car', 'motorcycle', 'bus', 'truck', 'bicycle']

IDD_MAP = {
    'car': 0, 'taxi': 0, 'van': 0, 'jeep': 0,
    'motorcycle': 1, 'scooter': 1, 'moped': 1,
    'bus': 2, 'minibus': 2,
    'truck': 3, 'pickup': 3, 'trailer': 3, 'tipper': 3,
    'bicycle': 4,
    'autorickshaw': 0, 'auto-rickshaw': 0, 'e-rickshaw': 0,
}

random.seed(42)

for split in ('train', 'valid'):
    (OUT_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
    (OUT_DIR / split / 'labels').mkdir(parents=True, exist_ok=True)
MODEL_OUT.mkdir(parents=True, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def xyxy2yolo(x1, y1, x2, y2, W, H, cls):
    cx = max(0.001, min(0.999, (x1+x2)/2/W))
    cy = max(0.001, min(0.999, (y1+y2)/2/H))
    w  = max(0.001, min(0.999, (x2-x1)/W))
    h  = max(0.001, min(0.999, (y2-y1)/H))
    return f'{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}'


# ── BDD100K — class-remapped ──────────────────────────────────────────────────
def _discover_bdd_classes(bdd_root):
    import yaml as _yaml
    for yf in sorted(glob.glob(f'{bdd_root}/**/*.yaml', recursive=True)):
        try:
            d = _yaml.safe_load(open(yf))
            if isinstance(d, dict) and 'names' in d:
                names = d['names']
                if isinstance(names, list) and any('car' in n.lower() for n in names):
                    return {n.lower().strip(): i for i, n in enumerate(names)}, names
        except Exception:
            pass
    return None, None

bdd_name_to_id, bdd_class_list = _discover_bdd_classes(BDD100K_DIR)
if bdd_class_list:
    print(f'BDD100K classes auto-detected: {bdd_class_list}')
else:
    bdd_name_to_id = {'car': 2, 'truck': 3, 'bus': 4, 'motorcycle': 6, 'bicycle': 0}
    print(f'BDD100K: yaml not found — using hardcoded fallback: {bdd_name_to_id}')

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


def remap_bdd_split(src_img_dir, src_lbl_dir, dst_img, dst_lbl, max_n, prefix='bdd_'):
    os.makedirs(dst_img, exist_ok=True)
    os.makedirs(dst_lbl, exist_ok=True)
    imgs = [f for f in os.listdir(src_img_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
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
                bdd_cls = int(parts[0])
            except ValueError:
                continue
            argus_cls = BDD_TO_ARGUS.get(bdd_cls)
            if argus_cls is not None:
                lines.append(f'{argus_cls} {" ".join(parts[1:])}')
                stats[CLASS_NAMES[argus_cls]] += 1
        if not lines:
            continue
        shutil.copy(os.path.join(src_img_dir, img_f), os.path.join(dst_img, prefix + img_f))
        with open(os.path.join(dst_lbl, prefix + stem + '.txt'), 'w') as wf:
            wf.write('\n'.join(lines) + '\n')
        kept += 1
    print(f'  kept={kept:,}  classes: {dict(stats)}')
    return kept


def _find_bdd_dirs(split):
    candidates = [
        (f'{BDD100K_DIR}/{split}/images', f'{BDD100K_DIR}/{split}/labels'),
        (f'{BDD100K_DIR}/images/{split}', f'{BDD100K_DIR}/labels/{split}'),
        (f'{BDD100K_DIR}/{split}',        f'{BDD100K_DIR}/{split}'),
    ]
    for si, sl in candidates:
        if os.path.isdir(si):
            return si, sl
    return None, None


print('\n--- BDD100K ---')
bdd_train_img, bdd_train_lbl = _find_bdd_dirs('train')
bdd_val_img,   bdd_val_lbl   = _find_bdd_dirs('val')

n_bdd_train = 0
if bdd_train_img:
    print(f'BDD100K train ({BDD_MAX_TRAIN:,} cap):')
    n_bdd_train = remap_bdd_split(
        bdd_train_img, bdd_train_lbl,
        OUT_DIR / 'train' / 'images', OUT_DIR / 'train' / 'labels', BDD_MAX_TRAIN)
else:
    print('BDD100K train: not found')

n_bdd_val = 0
if bdd_val_img:
    print(f'BDD100K val ({BDD_MAX_VAL:,} cap):')
    n_bdd_val = remap_bdd_split(
        bdd_val_img, bdd_val_lbl,
        OUT_DIR / 'valid' / 'images', OUT_DIR / 'valid' / 'labels', BDD_MAX_VAL)
else:
    print('BDD100K val: not found')

_bad = sum(
    1 for lbl in glob.glob(str(OUT_DIR / 'train' / 'labels' / 'bdd_*.txt'))
    for row in open(lbl).read().splitlines()
    if row.strip() and int(row.split()[0]) >= len(CLASS_NAMES)
)
print(f'Sanity: {_bad} out-of-range labels in BDD train (should be 0)')


# ── IDD ───────────────────────────────────────────────────────────────────────
def parse_idd_xml(xml_path):
    import cv2
    root = ET.parse(xml_path).getroot()
    size = root.find('size')
    W = int(size.find('width').text)  if size is not None else 0
    H = int(size.find('height').text) if size is not None else 0
    lines = []
    for obj in root.findall('object'):
        name_el = obj.find('name')
        if name_el is None: continue
        cls = IDD_MAP.get(name_el.text.strip().lower())
        if cls is None: continue
        b = obj.find('bndbox')
        if b is None: continue
        x1,y1,x2,y2 = (float(b.find(k).text) for k in ('xmin','ymin','xmax','ymax'))
        if x2<=x1 or y2<=y1 or W<=0 or H<=0: continue
        lines.append(xyxy2yolo(x1,y1,x2,y2,W,H,cls))
    return W, H, lines

def process_idd(split):
    import cv2
    out_split = 'valid' if split == 'val' else split
    out_img = OUT_DIR / out_split / 'images'
    out_lbl = OUT_DIR / out_split / 'labels'
    ann_root = os.path.join(IDD_DIR, 'Annotations', split)
    img_root = os.path.join(IDD_DIR, 'JPEGImages', split)
    if not os.path.isdir(ann_root):
        print(f'IDD {split}: not found — skipping'); return 0
    n = 0
    for seq in sorted(os.listdir(ann_root)):
        seq_ann = os.path.join(ann_root, seq)
        seq_img = os.path.join(img_root, seq)
        if not os.path.isdir(seq_ann): continue
        for xml_file in os.listdir(seq_ann):
            if not xml_file.endswith('.xml'): continue
            stem = os.path.splitext(xml_file)[0]
            W, H, lines = parse_idd_xml(os.path.join(seq_ann, xml_file))
            if not lines: continue
            img_path = None
            for ext in ('.jpg','.jpeg','.png'):
                c = os.path.join(seq_img, stem+ext)
                if os.path.exists(c): img_path=c; break
            if img_path is None: continue
            if W==0 or H==0:
                img = cv2.imread(img_path)
                if img is None: continue
                H, W = img.shape[:2]
                _, _, lines = parse_idd_xml(os.path.join(seq_ann, xml_file))
                if not lines: continue
            out_stem = f'idd_{seq}_{stem}'
            dst_img = out_img / (out_stem + '.jpg')
            if img_path.endswith(('.jpg','.jpeg')):
                shutil.copy(img_path, dst_img)
            else:
                cv2.imwrite(str(dst_img), cv2.imread(img_path), [cv2.IMWRITE_JPEG_QUALITY,90])
            (out_lbl / (out_stem + '.txt')).write_text('\n'.join(lines)+'\n')
            n += 1
    print(f'IDD {split}: {n} images')
    return n

print('\n--- IDD ---')
n_idd_train = process_idd('train')
n_idd_val   = process_idd('val')


# ── Motorcycle resampling ─────────────────────────────────────────────────────
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
print(f'\nMotorcycle copies added: {n_moto:,}')

print(f'\nDataset summary:')
print(f'  Train: BDD={n_bdd_train} + IDD={n_idd_train} + moto_resamp={n_moto} = {n_bdd_train+n_idd_train+n_moto}')
print(f'  Val:   BDD={n_bdd_val}   + IDD={n_idd_val}   = {n_bdd_val+n_idd_val}')


# ── data.yaml ─────────────────────────────────────────────────────────────────
yaml_path = OUT_DIR / 'data.yaml'
yaml_path.write_text(f"""path: {OUT_DIR.resolve()}
train: train/images
val:   valid/images

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
""")
print(f'\ndata.yaml written to {yaml_path}')


# ── Train ─────────────────────────────────────────────────────────────────────
from ultralytics import YOLO

model = YOLO(MODEL)

results = model.train(
    data    = str(yaml_path),
    epochs  = EPOCHS,
    batch   = BATCH,
    imgsz   = IMGSZ,
    device  = '0',          # single GPU on Lightning AI free tier
    workers = WORKERS,
    project = str(MODEL_OUT),
    name    = 'yolo12x_bdd_idd',
    exist_ok= True,
    optimizer     = 'AdamW',
    lr0           = 1e-3,
    lrf           = 0.01,
    weight_decay  = 5e-4,
    warmup_epochs = 3,
    hsv_h   = 0.015,
    hsv_s   = 0.7,
    hsv_v   = 0.4,
    degrees = 0.0,
    translate = 0.1,
    scale     = 0.5,
    fliplr    = 0.5,
    mosaic    = 1.0,
    mixup     = 0.1,
    patience    = 15,
    save        = True,
    save_period = 5,
    val         = True,
    plots       = True,
    verbose     = True,
)


# ── Eval + save ───────────────────────────────────────────────────────────────
best_pt  = MODEL_OUT / 'yolo12x_bdd_idd' / 'weights' / 'best.pt'
final_pt = BASE / 'argus_yolo12x_best.pt'
shutil.copy(best_pt, final_pt)
print(f'\nSaved: {final_pt}  ({final_pt.stat().st_size/1e6:.1f} MB)')

metrics = model.val(data=str(yaml_path), imgsz=IMGSZ, verbose=True)
print(f'\nmAP50:     {metrics.box.map50:.4f}')
print(f'mAP50-95:  {metrics.box.map:.4f}')
print(f'Precision: {metrics.box.mp:.4f}')
print(f'Recall:    {metrics.box.mr:.4f}')
if hasattr(metrics.box, 'ap_class_index'):
    for i, cls_idx in enumerate(metrics.box.ap_class_index):
        print(f'  {CLASS_NAMES[cls_idx]:12s} AP50={metrics.box.ap50[i]:.3f}')

model.export(format='onnx', imgsz=IMGSZ, simplify=True)
print('\nONNX exported.')
print('Done — download argus_yolo12x_best.pt from the Studio file browser.')
