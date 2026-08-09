# Thelerine 2.0 — Warp-based Virtual Try-On

A two-stage virtual try-on system built for **small triplet datasets** (tens to a
few hundred samples), where diffusion models have far too little data and plain
image-to-image translation plateaus at "blurry but roughly right".

---

## 1. Why not the approaches you already tried

**Stable Diffusion / any latent diffusion try-on.** These work by fine-tuning a
model that has already seen billions of images. Fine-tuning on ~100 samples
either barely moves the model (it ignores your garment) or destroys it
(catastrophic forgetting). The published small-data try-on diffusion papers all
still use VITON-HD or DressCode — 11k and 50k pairs respectively. There is no
regime where 100 images is enough to teach a diffusion model your garment
distribution.

**Image-to-image translation (pix2pix / UNet regression).** This is the approach
that "worked a bit". The failure is structural, not a matter of tuning:

> The network must reproduce every output pixel from its weights. Garment
> texture, colour and pattern therefore have to be *stored in the network* and
> retrieved from the input. Storing an image distribution takes thousands of
> examples. With a hundred, the L1-optimal solution is to output the conditional
> mean — which is exactly the blurry, colour-averaged result you saw.

Turning up the GAN weight sharpens it but does not fix it, because the
information was never there. Nothing in a pure translation architecture lets the
garment's actual pixels reach the output.

**What this project does instead.** Move the real garment pixels to where they
belong, then learn only the blend:

```
output = alpha · warped_real_garment  +  (1 − alpha) · synthesised
```

Roughly 90% of the output is now *copied* rather than *generated*. What remains
to be learned is (a) a deformation with 50 degrees of freedom, and (b) a soft
mask. Both are low-entropy targets that a small dataset can actually supply.
This is the CP-VTON / CP-VTON+ lineage, modernised with a bounded residual flow,
group normalisation, EMA and DiffAugment.

---

## 2. Architecture

```
                     ┌───────────────────────── STAGE 1: WARP ─────────────────────────┐
 flat garment ──────►│ encoder ─┐                                                       │
   + its mask        │          ├─► correlation ─► regressor ─► TPS grid (50 DOF)       │
                     │ encoder ─┘                       │                               │
 body condition ────►│                                  ▼                               │
   (14 channels)     │                          coarse warp ─► refiner ─► residual flow │
                     └──────────────────────────────────┬──────────────────────────────┘
                                                        ▼
                                              warped garment + mask
                                                        │
                     ┌───────────────────── STAGE 2: COMPOSE ──────┴───────────────────┐
 body condition ────►│  UNet  ──┬──► render  (synthesised fallback)                    │
 warped garment ────►│          └──► alpha   (per-pixel copy/synthesise decision)      │
                     │                                                                  │
                     │        output = alpha·warped + (1−alpha)·render                  │
                     └──────────────────────────────────────────────────────────────────┘
```

Parameter counts at the default config: warper **1.3M**, composer **7.7M**. Both
are small on purpose — capacity is the enemy when data is scarce.

### The body condition (14 channels)

Built in [`vtonwarp/data/agnostic.py`](vtonwarp/data/agnostic.py) from the CIHP
parse map:

| channels | content | why |
|---|---|---|
| 3 | agnostic RGB — person with garment **and arms** erased to grey | anything left behind leaks the answer |
| 3 | preserved identity RGB — face, hair, hat | never ask a 100-image model to redraw a face |
| 1 | body shape, downsampled to 16×12 then blurred back up | pose and build *without* the garment's exact outline |
| 7 | coarse parse groups (background / identity / upper / skin / lower / legs / feet) | structure, without wasting capacity on left-vs-right shoe |

The deliberate lossiness of the shape channel is important. A crisp silhouette
lets the model infer the original garment's outline, so it scores well on
training pairs and collapses when you give it a garment of a different cut.

### Stage 1 — the warper

`vtonwarp/models/warper.py`, `vtonwarp/models/tps.py`

1. Two encoders reduce the garment and the body condition to 16×12 feature maps.
2. A **correlation layer** computes, for every garment location, a heat map of
   how well it matches each body location. Matching is a comparison operation,
   so giving the regressor an explicit similarity volume is much easier to learn
   from than two stacks of features it must first learn to compare.
3. A regressor maps the correlation volume to **25 control-point offsets**,
   which a Thin-Plate Spline turns into a full sampling grid. 50 free parameters
   total — a dense flow field at this resolution would have ~98,000 and would
   memorise the training set outright. The final layer is zero-initialised, so
   training starts from the *identity* warp.
4. A refiner adds a **bounded** dense residual flow (±0.12 in normalised units)
   on top, which is what expresses folds, seams and sleeve creases that no 5×5
   lattice can.

Supervision is free, and this is the crux of the whole design: the CIHP parse
already tells us exactly which pixels of each training image are garment. So we
know both the silhouette the warp must match and the RGB it must produce.

| loss | weight | role |
|---|---|---|
| `shape` | 10.0 | L1 + soft Dice between warped mask and target garment region |
| `coarse_shape` | 2.0 | same, on the TPS-only result — keeps the coarse stage honest |
| `pixel` | 5.0 | masked L1 on warped garment RGB |
| `perceptual` | 1.0 | VGG19 feature L1, restricted to the garment region |
| `grid` | 2.0 | TPS lattice regularity — forbids folding the grid over itself |
| `tv` / `smooth` | 4.0 / 2.0 | first- and second-order flow smoothness |

Shape dominates early because silhouette matching is insensitive to colour and
lighting, so it gives a clean gradient before anything else has been learned.

### Stage 2 — the composer

`vtonwarp/models/composer.py`

A small GroupNorm UNet with two heads on a shared trunk. It emits `render` and
`alpha`, and the output is their blend with the warped garment. `alpha` is
multiplied by the warped mask, so it can only claim pixels where garment
actually exists.

The critical loss is `alpha` (weight 2.0), which pulls the composition mask
towards the warped garment mask. Without it, the network discovers it can set
alpha ≈ 0 and push everything through `render` — which minimises L1 fastest and
reproduces exactly the washed-out result of a plain image-to-image model. That
one term is the difference between this architecture and the one you already
tried.

---

## 3. Small-data machinery

Everything below buys quality without needing more images.

| technique | where | what it buys |
|---|---|---|
| Warp-then-compose | whole design | garment texture becomes an *input*, not something to learn |
| 50-DOF TPS before dense flow | `models/tps.py` | smoothness enforced by parameterisation, not by a loss |
| Zero-init flow heads | `models/blocks.py` | training starts at identity, first gradients are corrections |
| VGG19 perceptual loss | `losses/perceptual.py` | borrows texture priors from ImageNet; kills L1 blur |
| Aggressive **garment-only** jitter | `data/augment.py` | stops the warper collapsing to a near-identity transform |
| Shared photometric jitter | `data/augment.py` | lighting invariance without breaking the pixel targets |
| GroupNorm everywhere | `models/blocks.py` | valid at batch size 2–4, no train/eval statistics mismatch |
| Weight EMA | `engine/ema.py` | largest single quality gain per line of code; always export EMA |
| DiffAugment + spectral norm | `models/discriminator.py` | makes a GAN survivable on ~100 images |
| Delayed GAN start | `configs/tryon.yaml` | a discriminator started at step 0 wins instantly and never lets go |
| Frozen warper in stage 2 | `train_tryon.py` | otherwise the composer degrades the warp to make its own job easier |

---

## 4. Dataset layout

```
<root>/
  person/
    personA/  0001_person.jpg ...
    personB/  0007_person.jpg ...
  garments/
    personA/  0001_garment.jpg ...
    personB/  0007_garment.jpg ...
  cihp/
    personA/  0001_cihp.npy ...        # H×W integer CIHP label map
  segmentation/
    personA/  0001_seg.png ...         # binary garment mask
```

Files are matched across folders by a **normalised stem** — the filename with
role tokens (`_person`, `_garment`, `_cihp`, `_seg`, `_dress`, …) stripped, so
`0001_person.jpg` pairs with `0001_garment_dress.png` automatically. Subject
subfolders are matched first, then globally. See
[`vtonwarp/data/manifest.py`](vtonwarp/data/manifest.py).

`personA` and `personB` are treated as two subjects. Every person is **self-paired**
with the garment they are already wearing, which is what provides pixel-perfect
supervision. At inference you simply feed a different person's condition.

`cihp/` accepts `.npy` (H×W ints, or one-hot/probability maps), indexed PNGs, or
greyscale PNGs with scaled class ids. Colourised RGB parse visualisations are
rejected with an explicit error rather than silently misread — export raw label
indices.

### Label conventions — get this right first

A parse map is only integers; the taxonomy that produced them decides what those
integers mean, and the two common ones disagree precisely where it hurts:

| id | CIHP / LIP (20 classes) | ATR (18 classes) |
|---|---|---|
| 4 | sunglasses | **upper clothes** |
| 5 | **upper clothes** | skirt |
| 6 | **dress** | pants |
| 7 | **coat** | dress |
| 9 | **pants** | left shoe |
| 11 | scarf | **face** |
| 12 | **skirt** | left leg |

Assume CIHP on ATR maps and the model erases the person's face and trousers
instead of their shirt — silently, with no error. Identify yours:

```bash
python scripts/check_dataset.py --root /path/to/dataset --diagnose-labels
```

It scores each convention on colour agreement between garment and parse region,
plus the physical priors that heads are at the top of an image and shoes at the
bottom. Put the winner in `data.label_scheme` in **both** configs.

### Mixed garment types

`data.garment_type: auto` (the default) picks the target region per sample, from
the garment filename (`0007_garment_dress.jpg`) and from colour agreement with
each clothing region. A dataset containing dresses, tops, jeans and skirts needs
this — a single global `garment_type` would train the model to warp trousers
onto a T-shirt's silhouette. Force one region for the whole dataset with
`upper`, `lower` or `full`.

### Garment framing

`data.canonicalise_garment: true` crops each product shot to its mask and
rescales it so every garment enters the network at roughly the same size.
Without it, wildly inconsistent framing (a thumbnail on a large canvas next to a
full-bleed photo) makes the warper learn a large scale change on top of the
body-shape deformation, from very few examples.

The mask for the flat garment is taken from `segmentation/` when that looks
right, otherwise it is segmented from the product shot's background — measured
from the image border, so black, white or any other uniform backdrop works.

**`segmentation/` is ambiguous, so check what yours actually contains.** The code
uses it as the mask of the *flat garment product shot*. If yours instead holds
the garment mask **on the person**, or a full-body silhouette, set
`data.segmentation_role: ignore` — the flat garment's mask is then derived by
thresholding the product shot's white background, which is reliable for
catalogue images. `scripts/check_dataset.py` prints the folder's mean occupancy
and its best guess. Getting this wrong silently corrupts the warp supervision,
so it is worth thirty seconds of checking.

---

## 5. Usage

```bash
conda activate thelerine-vton          # torch 2.2 already installed
cd ~/Projects/Thelerine2.0_vton-warp
```

**Step 0 — always check the data first.** One bad parse map is 1% of your dataset.

```bash
python scripts/check_dataset.py --root /path/to/dataset
```

It reports how many triplets matched, which CIHP classes are actually present,
what fraction of the frame the garment covers, and writes a contact sheet. Look
at the `agnostic` column: **if you can still see the original garment, stop and
fix that before training anything.**

**Step 1 — train the warper to convergence.**

```bash
python train_warp.py --config configs/warp.yaml data.root=/path/to/dataset
```

Watch `outputs/warp/samples/*.png`. The `overlay` column is the one that matters:
the garment should sit on the body in the right place and shape. Do not proceed
until it does — a composer trained on a bad warp learns to ignore the warp, and
never unlearns it.

**Step 2 — train the composer.**

```bash
python train_tryon.py --config configs/tryon.yaml data.root=/path/to/dataset
```

Once it converges, optionally switch the GAN on for a sharpening pass:

```bash
python train_tryon.py --config configs/tryon.yaml data.root=/path/to/dataset \
    loss.gan=0.5 train.gan_start_step=0 train.steps=4000 train.lr=0.00005
```

**Step 3 — inference.**

```bash
# one pair
python infer.py --checkpoint outputs/tryon/tryon.pt \
    --person  data/person/personA/0001_person.jpg \
    --parse   data/cihp/personA/0001_cihp.npy \
    --garment data/garments/personB/0007_garment.jpg \
    --out results/0001_wearing_0007.png

# every garment on every person — the honest evaluation
python infer.py --checkpoint outputs/tryon/tryon.pt --grid --out results/grid.png
```

Any config value can be overridden on the command line: `train.steps=20000`,
`loss.alpha=3.0`, `data.height=320`.

### Training on Colab

Open [`notebooks/colab_train.ipynb`](notebooks/colab_train.ipynb) in Colab. It
mounts Drive, detects your dataset's folder names, runs the sanity check, and
trains both stages with checkpoints written straight to Drive. Both training
cells **resume automatically**, so a disconnect costs nothing — just re-run from
the top. Set `train.resume=false` to start clean.

### Smoke test without real data

```bash
python scripts/make_dummy_dataset.py --out data/dummy --count 24
python train_warp.py --config configs/warp.yaml data.root=data/dummy train.steps=200
```

---

### VGG19 weights

The perceptual loss needs ImageNet VGG19 (548 MB), which torchvision downloads
on first use. On this machine that download stalls, so either:

* let it finish once (it is cached at `~/.cache/torch/hub/checkpoints/`), or
* download `vgg19-dcbb9e9d.pth` separately and set `loss.vgg_weights:
  /path/to/vgg19-dcbb9e9d.pth` in both configs, or
* set `loss.perceptual: 0.0` to train without it — everything runs, but expect
  noticeably softer results.

---

## 6. Compute

Measured on this machine (Intel Mac, CPU-only — the installed torch build has no
MPS), batch size 4:

| config | warp s/step | tryon s/step | full run (12k + 15k steps) |
|---|---|---|---|
| default, 256×192, base 32 | 0.66 | 3.9 (5.5 with perceptual, 4.2 +GAN) | ~18 h |
| reduced, 192×144, base 24 | 0.35 | 1.6 | ~8 h |

Stage 1 is comfortable either way; stage 2 is the expensive half. Options:

* **Use a GPU.** A Colab T4 runs the whole pipeline in well under an hour.
  Nothing here is CPU-specific — `device: auto` picks up CUDA or MPS.
* **Stay on CPU** with the reduced config:

  ```bash
  python train_warp.py  --config configs/warp.yaml  data.root=... \
      data.height=192 data.width=144 model.base=24 train.steps=8000
  python train_tryon.py --config configs/tryon.yaml data.root=... \
      data.height=192 data.width=144 model.base=24 model.max_channels=192 \
      train.steps=10000
  ```

  Both stages must use the same resolution — stage 2 checks this and refuses to
  start otherwise, because the TPS kernel is precomputed per resolution. On a
  dataset this small the lower resolution costs less quality than you would
  expect.

---

## 7. Diagnosing results

| symptom | cause | fix |
|---|---|---|
| `warped` column is a black/white blob | garment mask covers the background | check the `garment mask` column; supply explicit masks or use plainer product shots |
| `target garment` shows different clothing to `garment` | wrong label scheme, or wrong per-sample region | `--diagnose-labels`; confirm `data.garment_type: auto` |
| `agnostic` still shows the original garment | wrong label scheme, or too little dilation | `--diagnose-labels`; raise `data.erase_dilate` |
| Garment lands in the wrong place | stage 1 undertrained | more steps; raise `loss.shape` |
| Garment torn or folded onto itself | flow unconstrained | raise `loss.tv`, `loss.smooth`, `loss.grid` |
| Garment looks like a rigid decal | residual flow too tight | raise `model.max_displacement`, lower `loss.tv` |
| Output blurry / washed out | composer ignoring the warp | raise `loss.alpha`; check `alpha` column is bright inside the garment |
| Great on train, bad on new garments | leak in the agnostic input | raise `data.erase_dilate`; verify the `agnostic` column in `check_dataset.py` |
| Face smeared | identity mask wrong | check CIHP labels 1/2/4/13 are present in `check_dataset.py` output |
| Quality collapsed after enabling GAN | discriminator won | raise `train.gan_start_step`, lower `train.lr_d`, keep DiffAugment on |

**Read the sample sheets left to right.** If the `warped` column is wrong,
nothing in stage 2 can save it — go back to stage 1.

**Judge on the off-diagonal.** Training is self-paired, so reconstructing a
person in their own clothes proves nothing. `infer.py --grid` puts every garment
on every person; those cells are the real result.

---

## 8. If you later get more data

The natural upgrades, in order of value:

1. **Add pose keypoints** (OpenPose 18-point heatmaps) to the condition. Worth
   about as much as doubling the dataset, and it is the standard input this
   family of models expects — we omit it only because your dataset does not
   ship it.
2. **Predict the target parse map** for the *new* garment (as in ACGPN/PF-AFN)
   instead of reusing the source person's. Necessary once garments differ in cut
   — a sleeveless top on a long-sleeved source person is currently the hardest
   case for this design.
3. **Parser-free distillation** (PF-AFN): use this model as a teacher to train a
   student that needs no parse map at inference. Removes the CIHP dependency
   from deployment entirely.
4. At ~5k+ pairs, revisit diffusion (a warped-garment-conditioned ControlNet).
   Not before.

---

## 9. File map

```
configs/warp.yaml            stage 1 hyperparameters, annotated with a tuning order
configs/tryon.yaml           stage 2 hyperparameters
train_warp.py                stage 1 training loop
train_tryon.py               stage 2 training loop (+ optional GAN)
infer.py                     single-pair and full-grid inference
scripts/check_dataset.py     run this first, every time
scripts/make_dummy_dataset.py synthetic data for smoke tests

vtonwarp/data/labels.py      CIHP taxonomy and semantic groupings
vtonwarp/data/manifest.py    dataset discovery and stem matching
vtonwarp/data/io.py          tolerant loaders for npy/png parse maps and masks
vtonwarp/data/agnostic.py    keep / erase / describe decomposition
vtonwarp/data/augment.py     paired augmentation
vtonwarp/data/dataset.py     the Dataset itself

vtonwarp/models/tps.py       thin-plate-spline grid generator
vtonwarp/models/warper.py    stage 1 network
vtonwarp/models/composer.py  stage 2 network
vtonwarp/models/discriminator.py  PatchGAN + DiffAugment
vtonwarp/models/blocks.py    shared conv blocks

vtonwarp/losses/perceptual.py    VGG19 perceptual and style loss
vtonwarp/losses/regularisers.py  flow smoothness, mask alignment, alpha terms
vtonwarp/losses/gan.py           hinge GAN objective

vtonwarp/engine/               dataloaders, EMA, checkpoints, visualisation
vtonwarp/utils/config.py       YAML config with CLI overrides
```

Every module has a header explaining *why* it exists, not just what it does.
