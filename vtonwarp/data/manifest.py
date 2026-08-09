"""
Dataset discovery.

Expected layout (subject folders are optional and may be nested arbitrarily):

    <root>/
        garments/
            personA/ 0001.jpg ...
            personB/ 0007.jpg ...
        person/
            personA/ 0001.jpg ...
            personB/ 0007.jpg ...
        cihp/
            personA/ 0001.npy ...          # H x W integer label map
        segmentation/
            personA/ 0001.png ...          # binary mask (garment or silhouette)

Files are matched across folders by a *normalised stem*: the filename without
its extension, lowercased, with common role suffixes/prefixes stripped
(`_person`, `_garment`, `_cihp`, `_seg`, `_mask`, ...). This tolerates the
inconsistent naming that hand-built datasets always end up with, e.g.
`0001_person.jpg` <-> `0001_garment_dress.png` <-> `0001_cihp.npy`.

The result is written to `manifest.json` so that a training run is exactly
reproducible even if files are later added to the folders.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ARRAY_EXTS = {".npy", ".npz"}
ANY_EXTS = IMAGE_EXTS | ARRAY_EXTS

# Tokens that describe a file's *role* rather than its identity. Stripped from
# both ends of a stem so that files from different folders line up.
ROLE_TOKENS = (
    "person", "people", "model", "img", "image",
    "garment", "garments", "cloth", "clothes", "clothing", "product",
    "cihp", "parse", "parsing", "label", "labels",
    "seg", "segm", "segmentation", "mask",
    "upper", "lower", "dress", "pants", "top", "shirt", "skirt", "coat",
)

_SPLIT = re.compile(r"[^a-z0-9]+")

# Folder names, in preference order, for each of the four roles. Hand-built
# datasets rename these constantly, so we resolve them rather than demand them.
ROLE_FOLDERS = {
    "person_dir": ("person", "persons", "people", "model", "models", "target"),
    "garment_dir": ("garments", "garment", "cloth", "cloths", "clothes",
                    "clothing", "product"),
    "cihp_dir": ("cihp", "cond", "conditional", "parse", "parsing", "parses",
                 "label", "labels", "human_parse"),
    "segmentation_dir": ("segmentation", "seg", "segm", "mask", "masks"),
}


def resolve_layout(root: Path, overrides: dict | None = None) -> dict[str, str | None]:
    """Work out what each of the four dataset folders is actually called.

    Tries, in order: an explicit override, an exact name match, a
    case-insensitive match, then a substring match (so `cihp_maps` resolves to
    the cihp role). Returns None for a role with no plausible folder.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"dataset root not found: {root}")

    present = {d.name.lower(): d.name for d in root.iterdir() if d.is_dir()}
    overrides = overrides or {}
    resolved: dict[str, str | None] = {}

    for role, candidates in ROLE_FOLDERS.items():
        override = overrides.get(role)
        if override:
            resolved[role] = override
            continue

        match = next((present[c] for c in candidates if c in present), None)
        if match is None:
            # Substring fallback: `cihp_maps`, `garment_images`, `seg_masks`.
            match = next(
                (actual for name, actual in sorted(present.items())
                 if any(c in name for c in candidates)),
                None,
            )
        resolved[role] = match

    return resolved


def describe_layout(root: Path, samples: int = 3) -> str:
    """A human-readable report of what is in the dataset root.

    Printed whenever matching fails. It shows each folder's file count,
    extensions, and a few filenames alongside the normalised stem they reduce
    to — which is what cross-folder matching actually compares. A mismatch is
    then obvious at a glance instead of being guessed at.
    """
    root = Path(root)
    lines = [f"Dataset root: {root}"]

    directories = sorted(d for d in root.iterdir() if d.is_dir())
    if not directories:
        lines.append("  (no subfolders at all — is this the right root?)")

    for directory in directories:
        files = [f for f in sorted(directory.rglob("*"))
                 if f.is_file() and not f.name.startswith(".")]
        extensions = sorted({f.suffix.lower() for f in files})
        subfolders = sorted(c.name for c in directory.iterdir() if c.is_dir())

        lines.append(f"\n  {directory.name}/  —  {len(files)} files  "
                     f"{extensions}  subfolders: {subfolders or '(none)'}")
        for f in files[:samples]:
            lines.append(f"      {f.relative_to(directory)}"
                         f"   ->  key '{normalise_stem(f.stem)}'")

    return "\n".join(lines)


@dataclass
class TripletRecord:
    """One training sample: a person, the garment they wear, and conditionals."""

    key: str
    subject: str          # "personA" / "personB" / "" — the subfolder it came from
    person: str           # paths are stored relative to the dataset root
    garment: str
    cihp: str | None
    segmentation: str | None

    def resolve(self, root: Path) -> dict:
        out = {"key": self.key, "subject": self.subject}
        for field in ("person", "garment", "cihp", "segmentation"):
            value = getattr(self, field)
            out[field] = (root / value) if value else None
        return out


def normalise_stem(stem: str) -> str:
    """Reduce a filename stem to a stable identity token.

    `0001_person` -> `0001`,  `img_0007_garment_dress` -> `0007`.
    If stripping removes everything we fall back to the original stem.
    """
    parts = [p for p in _SPLIT.split(stem.lower()) if p]
    kept = [p for p in parts if p not in ROLE_TOKENS]
    if not kept:
        kept = parts
    # Prefer the numeric id if there is exactly one — it is the most reliable
    # cross-folder anchor.
    digits = [p for p in kept if p.isdigit()]
    if len(digits) == 1:
        return digits[0].lstrip("0") or "0"
    return "_".join(kept)


def index_folder(folder: Path, exts=ANY_EXTS) -> dict[tuple[str, str], Path]:
    """Map (subject, normalised_stem) -> path for every file under `folder`.

    Also registers a ("", stem) fallback so that a flat garments folder can
    still serve subject-partitioned person images.
    """
    index: dict[tuple[str, str], Path] = {}
    if not folder.exists():
        return index

    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in exts:
            continue
        rel = path.relative_to(folder)
        subject = rel.parts[0] if len(rel.parts) > 1 else ""
        key = normalise_stem(path.stem)
        index.setdefault((subject, key), path)
        index.setdefault(("", key), path)
    return index


def build_manifest(
    root: Path,
    person_dir: str | None = "person",
    garment_dir: str | None = "garments",
    cihp_dir: str | None = "cihp",
    segmentation_dir: str | None = "segmentation",
) -> list[TripletRecord]:
    """Scan the dataset root and pair every person image with its conditionals.

    Any of the folder names may be None (meaning "not present"); the resulting
    records simply carry None for that field, and the caller decides whether
    that is fatal.
    """
    root = Path(root).resolve()
    if not person_dir:
        raise FileNotFoundError(
            f"No person folder could be identified under {root}.\n\n"
            + describe_layout(root)
        )

    people = root / person_dir
    if not people.exists():
        raise FileNotFoundError(f"person folder not found: {people}")

    index = lambda name, exts: (  # noqa: E731
        index_folder(root / name, exts) if name else {}
    )
    garments = index(garment_dir, IMAGE_EXTS)
    cihps = index(cihp_dir, ANY_EXTS)
    segs = index(segmentation_dir, ANY_EXTS)

    records: list[TripletRecord] = []
    skipped: list[str] = []

    for path in sorted(people.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue

        rel = path.relative_to(people)
        subject = rel.parts[0] if len(rel.parts) > 1 else ""
        key = normalise_stem(path.stem)

        garment = garments.get((subject, key)) or garments.get(("", key))
        if garment is None:
            # A person with no garment cannot supervise the warper.
            skipped.append(f"{rel} (no garment for key '{key}')")
            continue

        cihp = cihps.get((subject, key)) or cihps.get(("", key))
        seg = segs.get((subject, key)) or segs.get(("", key))

        records.append(
            TripletRecord(
                key=f"{subject}/{key}" if subject else key,
                subject=subject,
                person=str(path.relative_to(root)),
                garment=str(garment.relative_to(root)),
                cihp=str(cihp.relative_to(root)) if cihp else None,
                segmentation=str(seg.relative_to(root)) if seg else None,
            )
        )

    if skipped:
        print(f"[manifest] skipped {len(skipped)} unmatched person image(s):")
        for line in skipped[:10]:
            print(f"           - {line}")

    return records


def split_records(
    records: list[TripletRecord], val_fraction: float, seed: int = 42
) -> tuple[list[TripletRecord], list[TripletRecord]]:
    """Deterministic train/val split.

    On a tiny dataset we hold out whole samples rather than doing k-fold, but we
    guarantee at least one validation sample so that visual monitoring works.
    """
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_fraction))) if len(shuffled) > 1 else 0
    return shuffled[n_val:], shuffled[:n_val]


def write_manifest(path: Path, train, val) -> None:
    payload = {
        "train": [asdict(r) for r in train],
        "val": [asdict(r) for r in val],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2))


def read_manifest(path: Path) -> tuple[list[TripletRecord], list[TripletRecord]]:
    payload = json.loads(Path(path).read_text())
    to_records = lambda rows: [TripletRecord(**row) for row in rows]  # noqa: E731
    return to_records(payload["train"]), to_records(payload["val"])
