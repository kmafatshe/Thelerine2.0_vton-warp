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

# Bumped whenever the matching rules change, so a manifest written by older
# rules is rebuilt rather than silently reused. Version 1 collapsed phone
# filenames to their date, pairing many people with one garment.
MANIFEST_VERSION = 2

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

        if files:
            lines.append(f"      contents: {_describe_contents(files[0])}")

    return "\n".join(lines)


def _describe_contents(path: Path) -> str:
    """Report an array's shape and values, to identify what a folder holds.

    A label map, a one-hot stack, a pose heatmap and a binary mask are all just
    arrays on disk, and confusing them is silent: argmax over pose heatmaps
    produces a perfectly well-formed parse map that means nothing. Shape, dtype
    and the number of distinct values distinguish them at a glance.
    """
    try:
        import numpy as np

        suffix = path.suffix.lower()
        if suffix == ".npy":
            array = np.load(path, mmap_mode="r")
        elif suffix == ".npz":
            bundle = np.load(path)
            array = bundle[list(bundle.keys())[0]]
        else:
            from PIL import Image

            image = Image.open(path)
            array = np.array(image)
            mode = f"mode={image.mode} "
            unique = np.unique(array)
            kind = ("binary mask" if unique.size <= 2
                    else f"{unique.size} distinct values")
            return (f"{mode}shape={array.shape} dtype={array.dtype} {kind}"
                    f" range=[{array.min()}, {array.max()}]")

        array = np.asarray(array)
        unique = np.unique(array)
        if np.issubdtype(array.dtype, np.integer) and unique.size <= 32:
            kind = f"integer label map, ids {list(unique[:24])}"
        elif array.ndim == 3 and min(array.shape) <= 32:
            channels = min(array.shape)
            kind = (f"{channels}-channel stack — one-hot/probabilities or pose "
                    f"heatmaps, NOT a label map")
        elif unique.size <= 32:
            kind = f"{unique.size} distinct values {list(unique[:12])}"
        else:
            kind = f"continuous, {unique.size} distinct values"

        return (f"shape={array.shape} dtype={array.dtype} {kind}"
                f" range=[{array.min():.4g}, {array.max():.4g}]")
    except Exception as error:  # never let introspection break the report
        return f"(could not read: {type(error).__name__}: {error})"


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

    Every non-role token is kept. An earlier version collapsed the stem to its
    single all-digit token when there was exactly one, which looked tidy and was
    badly wrong for phone filenames: `IMG-20211010-WA0001`, `-WA0002` and
    `-WA0017` all reduced to `20211010`, because `wa0001` is not purely numeric
    and was discarded. Every photo taken on one date then shared a key, so they
    all matched whichever garment happened to be indexed first — a silent
    mispairing that trains the warper to deform one garment into another's
    silhouette.
    """
    parts = [p for p in _SPLIT.split(stem.lower()) if p]
    kept = [p for p in parts if p not in ROLE_TOKENS]
    if not kept:
        kept = parts
    return "_".join(kept)


def numeric_key(stem: str) -> str | None:
    """The stem's single numeric id, if it has exactly one.

    Used only as a fallback, so that `0001_person` still pairs with
    `0007_garment_dress` when the full stems differ. Ambiguous by nature, which
    is why it is never tried before an exact match.
    """
    parts = [p for p in _SPLIT.split(stem.lower()) if p]
    kept = [p for p in parts if p not in ROLE_TOKENS] or parts
    digits = [p for p in kept if p.isdigit()]
    if len(digits) == 1:
        return digits[0].lstrip("0") or "0"
    return None


def index_folder(folder: Path, exts=ANY_EXTS) -> dict:
    """Index a folder for cross-folder matching.

    Returns a dict with:
      exact     (subject, normalised_stem) -> path, plus a ("", stem) fallback
                so a flat garments folder can serve subject-partitioned people
      numeric   the same, keyed by the weaker numeric-id form
      collisions  key -> [paths] for any key claimed by more than one file

    Collisions are the thing to watch: two files reducing to one key means one
    of them silently wins every lookup.
    """
    exact: dict[tuple[str, str], Path] = {}
    numeric: dict[tuple[str, str], Path] = {}
    collisions: dict[str, list[Path]] = {}
    if not folder.exists():
        return {"exact": exact, "numeric": numeric, "collisions": collisions}

    seen: dict[str, list[Path]] = {}
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in exts:
            continue
        rel = path.relative_to(folder)
        subject = rel.parts[0] if len(rel.parts) > 1 else ""

        key = normalise_stem(path.stem)
        # Keyed by subject as well: the same filename under personA/ and
        # personB/ is not a collision, because lookup is subject-scoped and
        # each resolves to its own file. Only a clash *within* one subject
        # means a file can never be matched.
        seen.setdefault((subject, key), []).append(path)
        exact.setdefault((subject, key), path)
        exact.setdefault(("", key), path)

        weak = numeric_key(path.stem)
        if weak:
            numeric.setdefault((subject, weak), path)
            numeric.setdefault(("", weak), path)

    collisions = {key: paths for key, paths in seen.items() if len(paths) > 1}

    return {"exact": exact, "numeric": numeric, "collisions": collisions}


def _lookup(index: dict, subject: str, stem: str):
    """Exact match first, then the weaker numeric-id form."""
    key = normalise_stem(stem)
    for candidate in ((subject, key), ("", key)):
        if candidate in index["exact"]:
            return index["exact"][candidate], False

    weak = numeric_key(stem)
    if weak:
        for candidate in ((subject, weak), ("", weak)):
            if candidate in index["numeric"]:
                return index["numeric"][candidate], True
    return None, False


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
        index_folder(root / name, exts) if name
        else {"exact": {}, "numeric": {}, "collisions": {}}
    )
    garments = index(garment_dir, IMAGE_EXTS)
    cihps = index(cihp_dir, ANY_EXTS)
    segs = index(segmentation_dir, ANY_EXTS)
    people = index_folder(root / person_dir, IMAGE_EXTS)

    for name, folder in (("person", people), ("garments", garments),
                         ("cihp", cihps), ("segmentation", segs)):
        if folder["collisions"]:
            print(f"[manifest] !! {len(folder['collisions'])} key collision(s) in "
                  f"{name}/: different files reducing to the same id, so only one "
                  f"of each can ever be matched")
            for (subject, key), paths in list(folder["collisions"].items())[:6]:
                where = f"{subject}/" if subject else ""
                names = ", ".join(p.name for p in paths[:4])
                print(f"[manifest]    {where}'{key}': {names}"
                      + (" ..." if len(paths) > 4 else ""))

    records: list[TripletRecord] = []
    skipped: list[str] = []
    weak_matches = 0

    for path in sorted((root / person_dir).rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue

        rel = path.relative_to(root / person_dir)
        subject = rel.parts[0] if len(rel.parts) > 1 else ""
        key = normalise_stem(path.stem)

        garment, weak = _lookup(garments, subject, path.stem)
        if garment is None:
            # A person with no garment cannot supervise the warper.
            skipped.append(f"{rel} (no garment for key '{key}')")
            continue
        weak_matches += int(weak)

        cihp, _ = _lookup(cihps, subject, path.stem)
        seg, _ = _lookup(segs, subject, path.stem)

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

    if weak_matches:
        print(f"[manifest] {weak_matches} sample(s) paired by numeric id rather "
              f"than an exact stem match — check those pairs in --audit")

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
        "version": MANIFEST_VERSION,
        "train": [asdict(r) for r in train],
        "val": [asdict(r) for r in val],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2))


def read_manifest(path: Path):
    """Load a manifest, or None if it predates the current matching rules.

    Returning None rather than the stale records matters: a manifest is a cache
    of pairing decisions, and reusing one written under rules that mispaired
    files would quietly undo the fix.
    """
    payload = json.loads(Path(path).read_text())
    if payload.get("version", 1) != MANIFEST_VERSION:
        print(f"[manifest] {path} was written by older matching rules "
              f"(v{payload.get('version', 1)} < v{MANIFEST_VERSION}); rebuilding")
        return None

    to_records = lambda rows: [TripletRecord(**row) for row in rows]  # noqa: E731
    return to_records(payload["train"]), to_records(payload["val"])
