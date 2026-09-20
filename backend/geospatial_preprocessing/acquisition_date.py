"""Real capture dates for uploaded images, and the before/after decision built on them.

Nothing here invents a date. A date is only reported when it comes from either:
  * an explicit acquisition tag in the raster ("trusted"), or
  * a filename that matches a full, known product-naming convention ("trusted"), or
  * the generic TIFF DateTime tag, which is the time the FILE was written and is often a processing
    date rather than the capture date ("not trusted": shown, but never used to order images).
Anything else yields None, and the caller falls back to "ordered as uploaded".
"""
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

# Raster tag names (upper-cased) that state an ACQUISITION time.
_ACQUISITION_TAG_KEYS = (
    "ACQUISITIONDATETIME", "ACQUISITION_DATETIME", "ACQUISITIONDATE", "ACQUISITION_DATE",
    "DATE_ACQUIRED", "SENSING_TIME", "SENSING_START", "PRODUCT_START_TIME",
    "DATATAKE_1_DATATAKE_SENSING_START", "FIRST_LINE_TIME",
)
_FILE_TIME_TAG_KEYS = ("TIFFTAG_DATETIME",)

# Full product-ID conventions. Each captures (YYYYMMDD, [HHMMSS]) of the acquisition. They must match
# the whole naming pattern, not just contain eight digits.
_FILENAME_PATTERNS = (
    ("Sentinel-2 product ID", re.compile(r"^S2[A-D]_MSIL[12][AC]_(\d{8})T(\d{6})_", re.I)),
    ("Sentinel-2 tile/band name", re.compile(r"(?:^|_)T\d{2}[A-Z]{3}_(\d{8})T(\d{6})_", re.I)),
    ("Sentinel-1 product ID", re.compile(r"^S1[A-D]_[A-Z0-9]{2}_[A-Z]{3}[A-Z_]_\d[SA][A-Z]{2}_(\d{8})T(\d{6})_", re.I)),
    ("Landsat Collection 2 scene ID", re.compile(r"^L[CETOM]0?\d_[A-Z0-9]{4}_\d{6}_(\d{8})_\d{8}_\d{2}_[A-Z0-9]{2}(?:_|$)", re.I)),
)

_DATE_RE = re.compile(
    r"(?P<y>\d{4})[-:/]?(?P<m>\d{2})[-:/]?(?P<d>\d{2})"
    r"(?:[T ]+(?P<H>\d{2}):?(?P<M>\d{2}):?(?P<S>\d{2})?)?"
)


@dataclass(frozen=True)
class AcquisitionDate:
    when: datetime            # naive, UTC
    has_time: bool
    source: str               # human-readable provenance
    trusted: bool             # False = must not be used to decide image order

    @property
    def iso(self) -> str:
        return self.when.strftime("%Y-%m-%d %H:%M:%S UTC") if self.has_time else self.when.strftime("%Y-%m-%d")


def parse_date_string(text: str) -> Optional[tuple]:
    """(datetime, has_time) from an ISO-8601-like / EXIF-like / compact string, or None if it
    is not a plausible date (bad calendar date, or a year before Landsat-1 / after next year)."""
    m = _DATE_RE.search(str(text))
    if not m:
        return None
    try:
        y, mo, d = int(m["y"]), int(m["m"]), int(m["d"])
        has_time = m["H"] is not None
        h, mi, s = (int(m["H"]), int(m["M"]), int(m["S"] or 0)) if has_time else (0, 0, 0)
        when = datetime(y, mo, d, h, mi, s)
    except ValueError:
        return None
    if not (1972 <= y <= datetime.now(timezone.utc).year + 1):
        return None
    return when, has_time


def date_from_filename(filename: Optional[str]) -> Optional[AcquisitionDate]:
    if not filename:
        return None
    stem = os.path.splitext(os.path.basename(filename))[0]
    for label, pattern in _FILENAME_PATTERNS:
        m = pattern.search(stem)
        if not m:
            continue
        raw = m.group(1) + ("T" + m.group(2) if m.lastindex and m.lastindex >= 2 and m.group(2) else "")
        parsed = parse_date_string(raw)
        if parsed:
            return AcquisitionDate(parsed[0], parsed[1], f"filename ({label})", trusted=True)
    return None


def _all_tags(path: str) -> Dict[str, str]:
    import rasterio  # imported lazily: only GeoTIFF inputs need it
    tags: Dict[str, str] = {}
    with rasterio.open(path) as src:
        tags.update(src.tags())
        for ns in src.tag_namespaces():
            tags.update(src.tags(ns=ns))
    return {str(k).upper(): str(v) for k, v in tags.items()}


def date_from_tags(path: str) -> Optional[AcquisitionDate]:
    try:
        tags = _all_tags(path)
    except Exception:
        return None
    for key in _ACQUISITION_TAG_KEYS:
        if key in tags and (parsed := parse_date_string(tags[key])):
            return AcquisitionDate(parsed[0], parsed[1], f"raster tag {key}", trusted=True)
    for key in _FILE_TIME_TAG_KEYS:
        if key in tags and (parsed := parse_date_string(tags[key])):
            return AcquisitionDate(parsed[0], parsed[1], f"raster tag {key} (file creation time, not necessarily capture)", trusted=False)
    return None


def extract_acquisition_date(path: str, original_filename: Optional[str] = None) -> Optional[AcquisitionDate]:
    """Best available date, or None. Precedence: explicit acquisition tag > recognised product
    filename > generic file-time tag (untrusted). `original_filename` is the name the user uploaded
    (the temp path is renamed, so it carries no information)."""
    tag_date = date_from_tags(path) if path.lower().endswith((".tif", ".tiff")) else None
    if tag_date and tag_date.trusted:
        return tag_date
    return date_from_filename(original_filename) or tag_date


# ------------------------------------------------------------------ before / after decision

@dataclass
class TemporalOrder:
    swap: bool                       # True: the uploaded images must be exchanged (A was the later one)
    basis: str                       # "metadata" | "as-uploaded"
    summary: str
    before: Optional[AcquisitionDate] = None
    after: Optional[AcquisitionDate] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def prompt_dates(self) -> Optional[tuple]:
        """(before, after) date strings, only when both come from trusted metadata."""
        if self.basis == "metadata" and self.before and self.after:
            return self.before.iso, self.after.iso
        return None

    def to_dict(self) -> dict:
        def d(a):
            return None if a is None else {"date": a.iso, "source": a.source, "trusted": a.trusted}
        return {"basis": self.basis, "reordered": self.swap, "before": d(self.before), "after": d(self.after),
                "warnings": self.warnings, "summary": self.summary}


def resolve_temporal_order(a: Optional[AcquisitionDate], b: Optional[AcquisitionDate]) -> TemporalOrder:
    """Decide which uploaded image is 'before'. Images are only reordered when BOTH have trusted
    dates and A is later. Otherwise the upload order is taken as the user's assertion (A = before)."""
    warnings: List[str] = []

    if a and b and a.trusted and b.trusted:
        if a.when > b.when:
            return TemporalOrder(
                True, "metadata",
                f"Order taken from capture dates: before = {b.iso} ({b.source}), after = {a.iso} ({a.source}). "
                "The two images were swapped because the first upload is the later one.",
                before=b, after=a)
        if a.when == b.when:
            warnings.append("Both images carry the same capture date/time; order kept as uploaded.")
            return TemporalOrder(
                False, "as-uploaded",
                f"Both images are dated {a.iso}; treated Image A as BEFORE and Image B as AFTER, as uploaded.",
                before=a, after=b, warnings=warnings)
        return TemporalOrder(
            False, "metadata",
            f"Order confirmed by capture dates: before = {a.iso} ({a.source}), after = {b.iso} ({b.source}).",
            before=a, after=b)

    if a and b and a.when > b.when:
        warnings.append(
            "Date tags suggest Image A is later than Image B, but they are not reliable capture dates "
            "(e.g. file creation time); order kept as uploaded. Swap the uploads if that is wrong.")
    known = [f"{n} = {x.iso} ({x.source})" for n, x in (("Image A", a), ("Image B", b)) if x]
    extra = f" Dates found but not used to order the images: {'; '.join(known)}." if known else ""
    return TemporalOrder(
        False, "as-uploaded",
        "Capture dates are unknown or unreliable; treated Image A as BEFORE and Image B as AFTER, as "
        "uploaded (no date was assumed)." + extra,
        before=None, after=None, warnings=warnings)
