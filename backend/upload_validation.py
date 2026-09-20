"""Checks on uploaded bytes that do not trust anything the client claims (content type, filename)."""
import os
from typing import BinaryIO, Optional

# A guess at a sensible ceiling for one image (large GeoTIFFs are legitimately big). Override with
# the MAX_UPLOAD_MB environment variable.
DEFAULT_MAX_UPLOAD_MB = 100


class UploadTooLarge(Exception):
    pass


def max_upload_bytes() -> int:
    try:
        mb = float(os.environ.get("MAX_UPLOAD_MB", DEFAULT_MAX_UPLOAD_MB))
    except ValueError:
        mb = DEFAULT_MAX_UPLOAD_MB
    return int(max(mb, 0.001) * 1024 * 1024)


def max_request_bytes(max_images: int = 2) -> int:
    """Whole-request cap: every image at its limit plus a little room for the form fields."""
    return max_images * max_upload_bytes() + 1024 * 1024


def detect_image_kind(data: bytes) -> Optional[str]:
    """'png' | 'jpeg' | 'tiff' | 'webp' from the file's magic bytes, or None if unrecognised.
    'tiff' covers GeoTIFF too (classic and BigTIFF, both byte orders)."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if data[:4] in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"):
        return "tiff"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


def read_limited(file: BinaryIO, limit: int) -> bytes:
    """Read at most `limit` bytes; raise UploadTooLarge if the file is bigger (never loads more
    than limit + 1 bytes into memory)."""
    data = file.read(limit + 1)
    if len(data) > limit:
        raise UploadTooLarge(f"File exceeds the {limit // (1024 * 1024)} MB upload limit.")
    return data
