from __future__ import annotations

from typing import Any


_DATALOADERS: Any = None
_ORIGINAL_VERIFY: Any = None
_ORIGINAL_TRANSPOSE: Any = None


class _NoWriteImage:
    def __init__(self, image: Any) -> None:
        self.image = image

    def save(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def _read_only_transpose(image: Any) -> _NoWriteImage:
    return _NoWriteImage(_ORIGINAL_TRANSPOSE(image))


def verify_image_label_read_only(args: Any) -> Any:
    """Run YOLOv5 validation while preventing its truncated-JPEG write-back."""
    if _DATALOADERS is None:
        # Windows multiprocessing imports this callable in a fresh interpreter.
        # Initialize from the pinned source module inherited on sys.path.
        import utils.dataloaders as dataloaders

        install_read_only_verifier(dataloaders)
    image_ops = _DATALOADERS.ImageOps
    image_ops.exif_transpose = _read_only_transpose
    try:
        result = _ORIGINAL_VERIFY(args)
    finally:
        image_ops.exif_transpose = _ORIGINAL_TRANSPOSE
    if result and result[-1] and "corrupt JPEG restored and saved" in result[-1]:
        values = list(result)
        values[-1] = values[-1].replace("corrupt JPEG restored and saved", "incomplete JPEG accepted read-only")
        return tuple(values)
    return result


def install_read_only_verifier(dataloaders: Any) -> None:
    global _DATALOADERS, _ORIGINAL_TRANSPOSE, _ORIGINAL_VERIFY
    _DATALOADERS = dataloaders
    _ORIGINAL_VERIFY = dataloaders.verify_image_label
    _ORIGINAL_TRANSPOSE = dataloaders.ImageOps.exif_transpose
    dataloaders.verify_image_label = verify_image_label_read_only
