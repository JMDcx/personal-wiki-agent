"""Storage backends for resolved images."""

from __future__ import annotations

from pathlib import Path


class ImageStorage:
    """Storage abstraction for images."""

    def save(self, document_id: str, image_name: str, content: bytes) -> tuple[str, str]:
        raise NotImplementedError


class LocalImageStorage(ImageStorage):
    """Filesystem image storage with public URL mapping."""

    def __init__(self, root_dir: Path, url_prefix: str = "/assets/images") -> None:
        self.root_dir = root_dir
        self.url_prefix = url_prefix.rstrip("/")
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def save(self, document_id: str, image_name: str, content: bytes) -> tuple[str, str]:
        safe_name = Path(image_name).name or "image.bin"
        target_dir = self.root_dir / document_id
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / safe_name
        if target_path.exists():
            stem = target_path.stem
            suffix = target_path.suffix
            counter = 1
            while target_path.exists():
                target_path = target_dir / f"{stem}-{counter}{suffix}"
                counter += 1
        target_path.write_bytes(content)
        rel = target_path.relative_to(self.root_dir).as_posix()
        return str(target_path), f"{self.url_prefix}/{rel}"
