import re
import os
import requests
from dataclasses import dataclass
from pathlib import Path
import importlib.util
from character.constants import DATA_PATH, ensure_data_dirs
from dotenv import load_dotenv

load_dotenv()

CHECKPOINT_DIR = Path(os.getenv("CHARACTER_CHECKPOINT_DIR", DATA_PATH / "checkpoints"))

@dataclass
class TinkerStatus:
    installed: bool
    api_key_set: bool
    torch_installed: bool
    supported_models: list[str] | None = None
    capabilities_error: str | None = None

    @property
    def ready(self) -> bool:
        return self.installed and self.api_key_set and self.torch_installed

    @classmethod
    def check(cls) -> "TinkerStatus":
        """Detect whether we can talk to Tinker from this session."""
        torch_installed = importlib.util.find_spec("torch") is not None
        installed = importlib.util.find_spec("tinker") is not None
        api_key_set = bool(os.getenv("TINKER_API_KEY"))
        supported_models: list[str] | None = None
        capabilities_error: str | None = None

        if installed and api_key_set:
            try:
                # Local import to avoid circular dependency or early import issues
                import tinker
                service_client = tinker.ServiceClient()
                capabilities = service_client.get_server_capabilities()
                supported_models_raw = getattr(capabilities, "supported_models", []) or []
                supported_models = [
                    getattr(item, "model_name", getattr(item, "name", str(item)))
                    for item in supported_models_raw
                ]
            except Exception as exc:  # noqa: BLE001
                capabilities_error = str(exc)

        return cls(
            installed=installed,
            api_key_set=api_key_set,
            torch_installed=torch_installed,
            supported_models=supported_models,
            capabilities_error=capabilities_error,
        )

def slugify(label: str) -> str:
    """Convert a free-form name into a file-safe slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
    return slug

def is_url(target: str) -> bool:
    """Basic URL detection for remote checkpoints."""
    return target.startswith("http://") or target.startswith("https://")

def ensure_checkpoint_dir() -> Path:
    """Create a local directory for checkpoint downloads."""
    ensure_data_dirs()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    return CHECKPOINT_DIR

def download_artifact(target: str) -> Path:
    """
    Download a checkpoint from a URL or Tinker path to the local checkpoint directory.

    If the target is already a local path, return it unchanged.
    Handles:
    - Local files
    - HTTP/HTTPS URLs
    - Tinker paths (e.g., tinker://...sampler_weights/...)
    """
    path = Path(target)
    if path.exists():
        return path

    # Handle Tinker checkpoint paths (e.g., tinker://...)
    if target.startswith("tinker://"):
        try:
            import tinker
            ensure_checkpoint_dir()

            # Get signed URL from Tinker
            sc = tinker.ServiceClient()
            rc = sc.create_rest_client()
            url_response = rc.get_checkpoint_archive_url_from_tinker_path(target).result()

            # Download from signed URL
            filename = target.split("/")[-1] or "checkpoint"
            dest = CHECKPOINT_DIR / f"{filename}.tar"

            import urllib.request
            urllib.request.urlretrieve(url_response.url, str(dest))
            return dest

        except Exception as e:
            raise RuntimeError(f"Failed to download Tinker checkpoint {target}: {e}")

    if not is_url(target):
        raise FileNotFoundError(f"Checkpoint not found: {target}")

    # Handle regular HTTP/HTTPS URLs
    ensure_checkpoint_dir()
    filename = Path(target).name or "checkpoint.bin"
    dest = CHECKPOINT_DIR / filename

    with requests.get(target, stream=True, timeout=300) as response:
        response.raise_for_status()
        with dest.open("wb") as fp:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    fp.write(chunk)
    return dest
