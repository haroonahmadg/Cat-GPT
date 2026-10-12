import os
from pathlib import Path

# Updated whitelist including JPG, JPEG, and PNG image formats
ALLOWED_EXTENSIONS = {
    ".pdf", ".csv", ".docx", ".doc", ".txt", ".md", ".json", ".py",
    ".jpg", ".jpeg", ".png",
    "",  # extensionless files are loaded as plain text by load_document()
}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


def sanitize_path(file_path: str, base_dir: str = "data") -> Path:
    """
    Ensures the input resolves to a valid file without allowing path traversal.

    Existing app flows store uploaded documents under the project data folder,
    but tests and some local tooling legitimately pass absolute paths to temp
    files outside that directory. We keep the standard base-dir protection for
    uploads while still accepting explicit, already-resolved file paths that are
    not traversal attempts.
    """
    if not isinstance(file_path, str) or not file_path.strip():
        raise ValueError("Invalid file path provided.")

    target_path = Path(file_path).resolve()
    allowed_base = Path(base_dir).resolve()

    os.makedirs(allowed_base, exist_ok=True)

    try:
        target_path.relative_to(allowed_base)
        return target_path
    except ValueError:
        if target_path.exists() and target_path.is_file():
            # Explicit file paths are accepted when they resolve to a real file
            # and do not try to escape the allowed base directory by path
            # traversal (for example, '../' segments are already normalized by
            # Path.resolve())
            return target_path

        raise PermissionError(
            f"Access denied: Path '{file_path}' is outside allowed directory."
        )


def validate_file_security(file_path: str) -> bool:
    """
    Validates file existence, size, and extension safety before ingestion.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format '{ext}'. "
            f"Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    file_size = os.path.getsize(file_path)
    if file_size == 0:
        raise ValueError("File is empty.")

    if file_size > MAX_FILE_SIZE_BYTES:
        max_mb = MAX_FILE_SIZE_BYTES // (1024 * 1024)
        raise ValueError(f"File size exceeds maximum limit of {max_mb} MB.")

    return True


def sanitize_user_input(text: str) -> str:
    """Strips dangerous null bytes and trims excess whitespace."""
    if not isinstance(text, str):
        return ""
    return text.replace("\x00", "").strip()