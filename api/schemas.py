from __future__ import annotations

from pydantic import BaseModel, Field


class CreateThreadResponse(BaseModel):
    thread_id: str


class ThreadSummary(BaseModel):
    thread_id: str
    title: str
    pinned: bool
    archived: bool = False

class ArchiveResponse(BaseModel):
    archived: bool

class GoogleLoginRequest(BaseModel):
    credential: str


class RegisterRequest(BaseModel):
    email: str
    password: str
    name: str | None = None


class PasswordLoginRequest(BaseModel):
    email: str
    password: str


class SetPasswordRequest(BaseModel):
    new_password: str
    # Required only when the account already has a password.
    current_password: str | None = None


class SetPasswordResponse(BaseModel):
    password_set: bool


class LoginResponse(BaseModel):
    token: str
    email: str
    name: str | None = None
    # False for a Google-only account, which has no password yet. The
    # frontend uses this to offer "set a password" instead of "change
    # password", and to know whether current_password is required.
    has_password: bool = False


class MeResponse(BaseModel):
    """The signed-in account, used to validate a stored token on load."""

    email: str
    name: str | None = None
    has_password: bool = False

class RenameThreadRequest(BaseModel):
    title: str

class MessageOut(BaseModel):
    role: str
    content: str


class ImagePayload(BaseModel):
    """Matches the shape agent.graph.parse_input_messages already expects.

    image_bytes is a base64-encoded string — parse_input_messages already
    supports str input for this field (decode_image_bytes handles it), so
    no changes to graph.py are needed to accept this over JSON.
    """

    image_bytes: str
    image_type: str = "image/png"
    filename: str = "image"


class FilePayload(BaseModel):
    """A file already saved on disk via POST /files/upload.

    Matches the {filename, file_path} shape parse_input_messages expects.
    """

    filename: str
    file_path: str


class SendMessageRequest(BaseModel):
    content: str = ""
    images: list[ImagePayload] = Field(default_factory=list)
    files: list[FilePayload] = Field(default_factory=list)
    research_mode: bool = False


class GenerateImageRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=500)


class GeneratedImageOut(BaseModel):
    filename: str
    url: str
    created_at: float


class LibraryMediaOut(BaseModel):
    filename: str
    url: str
    created_at: float
    kind: str
    folder: str


class EditResponse(BaseModel):
    removed: bool


class DeleteThreadResponse(BaseModel):
    deleted: bool
    cleanup_warning: str | None = None


class PinResponse(BaseModel):
    pinned: bool


class UploadedFileResult(BaseModel):
    filename: str
    file_path: str | None = None
    chunks: int | None = None
    error: str | None = None
    progress: list[str] = Field(default_factory=list)


class TranscribeResponse(BaseModel):
    text: str


class SpeakRequest(BaseModel):
    text: str = Field(max_length=5000)