import {
  ChatMessage,
  ThreadSummary,
  ThreadSummaryDTO,
  CreateThreadResponseDTO,
  MessageOutDTO,
  PinResponseDTO,
  DeleteThreadResponseDTO,
  FilePayload,
  UploadedFileResultDTO,
  TranscribeResponseDTO,
  EditResponseDTO,
  StreamEvent,
  ApiError,
  GeneratedImageDTO,
  LibraryMediaDTO,
} from "./types";
import { getStoredToken } from "./useAuth";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

/**
 * Fired whenever any request comes back 401 (expired/invalid session),
 * so the app can drop back to the login screen. Set once from
 * page.tsx via setUnauthorizedHandler, rather than importing React
 * state into this plain module.
 */
let onUnauthorized: (() => void) | null = null;

export function setUnauthorizedHandler(handler: () => void) {
  onUnauthorized = handler;
}

function authHeaders(extra: Record<string, string> = {}): Record<string, string> {
  const token = getStoredToken();
  return token
    ? { ...extra, Authorization: `Bearer ${token}` }
    : extra;
}

async function handleJsonResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let message = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      message = body?.detail ?? body?.message ?? message;
    } catch {
      // body wasn't JSON — keep the generic message
    }

    if (res.status === 401) {
      onUnauthorized?.();
    }

    throw new ApiError(message, res.status);
  }
  return res.json() as Promise<T>;
}

function toThreadSummary(dto: ThreadSummaryDTO): ThreadSummary {
  return { threadId: dto.thread_id, title: dto.title, pinned: dto.pinned, archived: dto.archived };
}

function toChatMessage(dto: MessageOutDTO, index: number): ChatMessage {
  // MessageOut has no id — synthesize a stable-enough one for React keys.
  // ASSUMPTION FLAG: role is typed as `str` in schemas.py; we narrow it to
  // "user" | "assistant" and fall back to "assistant" for anything
  // unexpected (e.g. a future "system"/"tool" role) rather than crashing.
  const role = dto.role === "user" ? "user" : "assistant";
  return { id: `${role}-${index}`, role, content: dto.content };
}

/** GET /threads */
export async function listThreads(
  signal?: AbortSignal,
): Promise<ThreadSummary[]> {
  const res = await fetch(`${API_BASE_URL}/threads`, {
    headers: authHeaders(),
    signal,
  });
  const dtos = await handleJsonResponse<ThreadSummaryDTO[]>(res);
  return dtos.map(toThreadSummary);
}

export async function listGeneratedImages(
  signal?: AbortSignal,
): Promise<GeneratedImageDTO[]> {
  const res = await fetch(`${API_BASE_URL}/images`, {
    headers: authHeaders(),
    signal,
  });
  return handleJsonResponse<GeneratedImageDTO[]>(res);
}

export async function listLibraryMedia(
  signal?: AbortSignal,
): Promise<LibraryMediaDTO[]> {
  const res = await fetch(`${API_BASE_URL}/library/media`, {
    headers: authHeaders(),
    signal,
  });
  return handleJsonResponse<LibraryMediaDTO[]>(res);
}

export async function generateImage(
  prompt: string,
  signal?: AbortSignal,
): Promise<GeneratedImageDTO> {
  const res = await fetch(`${API_BASE_URL}/images/generate`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ prompt }),
    signal,
  });
  return handleJsonResponse<GeneratedImageDTO>(res);
}

export async function fetchGeneratedImage(
  imageUrl: string,
  signal?: AbortSignal,
): Promise<Blob> {
  const res = await fetch(`${API_BASE_URL}${imageUrl}`, {
    headers: authHeaders(),
    signal,
  });
  if (!res.ok) {
    if (res.status === 401) onUnauthorized?.();
    let message = `Could not load image (${res.status})`;
    try {
      const body = await res.json();
      message = body?.detail ?? message;
    } catch {
      // Keep the status-based message when the server response isn't JSON.
    }
    throw new ApiError(message, res.status);
  }
  return res.blob();
}

/** POST /threads — create a new (untitled) thread. */
export async function createThread(): Promise<ThreadSummary> {
  const res = await fetch(`${API_BASE_URL}/threads`, {
    method: "POST",
    headers: authHeaders(),
  });
  const dto = await handleJsonResponse<CreateThreadResponseDTO>(res);
  // CreateThreadResponse has no title/pinned yet — the backend hasn't
  // named it. Show a placeholder until the thread list is next refreshed.
  return { threadId: dto.thread_id, title: "New chat", pinned: false, archived: false };
}

/** GET /threads/{id}/messages */
export async function getThreadMessages(
  threadId: string,
  signal?: AbortSignal,
): Promise<ChatMessage[]> {
  const res = await fetch(`${API_BASE_URL}/threads/${threadId}/messages`, {
    headers: authHeaders(),
    signal,
  });
  const dtos = await handleJsonResponse<MessageOutDTO[]>(res);
  return dtos.map(toChatMessage);
}

/**
 * POST /threads/{id}/pin — matches main.py's pin_thread_endpoint, which
 * always sets pinned=true (a dedicated endpoint, not a toggle).
 */
export async function pinThread(threadId: string): Promise<boolean> {
  const res = await fetch(`${API_BASE_URL}/threads/${threadId}/pin`, {
    method: "POST",
    headers: authHeaders(),
  });
  const dto = await handleJsonResponse<PinResponseDTO>(res);
  return dto.pinned;
}

/** POST /threads/{id}/archive */
export async function archiveThread(threadId: string): Promise<boolean> {
  const res = await fetch(`${API_BASE_URL}/threads/${threadId}/archive`, {
    method: "POST",
    headers: authHeaders(),
  });
  const dto = await handleJsonResponse<{ archived: boolean }>(res);
  return dto.archived;
}

/** POST /threads/{id}/unarchive */
export async function unarchiveThread(threadId: string): Promise<boolean> {
  const res = await fetch(`${API_BASE_URL}/threads/${threadId}/unarchive`, {
    method: "POST",
    headers: authHeaders(),
  });
  const dto = await handleJsonResponse<{ archived: boolean }>(res);
  return dto.archived;
}

/**
 * POST /threads/{id}/unpin — matches main.py's unpin_thread_endpoint,
 * which always sets pinned=false.
 */
export async function unpinThread(threadId: string): Promise<boolean> {
  const res = await fetch(`${API_BASE_URL}/threads/${threadId}/unpin`, {
    method: "POST",
    headers: authHeaders(),
  });
  const dto = await handleJsonResponse<PinResponseDTO>(res);
  return dto.pinned;
}

/** DELETE /threads/{id} */
export async function deleteThread(threadId: string): Promise<void> {
  const res = await fetch(`${API_BASE_URL}/threads/${threadId}`, {
    method: "DELETE",
    headers: authHeaders(),
  });
  const dto = await handleJsonResponse<DeleteThreadResponseDTO>(res);
  if (!dto.deleted) {
    throw new ApiError(dto.cleanup_warning ?? "Delete did not complete.");
  }
}

/**
 * Parses one raw SSE frame (the text between blank-line separators) into
 * a StreamEvent, using the `event:`/`data:` lines api/streaming.py's
 * format_sse() writes. Unknown event names are ignored rather than
 * thrown, so a future/extra event type doesn't crash the stream.
 */
function parseSseFrame(frame: string): StreamEvent | null {
  const lines = frame.split("\n");
  const eventLine = lines.find((l) => l.startsWith("event:"));
  const dataLine = lines.find((l) => l.startsWith("data:"));
  if (!eventLine || !dataLine) return null;

  const eventName = eventLine.slice("event:".length).trim();
  const raw = dataLine.slice("data:".length).trim();
  if (!raw) return null;

  let data: Record<string, unknown>;
  try {
    data = JSON.parse(raw);
  } catch {
    return null; // malformed frame — skip rather than crash the stream
  }

  switch (eventName) {
    case "text":
      return typeof data.text === "string"
        ? { type: "text", text: data.text }
        : null;
    case "image":
      return typeof data.url === "string"
        ? { type: "image", url: data.url }
        : null;
    case "activity":
      return typeof data.message === "string"
        ? { type: "activity", message: data.message }
        : null;
    case "done":
      return { type: "done" };
    case "error":
      return {
        type: "error",
        message:
          typeof data.message === "string" ? data.message : "Stream error.",
      };
    default:
      return null; // unrecognized event name — ignore, don't crash
  }
}

/**
 * POST /threads/{id}/messages/stream — SSE chat streaming via fetch +
 * ReadableStream (not EventSource, since this is a POST with a body).
 * Sends both `files` (RAG documents already uploaded via /files/upload)
 * and `images` (base64 photos for the vision path) alongside the text,
 * matching SendMessageRequest exactly. Returns an abort() handle so
 * callers can cancel mid-stream.
 */
export function streamMessage(
  threadId: string,
  content: string,
  files: FilePayload[],
  images: { image_bytes: string; image_type: string; filename: string }[],
  onEvent: (event: StreamEvent) => void,
  onError: (error: Error) => void,
): { abort: () => void } {
  const controller = new AbortController();

  (async () => {
    try {
      const res = await fetch(
        `${API_BASE_URL}/threads/${threadId}/messages/stream`,
        {
          method: "POST",
          headers: authHeaders({ "Content-Type": "application/json" }),
          body: JSON.stringify({ content, files, images }),
          signal: controller.signal,
        },
      );

      if (!res.ok || !res.body) {
        if (res.status === 401) {
          onUnauthorized?.();
        }
        throw new ApiError(
          `Stream failed to start (${res.status})`,
          res.status,
        );
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        const frames = buffer.split("\n\n");
        buffer = frames.pop() ?? ""; // keep incomplete tail for next chunk

        for (const frame of frames) {
          const event = parseSseFrame(frame);
          if (event) onEvent(event);
        }
      }
    } catch (err) {
      if (controller.signal.aborted) return; // intentional cancel
      onError(err instanceof Error ? err : new Error("Stream failed"));
    }
  })();

  return { abort: () => controller.abort() };
}

/**
 * POST /files/upload — multipart form upload, matches main.py's
 * upload_files(thread_id: Form, files: list[UploadFile]).
 * One failed file doesn't throw for the whole batch — per-file results
 * (including per-file errors) come back in the array, matching the
 * backend's "independent per-file" behavior.
 */
export async function uploadFiles(
  threadId: string,
  files: File[],
): Promise<UploadedFileResultDTO[]> {
  const formData = new FormData();
  formData.append("thread_id", threadId);
  for (const file of files) {
    formData.append("files", file);
  }

  const res = await fetch(`${API_BASE_URL}/files/upload`, {
    method: "POST",
    headers: authHeaders(), // no Content-Type — browser sets multipart boundary
    body: formData,
  });

  return handleJsonResponse<UploadedFileResultDTO[]>(res);
}

/** POST /voice/transcribe — matches main.py's voice_transcribe(audio: UploadFile). */
export async function transcribeAudio(audioBlob: Blob): Promise<string> {
  const formData = new FormData();
  formData.append("audio", audioBlob, "recording.webm");

  const res = await fetch(`${API_BASE_URL}/voice/transcribe`, {
    method: "POST",
    headers: authHeaders(),
    body: formData,
  });

  const dto = await handleJsonResponse<TranscribeResponseDTO>(res);
  return dto.text;
}

/** POST /voice/speak — returns raw audio bytes (audio/mpeg), not JSON. */
export async function speakText(text: string, signal?: AbortSignal): Promise<Blob> {
  const res = await fetch(`${API_BASE_URL}/voice/speak`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ text }),
    signal,
  });

  if (!res.ok) {
    let message = `Could not generate audio (${res.status})`;
    try {
      const body = await res.json();
      message = body?.detail ?? message;
    } catch {
      // not JSON — keep generic message
    }
    if (res.status === 401) {
      onUnauthorized?.();
    }
    throw new ApiError(message, res.status);
  }

  return res.blob();
}

/**
 * GET /threads/{id}/export — triggers a browser download of the
 * returned Markdown transcript. Reads the filename from
 * Content-Disposition when present, falling back to a generic name.
 */
export async function exportThread(threadId: string): Promise<void> {
  const res = await fetch(`${API_BASE_URL}/threads/${threadId}/export`, {
    headers: authHeaders(),
  });

  if (!res.ok) {
    let message = `Could not export this chat (${res.status})`;
    try {
      const body = await res.json();
      message = body?.detail ?? message;
    } catch {
      // not JSON — keep generic message
    }
    if (res.status === 401) {
      onUnauthorized?.();
    }
    throw new ApiError(message, res.status);
  }

  const blob = await res.blob();
  const disposition = res.headers.get("Content-Disposition") ?? "";
  const match = disposition.match(/filename="?([^"]+)"?/);
  const filename = match?.[1] ?? `chat_${threadId.slice(0, 8)}.md`;

  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

/** POST /threads/{id}/edit — removes the last user turn from the checkpoint. */
export async function editLastMessage(threadId: string): Promise<boolean> {
  const res = await fetch(`${API_BASE_URL}/threads/${threadId}/edit`, {
    method: "POST",
    headers: authHeaders(),
  });
  const dto = await handleJsonResponse<EditResponseDTO>(res);
  return dto.removed;
}

/** PUT /threads/{id}/rename — sets a custom display title for the thread. */
export async function renameThread(
  threadId: string,
  title: string,
): Promise<ThreadSummary> {
  const res = await fetch(`${API_BASE_URL}/threads/${threadId}/rename`, {
    method: "PUT",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ title }),
  });
  const dto = await handleJsonResponse<ThreadSummaryDTO>(res);
  return toThreadSummary(dto);
}