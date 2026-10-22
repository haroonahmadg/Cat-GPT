/**
 * Types matching api/schemas.py exactly (snake_case fields), plus the
 * frontend-facing shapes (camelCase) that Sidebar/page.tsx already expect.
 * api.ts is responsible for mapping between the two.
 */

// ---- Frontend-facing shapes (unchanged from Phase 2's ThreadSummary) ----

export type ThreadSummary = {
  threadId: string;
  title: string;
  pinned: boolean;
  archived: boolean;
};

export type ChatRole = "user" | "assistant";

export type Citation = {
  file: string;
  page: number | null;
  preview: string;
};

export type ChatMessage = {
  id: string;
  role: ChatRole;
  content: string;
  citations?: Citation[];
  streaming?: boolean;
  imageUrl?: string;
};

// ---- Raw backend shapes (api/schemas.py) ----

export type ThreadSummaryDTO = {
  thread_id: string;
  title: string;
  pinned: boolean;
  archived: boolean;
};

export type LoginResponseDTO = {
  token?: string;
  email?: string;
  name?: string | null;
  has_password?: boolean;
  detail?: string;
};

export type MeResponseDTO = {
  email: string;
  name?: string | null;
  has_password: boolean;
};

export type AuthUser = {
  email: string;
  name: string | null;
  hasPassword: boolean;
};

export type CreateThreadResponseDTO = {
  thread_id: string;
};

export type MessageOutDTO = {
  role: string;
  content: string;
};

export type PinResponseDTO = {
  pinned: boolean;
};

export type DeleteThreadResponseDTO = {
  deleted: boolean;
  cleanup_warning?: string | null;
};

// ---- SSE event shapes (api/streaming.py's format_sse) ----
//
// ASSUMPTION FLAG: event names ("text" | "activity" | "done" | "error")
// are not visible in streaming.py itself (it only shows the generic
// format_sse(event, data) helper) — confirm the actual names emitted by
// api/main.py's stream route and adjust parseSseEvent() in api.ts if
// they differ. Everything else here is derived directly from
// extract_updates_text/extract_activity_message's known shapes.
export type StreamEvent =
  | { type: "text"; text: string }
  | { type: "activity"; message: string }
  | { type: "image"; url: string }
  | { type: "citations"; items: Citation[] }
  | { type: "done" }
  | { type: "error"; message: string };

export class ApiError extends Error {
  status?: number;
  constructor(message: string, status?: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}
// ---- File upload ----

export type FilePayload = {
  filename: string;
  file_path: string;
};

export type UploadedFileResultDTO = {
  filename: string;
  file_path: string | null;
  chunks: number | null;
  error: string | null;
  progress: string[];
};

// Frontend-facing: a file the user has attached and is ready to send.
export type Attachment = {
  id: string;
  filename: string;
  filePath: string | null; // null while uploading, or if upload failed
  status: "uploading" | "ready" | "error";
  error?: string;
};

// ---- Voice ----

export type TranscribeResponseDTO = {
  text: string;
};

// ---- Edit ----

export type EditResponseDTO = {
  removed: boolean;
};


// ---- image analyzer ----

export type ImageAttachment = {
  id: string;
  filename: string;
  imageBytes: string; // base64, no data: prefix
  imageType: string;
  previewUrl: string;
};

export type GeneratedImageDTO = {
  filename: string;
  url: string;
  created_at: number;
};

export type LibraryMediaDTO = GeneratedImageDTO & {
  kind: "image" | "file";
  folder: string;
};