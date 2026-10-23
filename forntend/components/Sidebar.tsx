"use client";

import { useEffect, useRef, useState } from "react";
import {
  Image as ImageIcon,
  Library,
  Clock,
  Terminal,
  MoreHorizontal,
  PanelLeftClose,
  PanelLeftOpen,
  Search,
  SquarePen,
  Pin,
  PinOff,
  Pencil,
  Trash2,
  Archive,
  ArchiveRestore,
  ChevronDown,
  ChevronRight,
  Download,
  LogOut,
} from "lucide-react";
import clsx from "clsx";
import { IconButton } from "@/components/ui/icon-button";
import { ThreadSummary } from "@/lib/types";
import { CatLogo } from "@/components/CatLogo";

export type AppView = "chat" | "images" | "library" | "scheduled" | "codex";

const NAV_ITEMS = [
  { label: "Images", icon: ImageIcon, view: "images" as const },
  { label: "Library", icon: Library, view: "library" as const },
  { label: "Scheduled", icon: Clock, view: "scheduled" as const },
  { label: "Codex", icon: Terminal, view: "codex" as const },
];

interface SidebarProps {
  collapsed: boolean;
  onToggleCollapsed: () => void;
  activeThreadId: string | null;
  onSelectThread: (threadId: string) => void;
  onNewChat: () => void;
  threads: ThreadSummary[];
  loading: boolean;
  error: string | null;
  onTogglePin: (threadId: string) => void;
  onDeleteThread: (threadId: string) => void;
  onRenameThread: (threadId: string, title: string) => void;
  onToggleArchive: (threadId: string) => void;
  onDownloadThread: (threadId: string) => void;
  onExport: () => void;
  onLogout: () => void;
  canExport: boolean;
  userDisplayName: string;
  activeView: AppView;
  onNavigate: (view: AppView) => void;
}

export function Sidebar({
  collapsed,
  onToggleCollapsed,
  activeThreadId,
  onSelectThread,
  onNewChat,
  threads,
  loading,
  error,
  onTogglePin,
  onDeleteThread,
  onRenameThread,
  onToggleArchive,
  onDownloadThread,
  onExport,
  onLogout,
  canExport,
  userDisplayName,
  activeView,
  onNavigate,
}: SidebarProps) {
  const [searchOpen, setSearchOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");

  const query = searchQuery.trim().toLowerCase();
  const matches = (t: ThreadSummary) =>
    !query || t.title.toLowerCase().includes(query);

  const pinned = threads.filter((chat) => chat.pinned && !chat.archived && matches(chat));
  const recent = threads.filter((chat) => !chat.pinned && !chat.archived && matches(chat));
  const archived = threads.filter((chat) => chat.archived && matches(chat));

  if (collapsed) {
    return (
      <aside className="flex h-full w-14 flex-col items-center gap-2 border-r border-surface-border bg-surface py-3">
        <IconButton onClick={onToggleCollapsed} aria-label="Expand sidebar">
          <PanelLeftOpen size={18} />
        </IconButton>
        <IconButton onClick={onNewChat} aria-label="New chat">
          <SquarePen size={18} />
        </IconButton>
        <IconButton onClick={() => onNavigate("images")} aria-label="Images">
          <ImageIcon size={18} />
        </IconButton>
        <IconButton onClick={() => onNavigate("library")} aria-label="Library">
          <Library size={18} />
        </IconButton>
        <IconButton onClick={() => onNavigate("scheduled")} aria-label="Scheduled">
          <Clock size={18} />
        </IconButton>
        <IconButton onClick={() => onNavigate("codex")} aria-label="Codex">
          <Terminal size={18} />
        </IconButton>
      </aside>
    );
  }

  return (
    <aside className="flex h-full w-64 flex-shrink-0 flex-col border-r border-surface-border bg-surface">
      <div className="flex items-center justify-between px-3 py-3">
        <span className="flex items-center gap-2 text-sm font-semibold tracking-wide">
          <CatLogo className="h-7 w-7" />
        </span>
        <div className="flex items-center gap-1">
          <IconButton aria-label="Search" onClick={() => setSearchOpen((p) => !p)}>
            <Search size={16} />
          </IconButton>
          <IconButton onClick={onToggleCollapsed} aria-label="Collapse sidebar">
            <PanelLeftClose size={16} />
          </IconButton>
        </div>
      </div>

      <div className="px-2">
        <button
          onClick={() => {
            onNavigate("chat");
            onNewChat();
          }}
          className="flex w-full items-center gap-2 rounded-lg px-2 py-2 text-sm text-white/90 hover:bg-surface-hover"
        >
          <SquarePen size={16} />
          New chat
        </button>
      </div>

      {searchOpen && (
        <div className="px-2 pt-2">
          <input
            autoFocus
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search chats..."
            className="w-full rounded-lg border border-surface-border bg-surface px-2 py-1.5 text-sm text-white outline-none focus:ring-1 focus:ring-accent"
          />
        </div>
      )}

      <nav className="mt-1 flex flex-col gap-0.5 px-2">
        {NAV_ITEMS.map(({ label, icon: Icon, view }) => (
          <button
            key={label}
            onClick={() => onNavigate(view)}
            aria-current={view === activeView ? "page" : undefined}
            className={`flex items-center gap-2 rounded-lg px-2 py-2 text-sm hover:bg-surface-hover hover:text-white ${
              view === activeView
                ? "bg-surface-hover text-white"
                : "text-white/80"
            }`}
          >
            <Icon size={16} />
            <span className="flex items-center gap-2">
              {label}
            </span>
          </button>
        ))}
      </nav>

      <div className="mt-4 flex-1 overflow-y-auto px-2 thin-scrollbar">
        {loading && <p className="px-2 py-2 text-xs text-white/40">Loading chats…</p>}
        {!loading && error && <p className="px-2 py-2 text-xs text-red-400">{error}</p>}
        {!loading && !error && threads.length === 0 && (
          <p className="px-2 py-2 text-xs text-white/40">No conversations yet.</p>
        )}
        {!loading && !error && query && pinned.length === 0 && recent.length === 0 && archived.length === 0 && (
          <p className="px-2 py-2 text-xs text-white/40">No chats match "{searchQuery}".</p>
        )}
        {!loading && !error && (
          <>
            {pinned.length > 0 && (
              <ThreadGroup
                label="Pinned"
                threads={pinned}
                activeThreadId={activeThreadId}
                onSelectThread={onSelectThread}
                onTogglePin={onTogglePin}
                onDeleteThread={onDeleteThread}
                onRenameThread={onRenameThread}
                onToggleArchive={onToggleArchive}
                onDownloadThread={onDownloadThread}
              />
            )}
            <ThreadGroup
              label="Recents"
              threads={recent}
              activeThreadId={activeThreadId}
              onSelectThread={onSelectThread}
              onTogglePin={onTogglePin}
              onDeleteThread={onDeleteThread}
              onRenameThread={onRenameThread}
              onToggleArchive={onToggleArchive}
              onDownloadThread={onDownloadThread}
            />
            {archived.length > 0 && (
              <ArchivedGroup
                threads={archived}
                activeThreadId={activeThreadId}
                onSelectThread={onSelectThread}
                onTogglePin={onTogglePin}
                onDeleteThread={onDeleteThread}
                onRenameThread={onRenameThread}
                onToggleArchive={onToggleArchive}
                onDownloadThread={onDownloadThread}
              />
            )}
          </>
        )}
      </div>

      <ProfileMenu onExport={onExport} onLogout={onLogout} canExport={canExport} userDisplayName={userDisplayName} />
    </aside>
  );
}

// ============================================================
// PROFILE MENU
//
// Minimal by design: just the user chip, Export chat, and Log out.
// No Plus/Personalization/Profile/Settings/Help — this app doesn't
// have those features, so they're deliberately not added here rather
// than replicated from a reference screenshot that doesn't apply.
// ============================================================

function ProfileMenu({
  onExport,
  onLogout,
  canExport,
  userDisplayName,
}: {
  onExport: () => void;
  onLogout: () => void;
  canExport: boolean;
  userDisplayName: string;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;

    const handleClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };

    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [open]);

  return (
    <div ref={ref} className="relative border-t border-surface-border p-3">
      {open && (
        <div className="absolute bottom-full left-3 right-3 z-20 mb-1 overflow-hidden rounded-xl border border-surface-border bg-surface-raised py-1 shadow-lg">
          <button
            onClick={() => {
              if (!canExport) return;
              setOpen(false);
              onExport();
            }}
            disabled={!canExport}
            title={canExport ? "Export this conversation as Markdown" : "Open a conversation first"}
            className="flex w-full items-center gap-2 px-3 py-2 text-left text-sm text-white/90 hover:bg-surface-hover disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:bg-transparent"
          >
            <Download size={14} />
            Export chat
          </button>

          <div className="my-1 border-t border-surface-border" />

          <button
            onClick={() => {
              setOpen(false);
              onLogout();
            }}
            className="flex w-full items-center gap-2 px-3 py-2 text-left text-sm text-white/90 hover:bg-surface-hover"
          >
            <LogOut size={14} />
            Log out
          </button>
        </div>
      )}

      <button
        onClick={() => setOpen((prev) => !prev)}
        aria-label="Account menu"
        aria-expanded={open}
        className={clsx(
          "flex w-full items-center gap-2 rounded-lg px-1 py-1 text-left hover:bg-surface-hover",
          open && "bg-surface-hover",
        )}
      >
        <div className="flex h-7 w-7 items-center justify-center rounded-full bg-accent text-xs font-semibold">
          {userDisplayName.trim().charAt(0).toUpperCase() || "A"}
        </div>
        <div className="flex flex-col leading-tight">
          <span className="text-sm">{userDisplayName}</span>
          <span className="text-xs text-white/50">Account</span>
        </div>
      </button>
    </div>
  );
}

function ArchivedGroup({
  threads,
  activeThreadId,
  onSelectThread,
  onTogglePin,
  onDeleteThread,
  onRenameThread,
  onToggleArchive,
  onDownloadThread,
}: {
  threads: ThreadSummary[];
  activeThreadId: string | null;
  onSelectThread: (threadId: string) => void;
  onTogglePin: (threadId: string) => void;
  onDeleteThread: (threadId: string) => void;
  onRenameThread: (threadId: string, title: string) => void;
  onToggleArchive: (threadId: string) => void;
  onDownloadThread: (threadId: string) => void;
}) {
  const [open, setOpen] = useState(false);

  return (
    <div className="mb-3">
      <button
        onClick={() => setOpen((prev) => !prev)}
        className="flex w-full items-center gap-1 px-2 pb-1 text-xs font-medium text-white/40 hover:text-white/60"
      >
        {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        Archived ({threads.length})
      </button>

      {open && (
        <ThreadGroup
          label=""
          threads={threads}
          activeThreadId={activeThreadId}
          onSelectThread={onSelectThread}
          onTogglePin={onTogglePin}
          onDeleteThread={onDeleteThread}
          onRenameThread={onRenameThread}
          onToggleArchive={onToggleArchive}
          onDownloadThread={onDownloadThread}
        />
      )}
    </div>
  );
}

function ThreadGroup({
  label,
  threads,
  activeThreadId,
  onSelectThread,
  onTogglePin,
  onDeleteThread,
  onRenameThread,
  onToggleArchive,
  onDownloadThread,
}: {
  label: string;
  threads: ThreadSummary[];
  activeThreadId: string | null;
  onSelectThread: (threadId: string) => void;
  onTogglePin: (threadId: string) => void;
  onDeleteThread: (threadId: string) => void;
  onRenameThread: (threadId: string, title: string) => void;
  onToggleArchive: (threadId: string) => void;
  onDownloadThread: (threadId: string) => void;
}) {
  const [openMenuId, setOpenMenuId] = useState<string | null>(null);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editValue, setEditValue] = useState("");

  if (threads.length === 0) return null;

  const handleDelete = (thread: ThreadSummary) => {
    setOpenMenuId(null);
    const confirmed = window.confirm(`Delete "${thread.title}"? This can't be undone.`);
    if (confirmed) onDeleteThread(thread.threadId);
  };

  const startEditing = (thread: ThreadSummary) => {
    setOpenMenuId(null);
    setEditingId(thread.threadId);
    setEditValue(thread.title);
  };

  const commitEdit = (threadId: string) => {
    const trimmed = editValue.trim();
    setEditingId(null);
    if (trimmed) onRenameThread(threadId, trimmed);
  };

  return (
    <div className="mb-3">
      {label && <p className="px-2 pb-1 text-xs font-medium text-white/40">{label}</p>}
      <div className="flex flex-col gap-0.5">
        {threads.map((thread) => (
          <div
            key={thread.threadId}
            className={clsx(
              "group relative flex items-center gap-1 rounded-lg px-2 py-1.5 hover:bg-surface-hover",
              activeThreadId === thread.threadId && "bg-surface-hover",
            )}
          >
            {editingId === thread.threadId ? (
              <input
                autoFocus
                value={editValue}
                onChange={(e) => setEditValue(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") commitEdit(thread.threadId);
                  if (e.key === "Escape") setEditingId(null);
                }}
                onBlur={() => commitEdit(thread.threadId)}
                maxLength={200}
                className="flex-1 rounded bg-surface px-1.5 py-0.5 text-sm text-white outline-none ring-1 ring-accent"
              />
            ) : (
              <button
                onClick={() => onSelectThread(thread.threadId)}
                className={clsx(
                  "flex flex-1 items-center gap-1.5 truncate text-left text-sm text-white/80 hover:text-white",
                  activeThreadId === thread.threadId && "text-white",
                )}
                title={thread.title}
              >
                <span className="truncate">{thread.title}</span>
              </button>
            )}

            {editingId !== thread.threadId && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  setOpenMenuId((prev) => (prev === thread.threadId ? null : thread.threadId));
                }}
                aria-label="Chat options"
                className="flex-shrink-0 rounded p-0.5 text-white/40 opacity-0 hover:bg-surface hover:text-white group-hover:opacity-100"
              >
                <MoreHorizontal size={15} />
              </button>
            )}

            {openMenuId === thread.threadId && (
              <ThreadMenu
                thread={thread}
                onClose={() => setOpenMenuId(null)}
                onTogglePin={onTogglePin}
                onDelete={handleDelete}
                onRename={() => startEditing(thread)}
                onToggleArchive={() => onToggleArchive(thread.threadId)}
                onDownload={() => onDownloadThread(thread.threadId)}
              />
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

function ThreadMenu({
  thread,
  onClose,
  onTogglePin,
  onDelete,
  onRename,
  onToggleArchive,
  onDownload,
}: {
  thread: ThreadSummary;
  onClose: () => void;
  onTogglePin: (threadId: string) => void;
  onDelete: (thread: ThreadSummary) => void;
  onRename: () => void;
  onToggleArchive: () => void;
  onDownload: () => void;
}) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) onClose();
    };
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [onClose]);

  return (
    <div
      ref={ref}
      className="absolute right-0 top-full z-20 mt-1 w-44 rounded-xl border border-surface-border bg-surface-raised py-1 shadow-lg"
    >
      <button
        onClick={() => {
          onTogglePin(thread.threadId);
          onClose();
        }}
        className="flex w-full items-center gap-2 px-3 py-2 text-left text-sm text-white/90 hover:bg-surface-hover"
      >
        {thread.pinned ? <PinOff size={14} /> : <Pin size={14} />}
        {thread.pinned ? "Unpin chat" : "Pin chat"}
      </button>

      <button
        onClick={onRename}
        className="flex w-full items-center gap-2 px-3 py-2 text-left text-sm text-white/90 hover:bg-surface-hover"
      >
        <Pencil size={14} />
        Rename
      </button>

      <button
        onClick={() => {
          onToggleArchive();
          onClose();
        }}
        className="flex w-full items-center gap-2 px-3 py-2 text-left text-sm text-white/90 hover:bg-surface-hover"
      >
        {thread.archived ? <ArchiveRestore size={14} /> : <Archive size={14} />}
        {thread.archived ? "Unarchive" : "Archive"}
      </button>

      <div className="my-1 border-t border-surface-border" />

      <button
        onClick={() => {
          onDownload();
          onClose();
        }}
        className="flex w-full items-center gap-2 px-3 py-2 text-left text-sm text-white/90 hover:bg-surface-hover"
      >
        <Download size={14} />
        Download .md
      </button>

      <div className="my-1 border-t border-surface-border" />

      <button
        onClick={() => onDelete(thread)}
        className="flex w-full items-center gap-2 px-3 py-2 text-left text-sm text-red-400 hover:bg-surface-hover"
      >
        <Trash2 size={14} />
        Delete
      </button>
    </div>
  );
}