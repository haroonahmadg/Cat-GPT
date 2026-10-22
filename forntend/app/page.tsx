"use client";

import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import { Volume2, Square, Loader2, Copy, Check, Pencil, RotateCcw, Eye, EyeOff, ImagePlus, MoreHorizontal, Pin, PinOff, Archive, ArchiveRestore, Trash2, Download, PencilLine, FolderClosed } from "lucide-react";
import { Sidebar, type AppView } from "@/components/Sidebar";
import { CatLogo } from "@/components/CatLogo";
import { HomeHero } from "@/components/HomeHero";
import { QuickActions } from "@/components/QuickActions";
import { ChatInput } from "@/components/ChatInput";
import { useChat } from "@/lib/useChat";
import { useAuth } from "@/lib/useAuth";
import { setUnauthorizedHandler } from "@/lib/api";
import { fetchGeneratedImage, generateImage, listLibraryMedia } from "@/lib/api";
import type { LibraryMediaDTO } from "@/lib/types";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import type { Components } from "react-markdown";

function markdownText(node: ReactNode): string {
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(markdownText).join("");
  if (node && typeof node === "object" && "props" in node) {
    return markdownText((node.props as { children?: ReactNode }).children);
  }
  return "";
}

function CodeBlock({
  language,
  code,
  children,
}: {
  language: string;
  code: string;
  children: ReactNode;
}) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard unavailable — fail quietly rather than disrupting
      // the chat over a non-critical convenience feature.
    }
  };

  return (
    <div className="my-2 overflow-hidden rounded-xl border border-surface-border bg-[#1e1e1e]">
      <div className="flex items-center justify-between border-b border-surface-border px-3 py-1.5 text-xs text-white/60">
        <span>{language || "code"}</span>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1 hover:text-white"
          aria-label="Copy code"
        >
          {copied ? <Check size={12} /> : <Copy size={12} />}
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <pre className="overflow-x-auto p-3 text-sm">{children}</pre>
    </div>
  );
}

const markdownComponents: Components = {
  pre({ children }) {
    const child = Array.isArray(children) ? children[0] : children;
    const className =
      child && typeof child === "object" && "props" in child
        ? (child.props as { className?: string }).className ?? ""
        : "";
    const language = /language-([\w-]+)/.exec(className)?.[1] ?? "";
    const code = markdownText(children).replace(/\n$/, "");
    return (
      <CodeBlock language={language || "code"} code={code}>
        {children}
      </CodeBlock>
    );
  },
  code({ className, children, ...props }) {
    return (
      <code className={className} {...props}>
        {children}
      </code>
    );
  },
};

// Shown inside a streaming assistant bubble before any real text has
// arrived yet — either live backend activity ("Searching the web...")
// or a generic "Thinking..." fallback, instead of a bare blinking
// cursor with an unrelated caption floating elsewhere on the page.
function ThinkingIndicator({ label }: { label: string | null }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-sm italic text-white/50">
      <Loader2 size={13} className="animate-spin" />
      {label ?? "Thinking…"}
    </span>
  );
}

export default function HomePage() {
  const {
    isAuthenticated,
    user,
    loginError,
    loggingIn,
    loginWithGoogle,
    loginWithEmail,
    registerWithEmail,
    logout,
    handleUnauthorized,
  } = useAuth();

  useEffect(() => {
    setUnauthorizedHandler(handleUnauthorized);
  }, [handleUnauthorized]);

  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [activeView, setActiveView] = useState<AppView>("chat");
  const [imagePrompt, setImagePrompt] = useState("");
  const [imageGenerating, setImageGenerating] = useState(false);
  const [imageError, setImageError] = useState<string | null>(null);
  const [libraryLoading, setLibraryLoading] = useState(false);
  const [libraryError, setLibraryError] = useState<string | null>(null);
  const [libraryItems, setLibraryItems] = useState<
    Array<LibraryMediaDTO & { previewUrl: string }>
  >([]);
  const [latestGeneratedUrl, setLatestGeneratedUrl] = useState<string | null>(null);
  const generatedPreviewUrlsRef = useRef<Set<string>>(new Set());
  const imageGenerationRef = useRef<AbortController | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [chatMenuOpen, setChatMenuOpen] = useState(false);
  const [editPrefill, setEditPrefill] = useState("");
  const [editSignal, setEditSignal] = useState(0);

  const {
    threads,
    threadsLoading,
    threadsError,
    refreshThreads,
    activeThreadId,
    messages,
    messagesLoading,
    isStreaming,
    resetChatState,
    activityLabel,
    sendError,
    selectThread,
    startNewChat,
    sendMessage,
    togglePin,
    removeThread,
    renameThread,
    toggleArchive,
    attachments,
    addAttachments,
    removeAttachment,
    imageAttachments,
    addImageAttachments,
    removeImageAttachment,
    isRecording,
    voiceError,
    startRecording,
    stopRecording,
    voiceStatus,
    speakMessage,
    exportCurrentThread,
    editLastMessage,
    retryLastMessage,
  } = useChat();

  const previousUserKey = useRef<string | null>(null);

  const clearLibraryPreviews = () => {
    imageGenerationRef.current?.abort();
    imageGenerationRef.current = null;
    setImageGenerating(false);
    setLibraryItems((previous) => {
      previous.forEach((item) => URL.revokeObjectURL(item.previewUrl));
      return [];
    });
    setLatestGeneratedUrl((previous) => {
      if (previous) {
        URL.revokeObjectURL(previous);
        generatedPreviewUrlsRef.current.delete(previous);
      }
      return null;
    });
  };

  const loadLibrary = async (signal?: AbortSignal) => {
    setLibraryLoading(true);
    setLibraryError(null);
    const createdUrls: string[] = [];
    try {
      const items = await listLibraryMedia(signal);
      const withPreviews = await Promise.all(
        items.map(async (item) => {
          const previewUrl = URL.createObjectURL(await fetchGeneratedImage(item.url, signal));
          createdUrls.push(previewUrl);
          return { ...item, previewUrl };
        }),
      );
      if (signal?.aborted) {
        createdUrls.forEach((url) => URL.revokeObjectURL(url));
        return;
      }
      setLibraryItems((previous) => {
        previous.forEach((item) => {
          URL.revokeObjectURL(item.previewUrl);
          generatedPreviewUrlsRef.current.delete(item.previewUrl);
        });
        withPreviews.forEach((item) => generatedPreviewUrlsRef.current.add(item.previewUrl));
        return withPreviews;
      });
    } catch (error) {
      createdUrls.forEach((url) => URL.revokeObjectURL(url));
      if (!signal?.aborted) {
        setLibraryError(error instanceof Error ? error.message : "Could not load your image library.");
      }
    } finally {
      if (!signal?.aborted) setLibraryLoading(false);
    }
  };

  useEffect(() => {
    if (activeView !== "library" || !isAuthenticated) return;
    const controller = new AbortController();
    void loadLibrary(controller.signal);
    return () => controller.abort();
  }, [activeView, isAuthenticated, user?.email]);

  const handleGenerateImage = async () => {
    const prompt = imagePrompt.trim();
    if (!prompt || prompt.length > 500 || imageGenerating) return;
    const controller = new AbortController();
    imageGenerationRef.current = controller;
    setImageGenerating(true);
    setImageError(null);
    try {
      const result = await generateImage(prompt, controller.signal);
      const previewUrl = URL.createObjectURL(
        await fetchGeneratedImage(result.url, controller.signal),
      );
      if (controller.signal.aborted) {
        URL.revokeObjectURL(previewUrl);
        return;
      }
      generatedPreviewUrlsRef.current.add(previewUrl);
      setLatestGeneratedUrl((previous) => {
        if (previous) {
          URL.revokeObjectURL(previous);
          generatedPreviewUrlsRef.current.delete(previous);
        }
        return previewUrl;
      });
      setImagePrompt("");
    } catch (error) {
      if (!controller.signal.aborted) {
        setImageError(error instanceof Error ? error.message : "Image generation failed. Please try again.");
      }
    } finally {
      if (imageGenerationRef.current === controller) {
        imageGenerationRef.current = null;
        setImageGenerating(false);
      }
    }
  };

  useEffect(() => {
    return () => {
      imageGenerationRef.current?.abort();
      generatedPreviewUrlsRef.current.forEach((url) => URL.revokeObjectURL(url));
      generatedPreviewUrlsRef.current.clear();
    };
  }, []);

  useEffect(() => {
    const sessionKey = user?.email ?? user?.name ?? (isAuthenticated ? "authenticated" : "logged-out");
    if (previousUserKey.current !== sessionKey) {
      previousUserKey.current = sessionKey;
      resetChatState();
      clearLibraryPreviews();
      setImageError(null);
      setLibraryError(null);
      void refreshThreads();
    }
  }, [isAuthenticated, refreshThreads, resetChatState, user?.email, user?.name]);

  const hasConversation = activeThreadId !== null || messages.length > 0;
  const activeThread = threads.find((thread) => thread.threadId === activeThreadId);
  const libraryFolders = libraryItems.reduce<Record<string, typeof libraryItems>>(
    (groups, item) => {
      (groups[item.folder] ??= []).push(item);
      return groups;
    },
    {},
  );

  const lastUserMessageId = [...messages].reverse().find((m) => m.role === "user")?.id;

  const handleCopy = async (id: string, text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedId(id);
      setTimeout(() => setCopiedId((current) => (current === id ? null : current)), 1500);
    } catch {
      // Clipboard permission denied or unsupported — fail quietly rather
      // than surfacing a disruptive error for a non-critical action.
    }
  };

  const handleEdit = async () => {
    const text = await editLastMessage();
    if (text !== null) {
      setEditPrefill(text);
      setEditSignal((s) => s + 1);
    }
  };

  const handleRetry = async () => {
    if (isStreaming) return;
    await retryLastMessage();
  };

  // ---- Auth gate: render before the authenticated app shell below ----

  if (isAuthenticated === null) {
    return (
      <div className="flex h-screen w-full items-center justify-center bg-surface text-white/40">
        Loading…
      </div>
    );
  }

  if (!isAuthenticated) {
    return (
      <LoginScreen
        onGoogleLogin={loginWithGoogle}
        onEmailLogin={loginWithEmail}
        onRegister={registerWithEmail}
        error={loginError}
        loading={loggingIn}
      />
    );
  }

  return (
    <div className="flex h-screen w-full overflow-hidden bg-surface text-white">
      <Sidebar
        collapsed={sidebarCollapsed}
        onToggleCollapsed={() => setSidebarCollapsed((prev) => !prev)}
        activeThreadId={activeThreadId}
        onSelectThread={selectThread}
        onNewChat={() => {
          setActiveView("chat");
          startNewChat();
        }}
        threads={threads}
        loading={threadsLoading}
        error={threadsError}
        onTogglePin={togglePin}
        onDeleteThread={removeThread}
        onRenameThread={renameThread}
        onToggleArchive={toggleArchive}
        onExport={exportCurrentThread}
        onLogout={logout}
        canExport={hasConversation}
        userDisplayName={user?.name?.trim() || user?.email || "Account"}
        activeView={activeView}
        onNavigate={setActiveView}
      />

      <div className="flex min-w-0 flex-1 flex-col">
        {activeView === "chat" && activeThread && (
          <div className="flex h-12 shrink-0 items-center justify-between border-b border-surface-border px-4">
            <h2 className="truncate text-sm font-medium text-white/80">{activeThread.title}</h2>
            <div className="relative">
              <button
                type="button"
                onClick={() => setChatMenuOpen((open) => !open)}
                aria-label="Chat options"
                aria-expanded={chatMenuOpen}
                className="rounded-lg p-2 text-white/60 hover:bg-surface-hover hover:text-white"
              >
                <MoreHorizontal size={18} />
              </button>
              {chatMenuOpen && (
                <div className="absolute right-0 top-full z-30 mt-1 w-48 rounded-xl border border-surface-border bg-surface-raised py-1 shadow-lg">
                  <button onClick={() => { void togglePin(activeThread.threadId); setChatMenuOpen(false); }} className="flex w-full items-center gap-2 px-3 py-2 text-left text-sm hover:bg-surface-hover">
                    {activeThread.pinned ? <PinOff size={14} /> : <Pin size={14} />}
                    {activeThread.pinned ? "Unpin chat" : "Pin chat"}
                  </button>
                  <button
                    onClick={() => {
                      const title = window.prompt("Rename chat", activeThread.title);
                      if (title?.trim()) void renameThread(activeThread.threadId, title);
                      setChatMenuOpen(false);
                    }}
                    className="flex w-full items-center gap-2 px-3 py-2 text-left text-sm hover:bg-surface-hover"
                  >
                    <PencilLine size={14} /> Rename
                  </button>
                  <button onClick={() => { void toggleArchive(activeThread.threadId); setChatMenuOpen(false); }} className="flex w-full items-center gap-2 px-3 py-2 text-left text-sm hover:bg-surface-hover">
                    {activeThread.archived ? <ArchiveRestore size={14} /> : <Archive size={14} />}
                    {activeThread.archived ? "Unarchive chat" : "Archive chat"}
                  </button>
                  <button onClick={() => { void exportCurrentThread(); setChatMenuOpen(false); }} className="flex w-full items-center gap-2 px-3 py-2 text-left text-sm hover:bg-surface-hover">
                    <Download size={14} /> Export chat
                  </button>
                  <div className="my-1 border-t border-surface-border" />
                  <button
                    onClick={() => {
                      if (window.confirm(`Delete "${activeThread.title}"? This can't be undone.`)) {
                        void removeThread(activeThread.threadId);
                      }
                      setChatMenuOpen(false);
                    }}
                    className="flex w-full items-center gap-2 px-3 py-2 text-left text-sm text-red-400 hover:bg-surface-hover"
                  >
                    <Trash2 size={14} /> Delete chat
                  </button>
                </div>
              )}
            </div>
          </div>
        )}
        {activeView === "images" ? (
          <main className="flex flex-1 flex-col items-center overflow-y-auto px-4 py-10">
            <section className="w-full max-w-2xl">
              <div className="mb-6 flex items-center gap-3">
                <CatLogo />
                <div>
                  <h1 className="text-xl font-semibold">Create an image</h1>
                  <p className="text-sm text-white/50">Describe what you want Cat GPT to make.</p>
                </div>
              </div>
              <label htmlFor="image-prompt" className="mb-2 block text-sm text-white/70">
                Image description
              </label>
              <textarea
                id="image-prompt"
                value={imagePrompt}
                onChange={(event) => setImagePrompt(event.target.value)}
                maxLength={500}
                rows={4}
                disabled={imageGenerating}
                placeholder="A curious cat exploring a moonlit garden, watercolor style…"
                className="w-full resize-y rounded-xl border border-surface-border bg-surface-raised px-4 py-3 text-sm text-white outline-none placeholder:text-white/30 focus:border-accent disabled:opacity-60"
              />
              <div className="mt-2 flex items-center justify-between text-xs text-white/40">
                <span>Maximum 500 characters</span>
                <span>{imagePrompt.length}/500</span>
              </div>
              <button
                type="button"
                onClick={handleGenerateImage}
                disabled={imageGenerating || !imagePrompt.trim() || imagePrompt.trim().length > 500}
                className="mt-4 inline-flex items-center gap-2 rounded-xl bg-accent px-4 py-2.5 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-50"
              >
                {imageGenerating ? <Loader2 size={16} className="animate-spin" /> : <ImagePlus size={16} />}
                {imageGenerating ? "Creating image…" : "Generate image"}
              </button>
              {imageError && <p role="alert" className="mt-3 text-sm text-red-400">{imageError}</p>}
              {imageGenerating && (
                <div className="mt-5 flex items-center gap-2 text-sm text-white/55" aria-live="polite">
                  <CatLogo animated className="h-7 w-7" />
                  Cat GPT is creating your image…
                </div>
              )}
              {latestGeneratedUrl && (
                <img
                  src={latestGeneratedUrl}
                  alt="Your generated image"
                  className="mt-6 max-h-[65vh] max-w-full rounded-xl border border-surface-border"
                />
              )}
            </section>
          </main>
        ) : activeView === "library" ? (
          <main className="flex-1 overflow-y-auto px-6 py-8">
            <h1 className="mb-1 text-xl font-semibold">Your Library</h1>
            <p className="mb-6 text-sm text-white/50">
              All your account’s media, including images and files, is collected here.
            </p>
            {libraryLoading && <p className="text-sm text-white/50">Loading your media…</p>}
            {libraryError && (
              <div role="alert" className="mb-4 text-sm text-red-400">
                {libraryError}{" "}
                <button type="button" onClick={() => void loadLibrary()} className="underline">Try again</button>
              </div>
            )}
            {!libraryLoading && !libraryError && libraryItems.length === 0 && (
              <p className="text-sm text-white/45">No media yet. Uploaded files and images will appear here.</p>
            )}
            {Object.entries(libraryFolders).map(([folder, items]) => (
              <section key={folder} className="mb-8">
                <h2 className="mb-3 flex items-center gap-2 text-sm font-medium text-white/75">
                    <FolderClosed size={15} /> {folder === "Generated" ? "Generated" : `Chat ${folder.slice(0, 8)}`}
                </h2>
                <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
                    {items.map((item) => (
                      <figure key={`${item.folder}/${item.filename}`} className="overflow-hidden rounded-xl border border-surface-border bg-surface-raised">
                        {item.kind === "image" ? (
                          <img src={item.previewUrl} alt={item.filename} className="aspect-square w-full object-cover" loading="lazy" />
                        ) : (
                          <a
                            href={item.previewUrl}
                            download={item.filename}
                            className="flex aspect-square items-center justify-center break-all p-4 text-center text-sm text-white/75 hover:bg-surface-hover"
                          >
                            {item.filename}
                          </a>
                        )}
                        <figcaption className="px-2 py-1.5 text-xs text-white/45">
                          {new Date(item.created_at * 1000).toLocaleString()}
                        </figcaption>
                      </figure>
                    ))}
                </div>
              </section>
            ))}
          </main>
        ) : !hasConversation ? (
          <main className="flex flex-1 flex-col items-center justify-center px-4">
            <div className="w-full max-w-2xl">
              <HomeHero />

              <ChatInput
                onSend={sendMessage}
                disabled={isStreaming}
                attachments={attachments}
                onAddFiles={addAttachments}
                onRemoveAttachment={removeAttachment}
                imageAttachments={imageAttachments}
                onAddImages={addImageAttachments}
                onRemoveImageAttachment={removeImageAttachment}
                isRecording={isRecording}
                onStartRecording={startRecording}
                onStopRecording={stopRecording}
                voiceError={voiceError}
                prefillText={editPrefill}
                prefillKey={editSignal}
              />

              <QuickActions onSelect={sendMessage} />

              {sendError && (
                <p
                  role="alert"
                  className="mt-4 rounded-lg border border-red-400/30 bg-red-400/10 px-3 py-2 text-center text-xs text-red-400"
                >
                  {sendError}
                </p>
              )}
            </div>
          </main>
        ) : (
          <main className="flex flex-1 flex-col overflow-hidden px-4">
            <div className="mx-auto flex w-full max-w-2xl flex-1 flex-col overflow-hidden">
              <div className="flex-1 overflow-y-auto thin-scrollbar py-4">
                {messagesLoading && (
                  <p className="text-center text-xs text-white/40">
                    Loading conversation…
                  </p>
                )}

                {!messagesLoading &&
                  messages.map((message) => {
                    const isListening = voiceStatus?.messageId === message.id;
                    const isLastUserMessage =
                      message.role === "user" && message.id === lastUserMessageId;
                    const isUser = message.role === "user";

                    // Nothing has streamed in for this message yet —
                    // show the inline thinking/activity indicator instead
                    // of an empty bubble with just a cursor.
                    const showThinking = message.streaming && !message.content;

                    return (
                      <div
                        key={message.id}
                        className="group mb-3 flex w-full flex-col"
                      >
                        <div
                          className={
                            isUser
                              ? "max-w-[80%] self-end rounded-2xl bg-accent px-4 py-2 text-sm text-white"
                              : "max-w-[80%] self-start rounded-2xl bg-surface-raised px-4 py-2 text-sm text-white/90"
                          }
                        >
                        <div className="prose prose-invert prose-base max-w-none [&_p]:my-1 [&_ul]:my-1 [&_ol]:my-1">
                          {showThinking ? (
                            <span className="inline-flex items-center gap-2">
                              <CatLogo animated className="h-7 w-7" />
                              <ThinkingIndicator label={activityLabel} />
                            </span>
                          ) : (
                            <>
                              {message.streaming && (
                                <span
                                  className="mb-2 inline-flex items-center gap-2"
                                  aria-label="Cat GPT is responding"
                                >
                                  <CatLogo animated className="h-7 w-7" />
                                  <span className="text-xs font-medium text-white/55">Cat GPT</span>
                                </span>
                              )}
                              {message.content && isUser ? (
                                <p className="whitespace-pre-wrap">{message.content}</p>
                              ) : message.content ? (
                                <ReactMarkdown
                                  remarkPlugins={[remarkGfm]}
                                  rehypePlugins={[rehypeHighlight]}
                                  components={markdownComponents}
                                >
                                  {message.content}
                                </ReactMarkdown>
                              ) : null}
                              {message.streaming && (
                                <span className="ml-0.5 inline-block h-4 w-[2px] animate-pulse bg-white/70 align-middle" />
                              )}{message.imageUrl && (
  <img
    src={message.imageUrl}
    alt="Generated image"
    className="mt-2 max-w-full rounded-xl border border-surface-border"
  />
)}
                            </>
                          )}
                        </div>
                        </div>

                        {!message.streaming && isUser && (
                          <div className="mt-1 flex max-w-[80%] self-end gap-1">
                            <button
                              onClick={() => handleCopy(message.id, message.content)}
                              className="rounded p-1.5 text-white/45 hover:bg-surface-hover hover:text-white focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
                              aria-label="Copy message"
                              title="Copy message"
                            >
                              {copiedId === message.id ? <Check size={14} /> : <Copy size={14} />}
                            </button>
                            {isLastUserMessage && (
                              <>
                                <button
                                  onClick={handleRetry}
                                  disabled={isStreaming}
                                  className="rounded p-1.5 text-white/45 hover:bg-surface-hover hover:text-white disabled:cursor-not-allowed disabled:opacity-40 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
                                  aria-label="Retry message"
                                  title="Retry message"
                                >
                                  <RotateCcw size={14} />
                                </button>
                                <button
                                  onClick={handleEdit}
                                  disabled={isStreaming}
                                  className="rounded p-1.5 text-white/45 hover:bg-surface-hover hover:text-white disabled:cursor-not-allowed disabled:opacity-40 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
                                  aria-label="Edit message"
                                  title="Edit message"
                                >
                                  <Pencil size={14} />
                                </button>
                              </>
                            )}
                          </div>
                        )}
                        {!message.streaming && message.role === "assistant" && (message.content || message.imageUrl) && (
                          <div className="mt-1 flex max-w-[80%] self-start gap-2">
                            <button
                              onClick={() => handleCopy(message.id, message.content)}
                              disabled={!message.content}
                              className="inline-flex items-center gap-1 rounded px-2 py-1 text-xs text-white/50 hover:bg-surface hover:text-white disabled:cursor-not-allowed disabled:opacity-40 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
                              aria-label="Copy response"
                              title="Copy response"
                            >
                              {copiedId === message.id ? <Check size={13} /> : <Copy size={13} />}
                            </button>
                            <button
                              onClick={() => speakMessage(message.id, message.content)}
                              className="inline-flex items-center gap-1 rounded px-2 py-1 text-xs text-white/50 hover:bg-surface hover:text-white focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
                              aria-label={isListening ? "Stop listening" : "Listen to response"}
                              title={isListening ? "Stop listening" : "Listen to response"}
                            >
                              {isListening && voiceStatus?.state === "loading" ? (
                                <Loader2 size={13} className="animate-spin" />
                              ) : isListening && voiceStatus?.state === "playing" ? (
                                <Square size={13} />
                              ) : (
                                <Volume2 size={13} />
                              )}
                            </button>
                          </div>
                        )}
                      </div>
                    );
                  })}

                {sendError && (
                  <p
                    role="alert"
                    className="mb-2 rounded-lg border border-red-400/30 bg-red-400/10 px-3 py-2 text-xs text-red-400"
                  >
                    {sendError}
                  </p>
                )}
              </div>

              <div className="pb-4">
                <ChatInput
                  onSend={sendMessage}
                  disabled={isStreaming}
                  attachments={attachments}
                  onAddFiles={addAttachments}
                  onRemoveAttachment={removeAttachment}
                  imageAttachments={imageAttachments}
                  onAddImages={addImageAttachments}
                  onRemoveImageAttachment={removeImageAttachment}
                  isRecording={isRecording}
                  onStartRecording={startRecording}
                  onStopRecording={stopRecording}
                  voiceError={voiceError}
                  prefillText={editPrefill}
                  prefillKey={editSignal}
                />
              </div>
            </div>
          </main>
        )}
      </div>
    </div>
  );
}

function LoginScreen({
  onGoogleLogin,
  onEmailLogin,
  onRegister,
  error,
  loading,
}: {
  onGoogleLogin: (credential: string) => void;
  onEmailLogin: (email: string, password: string) => void;
  onRegister: (email: string, password: string, name?: string) => void;
  error: string | null;
  loading: boolean;
}) {
  const buttonRef = useRef<HTMLDivElement>(null);
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [name, setName] = useState("");
  const [googleButtonError, setGoogleButtonError] = useState<string | null>(null);

  useEffect(() => {
    const scriptId = "google-identity-services";
    const googleClientId = process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID;

    if (!googleClientId) {
      setGoogleButtonError("Google sign-in is not configured for this app.");
      return;
    }

    function renderButton() {
      const google = (window as any).google;
      if (!google?.accounts?.id || !buttonRef.current) {
        setGoogleButtonError("Google sign-in could not be loaded. Refresh and try again.");
        return;
      }

      google.accounts.id.initialize({
        client_id: googleClientId,
        callback: (response: { credential?: string }) => {
          if (!response.credential) {
            setGoogleButtonError("Google did not return a sign-in credential. Please try again.");
            return;
          }
          setGoogleButtonError(null);
          onGoogleLogin(response.credential);
        },
      });

      google.accounts.id.renderButton(buttonRef.current, {
        theme: "filled_black",
        size: "large",
        shape: "pill",
      });
    }

    const existingScript = document.getElementById(scriptId) as HTMLScriptElement | null;
    if (existingScript) {
      if ((window as any).google?.accounts?.id) {
        renderButton();
      } else {
        existingScript.addEventListener("load", renderButton, { once: true });
        existingScript.addEventListener(
          "error",
          () => setGoogleButtonError("Could not load Google sign-in. Check your connection and try again."),
          { once: true },
        );
      }
      return () => existingScript.removeEventListener("load", renderButton);
    }

    const script = document.createElement("script");
    script.id = scriptId;
    script.src = "https://accounts.google.com/gsi/client";
    script.async = true;
    script.onload = renderButton;
    script.onerror = () => {
      setGoogleButtonError("Could not load Google sign-in. Check your connection and try again.");
    };
    document.body.appendChild(script);
  }, [onGoogleLogin]);

  const handleSubmit = () => {
    const trimmedEmail = email.trim();
    const trimmedPassword = password.trim();

    if (!trimmedEmail || !trimmedPassword) {
      return;
    }

    if (mode === "register") {
      onRegister(trimmedEmail, trimmedPassword, name.trim() || undefined);
      return;
    }

    onEmailLogin(trimmedEmail, trimmedPassword);
  };

  return (
    <div className="flex h-screen w-full items-center justify-center bg-surface text-white">
      <div className="w-full max-w-md rounded-2xl border border-surface-border bg-surface-raised p-6 text-center">
        <h1 className="mb-2 flex items-center justify-center gap-2 text-lg font-semibold">
          <CatLogo className="h-7 w-7" />
          Cat GPT
        </h1>
        <p className="mb-5 text-sm text-white/60">Sign in to continue</p>

        <div className="mb-4 grid grid-cols-2 rounded-xl border border-surface-border bg-surface p-1">
          <button
            type="button"
            onClick={() => setMode("login")}
            className={mode === "login" ? "rounded-lg bg-accent px-3 py-2 text-sm font-medium text-white" : "px-3 py-2 text-sm text-white/60"}
          >
            Sign in
          </button>
          <button
            type="button"
            onClick={() => setMode("register")}
            className={mode === "register" ? "rounded-lg bg-accent px-3 py-2 text-sm font-medium text-white" : "px-3 py-2 text-sm text-white/60"}
          >
            Create account
          </button>
        </div>

        <div className="space-y-3 text-left">
          {mode === "register" && (
            <label className="block">
              <span className="mb-1 block text-xs text-white/60">Name</span>
              <input
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Jane Doe"
                className="w-full rounded-xl border border-surface-border bg-surface px-3 py-2 text-sm text-white placeholder:text-white/30 outline-none focus:border-accent"
              />
            </label>
          )}

          <label className="block">
            <span className="mb-1 block text-xs text-white/60">Email</span>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@example.com"
              className="w-full rounded-xl border border-surface-border bg-surface px-3 py-2 text-sm text-white placeholder:text-white/30 outline-none focus:border-accent"
            />
          </label>

          <label className="block">
            <span className="mb-1 block text-xs text-white/60">Password</span>
            <span className="relative block">
              <input
                type={showPassword ? "text" : "password"}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder={mode === "register" ? "At least 8 characters" : "Your password"}
                autoComplete={mode === "register" ? "new-password" : "current-password"}
                className="w-full rounded-xl border border-surface-border bg-surface px-3 py-2 pr-10 text-sm text-white placeholder:text-white/30 outline-none focus:border-accent"
              />
              <button
                type="button"
                onClick={() => setShowPassword((visible) => !visible)}
                aria-label={showPassword ? "Hide password" : "Show password"}
                aria-pressed={showPassword}
                className="absolute inset-y-0 right-2 flex items-center px-1 text-white/50 hover:text-white focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
              >
                {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </span>
          </label>

          <button
            type="button"
            onClick={handleSubmit}
            disabled={loading || !email.trim() || !password.trim()}
            className="w-full rounded-xl bg-accent px-3 py-2.5 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-60"
          >
            {loading ? (mode === "register" ? "Creating account…" : "Signing in…") : mode === "register" ? "Create account" : "Sign in"}
          </button>
        </div>

        {process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID ? (
          <>
            <div className="my-4 flex items-center gap-3 text-xs uppercase tracking-[0.2em] text-white/35">
              <span className="h-px flex-1 bg-surface-border" />
              <span>or</span>
              <span className="h-px flex-1 bg-surface-border" />
            </div>
            <div ref={buttonRef} className="flex justify-center" />
            <p className="mt-2 text-center text-xs text-white/45">
              Continue with your Google account. Cat GPT will sign you into the
              account for that verified email, or create one automatically if
              you&apos;re new.
            </p>
          </>
        ) : (
          <p role="status" className="mt-3 text-xs text-amber-300/80">
            Google sign-in is unavailable because it is not configured.
          </p>
        )}

        {loading && <p className="mt-3 text-xs text-white/50">Please wait…</p>}

        {googleButtonError && process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID && (
          <p role="alert" className="mt-3 text-xs text-red-400">
            {googleButtonError}
          </p>
        )}

        {error && (
          <p role="alert" className="mt-3 text-xs text-red-400">
            {error}
          </p>
        )}
      </div>
    </div>
  );
}