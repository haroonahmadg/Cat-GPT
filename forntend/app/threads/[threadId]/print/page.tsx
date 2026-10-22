"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { fetchThreadExportMarkdown, setUnauthorizedHandler } from "@/lib/api";
import { useAuth } from "@/lib/useAuth";

export default function PrintThreadPage() {
  const { threadId } = useParams<{ threadId: string }>();
  const [markdown, setMarkdown] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { handleUnauthorized } = useAuth();

  useEffect(() => {
    setUnauthorizedHandler(handleUnauthorized);
  }, [handleUnauthorized]);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    void fetchThreadExportMarkdown(threadId)
      .then((content) => {
        if (!cancelled) setMarkdown(content);
      })
      .catch((reason: unknown) => {
        if (!cancelled) {
          setError(reason instanceof Error ? reason.message : "Could not load this transcript.");
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [threadId]);

  return (
    <main className="print-page min-h-screen bg-white px-6 py-8 text-zinc-900">
      <div className="no-print mx-auto mb-6 flex max-w-3xl justify-end">
        <button
          type="button"
          onClick={() => window.print()}
          disabled={loading || Boolean(error)}
          className="rounded-lg bg-zinc-900 px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
        >
          Print / Save as PDF
        </button>
      </div>
      <article className="print-transcript prose mx-auto max-w-3xl">
        {loading && <p>Loading transcript…</p>}
        {error && (
          <div role="alert">
            <p>{error}</p>
            <button type="button" onClick={() => window.location.reload()} className="underline">
              Try again
            </button>
          </div>
        )}
        {!loading && !error && (
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{markdown}</ReactMarkdown>
        )}
      </article>
    </main>
  );
}
