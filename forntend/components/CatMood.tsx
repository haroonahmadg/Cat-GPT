"use client";

export type CatMoodState = "idle" | "thinking" | "working" | "happy" | "error";

const labels: Record<CatMoodState, string> = {
  idle: "Ready",
  thinking: "Thinking",
  working: "Working",
  happy: "Done",
  error: "Error",
};

export function CatMood({ mood }: { mood: CatMoodState }) {
  return (
    <div
      className={`cat-mood cat-mood-${mood}`}
      role="status"
      aria-label={`Cat GPT: ${labels[mood]}`}
    >
      <svg viewBox="0 0 48 48" aria-hidden="true" className="h-9 w-9">
        <path className="cat-tail" d="M37 34c8 0 8-10 3-11" />
        <path d="M10 18 8 6l12 6m8 0L40 6l-2 12" />
        <path d="M8 25c0-8 7-13 16-13s16 5 16 13v5c0 8-7 13-16 13S8 38 8 30z" />
        <path className="cat-face" d="M17 27v1m14-1v1" />
        <path className="cat-mouth" d="M21 34q3 3 6 0" />
        {mood === "thinking" && <circle className="cat-thought" cx="42" cy="10" r="2" />}
        {mood === "working" && <path className="cat-spark" d="m42 7 1 3 3 1-3 1-1 3-1-3-3-1 3-1z" />}
        {mood === "happy" && <path className="cat-cheer" d="m5 16 2-3 2 3m28 0 2-3 2 3" />}
        {mood === "error" && <path className="cat-error" d="m20 27 3 3m0-3-3 3m7-3 3 3m0-3-3 3" />}
      </svg>
      <span className="text-xs text-white/55">{labels[mood]}</span>
    </div>
  );
}
