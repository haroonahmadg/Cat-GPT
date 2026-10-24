import clsx from "clsx";
import { useState } from "react";

export function CatLogo({
  className,
  animated = false,
  label = "Nova AI",
  src = "/nova-ai.jpg",
}: {
  className?: string;
  animated?: boolean;
  label?: string;
  src?: string;
}) {
  const [useFallback, setUseFallback] = useState(false);
  return (
    <span
      role="img"
      aria-label={label}
      className={clsx(
        "inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-accent/15 text-accent",
        animated && "animate-pulse",
        className,
      )}
    >
      {useFallback ? (
        <svg viewBox="0 0 32 32" aria-hidden="true" className="h-6 w-6 fill-current">
          <path d="M7 14 5.5 5.5 13 10a13 13 0 0 1 6 0l7.5-4.5L25 14a10 10 0 1 1-18 0Z" />
          <path d="M11 17.25a1.25 1.25 0 1 0 0 2.5 1.25 1.25 0 0 0 0-2.5Zm10 0a1.25 1.25 0 1 0 0 2.5 1.25 1.25 0 0 0 0-2.5ZM16 21l-2 2h4l-2-2Z" className="fill-surface" />
        </svg>
      ) : (
        <img
          src={src}
          alt=""
          aria-hidden="true"
          className="h-full w-full rounded-full object-contain"
          onError={() => setUseFallback(true)}
        />
      )}
    </span>
  );
}
