import { ButtonHTMLAttributes, forwardRef } from "react";
import clsx from "clsx";

interface IconButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  active?: boolean;
}

/**
 * Circular icon-only button, matching the sidebar/top-bar icon buttons
 * (search, collapse, mic, etc.) in the reference screenshot.
 */
export const IconButton = forwardRef<HTMLButtonElement, IconButtonProps>(
  ({ className, active = false, ...props }, ref) => {
    return (
      <button
        ref={ref}
        type="button"
        className={clsx(
          "inline-flex h-9 w-9 items-center justify-center rounded-full text-white/80 transition-colors hover:bg-surface-hover hover:text-white disabled:opacity-40 disabled:pointer-events-none",
          active && "bg-surface-hover text-white",
          className,
        )}
        {...props}
      />
    );
  },
);

IconButton.displayName = "IconButton";
