import { ButtonHTMLAttributes, forwardRef } from "react";
import clsx from "clsx";

type ButtonVariant = "default" | "ghost" | "pill";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
}

const variantClasses: Record<ButtonVariant, string> = {
  default:
    "bg-white text-black hover:bg-white/90 rounded-lg px-4 py-2",
  ghost:
    "bg-transparent text-white/80 hover:bg-surface-hover rounded-lg px-3 py-2",
  pill: "bg-surface-hover text-white rounded-pill px-4 py-1.5 text-sm",
};

/**
 * Minimal button primitive matching shadcn/ui's default styling
 * conventions, hand-written so this project has no dependency on the
 * shadcn CLI/registry. Visually interchangeable with a real shadcn
 * <Button /> if that's introduced later.
 */
export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = "default", ...props }, ref) => {
    return (
      <button
        ref={ref}
        className={clsx(
          "inline-flex items-center justify-center gap-2 text-sm font-medium transition-colors disabled:opacity-50 disabled:pointer-events-none",
          variantClasses[variant],
          className,
        )}
        {...props}
      />
    );
  },
);

Button.displayName = "Button";
