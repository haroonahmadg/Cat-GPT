import { ImagePlus, PenLine, Globe, LucideIcon } from "lucide-react";

interface QuickAction {
  label: string;
  icon: LucideIcon;
}

const ACTIONS: QuickAction[] = [
  { label: "Create an image or sticker", icon: ImagePlus },
  { label: "Write or edit", icon: PenLine },
  { label: "Search the web", icon: Globe },
];

interface QuickActionsProps {
  onSelect: (label: string) => void;
}

export function QuickActions({ onSelect }: QuickActionsProps) {
  return (
    <div className="mt-4 flex flex-col gap-1">
      {ACTIONS.map(({ label, icon: Icon }) => (
        <button
          key={label}
          onClick={() => onSelect(label)}
          className="flex items-center gap-3 rounded-lg px-2 py-1.5 text-left text-sm text-white/70 transition-colors hover:bg-surface-hover hover:text-white"
        >
          <Icon size={16} className="text-white/50" />
          {label}
        </button>
      ))}
    </div>
  );
}
