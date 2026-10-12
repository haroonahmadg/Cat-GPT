/**
 * Placeholder data for the Phase 2 static shell.
 *
 * None of this is wired to the FastAPI backend (api/main.py) yet — that
 * is Phase 3 ("wire it up"). Keeping it isolated in one file makes that
 * swap later a matter of replacing this module's exports with real
 * fetch/SSE calls, without touching the components that consume it.
 */

export type ThreadSummary = {
  threadId: string;
  title: string;
  pinned: boolean;
};

export const mockRecentChats: ThreadSummary[] = [
  { threadId: "1", title: "Create Grade One Lesson Plan", pinned: false },
  { threadId: "2", title: "Create Grade 1 Lesson Plan", pinned: false },
  { threadId: "3", title: "Understanding Emojis", pinned: false },
  { threadId: "4", title: "Suggest LinkedIn Graphic Change", pinned: false },
  { threadId: "5", title: "Create Timetable Image PDF", pinned: false },
  { threadId: "6", title: "Create lesson plan", pinned: false },
  { threadId: "7", title: "Write Maths Lesson Plan", pinned: false },
  { threadId: "8", title: "Create Maths Worksheet", pinned: false },
  { threadId: "9", title: "Convert Tags to Outline", pinned: false },
  { threadId: "10", title: "Maths Worksheet Design", pinned: false },
];

export const mockUser = {
  name: "Guest",
  plan: "Free",
};
