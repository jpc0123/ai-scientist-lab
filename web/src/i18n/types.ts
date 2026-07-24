export type Locale = "zh" | "en";

export const LOCALE_STORAGE_KEY = "scientist-lab.locale";

export type MessageTree = { [key: string]: string | MessageTree };

export function getMessage(
  tree: MessageTree,
  path: string,
  vars?: Record<string, string | number>,
): string {
  const parts = path.split(".");
  let cur: string | MessageTree | undefined = tree;
  for (const part of parts) {
    if (!cur || typeof cur === "string") return path;
    cur = cur[part];
  }
  if (typeof cur !== "string") return path;
  if (!vars) return cur;
  return cur.replace(/\{(\w+)\}/g, (_, key: string) =>
    vars[key] !== undefined ? String(vars[key]) : `{${key}}`,
  );
}
