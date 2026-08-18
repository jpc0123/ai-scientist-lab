import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import en from "./en";
import zh from "./zh";
import {
  getMessage,
  LOCALE_STORAGE_KEY,
  type Locale,
  type MessageTree,
} from "./types";

const catalogs: Record<Locale, MessageTree> = { zh, en };

type I18nContextValue = {
  locale: Locale;
  setLocale: (locale: Locale) => void;
  t: (path: string, vars?: Record<string, string | number>) => string;
};

const I18nContext = createContext<I18nContextValue | null>(null);

function readInitialLocale(): Locale {
  try {
    const raw = localStorage.getItem(LOCALE_STORAGE_KEY);
    if (raw === "en" || raw === "zh") return raw;
  } catch {
    /* ignore */
  }
  const nav = typeof navigator !== "undefined" ? navigator.language : "zh";
  return nav.toLowerCase().startsWith("en") ? "en" : "zh";
}

export function LocaleProvider({ children }: { children: ReactNode }) {
  const [locale, setLocaleState] = useState<Locale>(readInitialLocale);

  const setLocale = useCallback((next: Locale) => {
    setLocaleState(next);
    try {
      localStorage.setItem(LOCALE_STORAGE_KEY, next);
    } catch {
      /* ignore */
    }
    if (typeof document !== "undefined") {
      document.documentElement.lang = next === "zh" ? "zh-CN" : "en";
    }
  }, []);

  const t = useCallback(
    (path: string, vars?: Record<string, string | number>) =>
      getMessage(catalogs[locale], path, vars),
    [locale],
  );

  const value = useMemo(
    () => ({ locale, setLocale, t }),
    [locale, setLocale, t],
  );

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}

export function useI18n(): I18nContextValue {
  const ctx = useContext(I18nContext);
  if (!ctx) throw new Error("useI18n must be used within LocaleProvider");
  return ctx;
}

export function useT() {
  return useI18n().t;
}
