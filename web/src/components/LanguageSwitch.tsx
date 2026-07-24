import { useI18n, type Locale } from "../i18n";

type Props = {
  compact?: boolean;
  className?: string;
};

export function LanguageSwitch({ compact = false, className = "" }: Props) {
  const { locale, setLocale, t } = useI18n();

  const set = (next: Locale) => () => setLocale(next);

  return (
    <div
      className={`lang-switch ${compact ? "compact" : ""} ${className}`.trim()}
      role="group"
      aria-label={t("lang.label")}
    >
      {!compact ? <span className="lang-switch-label">{t("lang.label")}</span> : null}
      <div className="lang-switch-btns">
        <button
          type="button"
          className={locale === "zh" ? "lang-btn active" : "lang-btn"}
          aria-pressed={locale === "zh"}
          title={t("lang.switchToZh")}
          onClick={set("zh")}
        >
          {t("lang.zh")}
        </button>
        <button
          type="button"
          className={locale === "en" ? "lang-btn active" : "lang-btn"}
          aria-pressed={locale === "en"}
          title={t("lang.switchToEn")}
          onClick={set("en")}
        >
          {t("lang.en")}
        </button>
      </div>
    </div>
  );
}
