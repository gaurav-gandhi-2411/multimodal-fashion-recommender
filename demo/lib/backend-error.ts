// FastAPI error bodies are JSON (`{"detail": "..."}` or, for pydantic validation
// errors, `{"detail": [{"msg": "...", "loc": [...]}, ...]}`). Proxy routes previously
// forwarded the raw JSON text verbatim as the error message, so failures rendered as
// e.g. `{"detail":"Uploaded file is empty"}` in the UI instead of a clean sentence.
export function formatBackendError(rawText: string): string {
  try {
    const parsed: unknown = JSON.parse(rawText);
    if (parsed && typeof parsed === "object" && "detail" in parsed) {
      const detail = (parsed as { detail: unknown }).detail;
      if (typeof detail === "string") return detail;
      if (Array.isArray(detail)) {
        return detail
          .map((d) => (d && typeof d === "object" && "msg" in d ? String(d.msg) : JSON.stringify(d)))
          .join("; ");
      }
    }
  } catch {
    // Not JSON — fall through and return the raw text as-is.
  }
  return rawText;
}
