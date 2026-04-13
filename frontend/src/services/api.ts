const BASE = (import.meta.env.VITE_API_BASE as string | undefined)?.trim() || "";

const TENANT_STORAGE_KEY = "multimodal_rag_tenant_id";

export function getTenantId(): string {
  try {
    return localStorage.getItem(TENANT_STORAGE_KEY) || "default";
  } catch {
    return "default";
  }
}

export function setTenantId(id: string): void {
  const trimmed = id.trim();
  if (!trimmed || trimmed.length > 64) {
    throw new Error("Workspace id must be 1–64 characters.");
  }
  localStorage.setItem(TENANT_STORAGE_KEY, trimmed);
}

function withAuthHeaders(init?: RequestInit): RequestInit {
  const headers = new Headers(init?.headers);
  headers.set("X-Tenant-ID", getTenantId());
  const key = import.meta.env.VITE_API_KEY as string | undefined;
  if (key) {
    headers.set("X-API-Key", key);
  }
  return { ...init, headers };
}

async function apiFetch(input: string, init?: RequestInit): Promise<Response> {
  return fetch(input, withAuthHeaders(init));
}

/** Readable message from FastAPI `{ "detail": "..." }` or validation errors */
async function parseHttpError(r: Response): Promise<string> {
  const t = await r.text();
  try {
    const j = JSON.parse(t) as { detail?: unknown };
    if (typeof j.detail === "string") return j.detail;
    if (Array.isArray(j.detail)) {
      return j.detail
        .map((x: { msg?: string; loc?: unknown }) => x.msg ?? JSON.stringify(x))
        .join("; ");
    }
  } catch {
    /* plain text body */
  }
  if (t.trim()) return t;
  return `Request failed (${r.status})`;
}

function networkHint(err: unknown): string {
  if (err instanceof TypeError && err.message === "Failed to fetch") {
    return (
      "Cannot reach API. Start the backend (cd backend && ./run_dev.sh on port 8000) and open the app with " +
      "npm run dev (port 5173) so /upload is proxied. If the API runs on another port, update vite.config.ts proxy target."
    );
  }
  if (err instanceof Error) return err.message;
  return "Request failed";
}

export type SourceType = "pdf" | "docx" | "txt" | "image" | "video";

export interface SourceListItem {
  document_id: string;
  source_name: string;
  source_type: SourceType;
  created_at: string;
}

export interface UploadResponse {
  document_id: string;
  source_name: string;
  source_type: SourceType;
  chunks_indexed: number;
  message: string;
}

export interface Citation {
  document_id: string;
  chunk_id: string;
  source_name: string;
  source_type: SourceType;
  similarity_score: number;
  excerpt: string;
  chunk_index: number | null;
}

export interface QueryResponse {
  answer: string;
  citations: Citation[];
  retrieved_chunks_preview: Array<Record<string, unknown>>;
}

export async function fetchSources(): Promise<SourceListItem[]> {
  let r: Response;
  try {
    r = await apiFetch(`${BASE}/sources`);
  } catch (e) {
    throw new Error(networkHint(e));
  }
  if (!r.ok) throw new Error(await parseHttpError(r));
  return r.json();
}

export async function deleteSource(documentId: string): Promise<void> {
  let r: Response;
  try {
    r = await apiFetch(`${BASE}/sources/${encodeURIComponent(documentId)}`, {
      method: "DELETE",
    });
  } catch (e) {
    throw new Error(networkHint(e));
  }
  if (!r.ok) throw new Error(await parseHttpError(r));
}

export async function uploadFiles(files: File[]): Promise<UploadResponse[]> {
  const fd = new FormData();
  files.forEach((f) => fd.append("files", f));
  const r = await apiFetch(`${BASE}/upload`, { method: "POST", body: fd });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export type JobStatus = "pending" | "running" | "completed" | "failed";

export interface IngestionJob {
  job_id: string;
  status: JobStatus;
  source_name: string;
  created_at: string;
  updated_at: string;
  error: string | null;
  result: UploadResponse | null;
}

export interface AsyncUploadAccepted {
  job_id: string;
  source_name: string;
}

/** Non-blocking upload: returns job ids; use waitForIngestionJobs to poll. */
export async function uploadFilesAsync(files: File[]): Promise<AsyncUploadAccepted[]> {
  const fd = new FormData();
  files.forEach((f) => fd.append("files", f));
  let r: Response;
  try {
    r = await apiFetch(`${BASE}/upload/async`, { method: "POST", body: fd });
  } catch (e) {
    throw new Error(networkHint(e));
  }
  if (!r.ok) throw new Error(await parseHttpError(r));
  return r.json();
}

/** Paste plain text as a .txt source (async indexing). */
export async function uploadTextAsync(text: string, title?: string): Promise<AsyncUploadAccepted[]> {
  let r: Response;
  try {
    r = await apiFetch(`${BASE}/upload/text/async`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, title: title?.trim() || undefined }),
    });
  } catch (e) {
    throw new Error(networkHint(e));
  }
  if (!r.ok) throw new Error(await parseHttpError(r));
  return r.json();
}

/** Fetch a public URL; server extracts HTML/plain text and indexes as .txt. */
export async function uploadUrlAsync(url: string): Promise<AsyncUploadAccepted[]> {
  let r: Response;
  try {
    r = await apiFetch(`${BASE}/upload/url/async`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: url.trim() }),
    });
  } catch (e) {
    throw new Error(networkHint(e));
  }
  if (!r.ok) throw new Error(await parseHttpError(r));
  return r.json();
}

export async function fetchJob(jobId: string): Promise<IngestionJob> {
  let r: Response;
  try {
    r = await apiFetch(`${BASE}/jobs/${encodeURIComponent(jobId)}`);
  } catch (e) {
    throw new Error(networkHint(e));
  }
  if (!r.ok) throw new Error(await parseHttpError(r));
  return r.json();
}

/**
 * Poll until all jobs finish. Throws on timeout or if any job fails (error message lists sources).
 */
export async function waitForIngestionJobs(
  jobIds: string[],
  opts?: {
    onUpdate?: (j: IngestionJob) => void;
    intervalMs?: number;
    maxWaitMs?: number;
  },
): Promise<IngestionJob[]> {
  const interval = opts?.intervalMs ?? 1000;
  const maxWait = opts?.maxWaitMs ?? 3_600_000;
  const start = Date.now();
  const results = new Map<string, IngestionJob>();
  const pending = new Set(jobIds);

  while (pending.size && Date.now() - start < maxWait) {
    for (const id of [...pending]) {
      const j = await fetchJob(id);
      opts?.onUpdate?.(j);
      if (j.status === "completed" || j.status === "failed") {
        results.set(id, j);
        pending.delete(id);
      }
    }
    if (!pending.size) break;
    await new Promise((r) => setTimeout(r, interval));
  }

  if (pending.size) {
    throw new Error("Ingestion timed out. Check /jobs/{job_id} for status.");
  }

  const failed = [...results.values()].filter((x) => x.status === "failed");
  if (failed.length) {
    const msg = failed.map((f) => `${f.source_name}: ${f.error || "failed"}`).join("; ");
    throw new Error(msg);
  }

  return jobIds.map((id) => results.get(id)!);
}

export async function queryApi(
  query: string,
  opts?: {
    top_k?: number;
    document_ids?: string[];
    response_format?: string;
  },
): Promise<QueryResponse> {
  const r = await apiFetch(`${BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      top_k: opts?.top_k,
      document_ids: opts?.document_ids,
      response_format: opts?.response_format,
    }),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function fetchChunks(docId: string): Promise<{
  document_id: string;
  chunks: Array<{
    chunk_id: string;
    document_id: string;
    chunk_index: number | null;
    text: string;
    source_name: string;
  }>;
}> {
  const r = await apiFetch(`${BASE}/chunks/${encodeURIComponent(docId)}`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}
