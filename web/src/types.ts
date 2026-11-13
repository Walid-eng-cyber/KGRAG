// JSON contract served by server/main.py.

export type EntityType =
  | "COMPANY"
  | "PERSON"
  | "PRODUCT"
  | "SEGMENT"
  | "RISK"
  | "AUDITOR"
  | "LOCATION"
  | "GOVERNMENT_AGENCY"
  | "STOCK_EXCHANGE";

export type RoutePath = "GRAPH" | "VECTOR" | "HYBRID" | "NONE";

export interface GraphNode {
  id: string;
  label: string;
  type: EntityType;
  role: "anchor" | "target";
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  relation: string;
  tag: string | null;
}

export interface Metric {
  label: string;
  value: number;
  tag: string;
}

export interface GraphView {
  query_type: string | null;
  cypher: string | null;
  layout: "radial" | "bipartite" | "path" | "none";
  nodes: GraphNode[];
  edges: GraphEdge[];
  metrics: Metric[];
  error?: string | null;
}

export interface Source {
  tag: string;
  kind: "graph" | "passage" | "computed";
  text: string;
  cite?: string | null;
}

export interface Passage {
  tag: string | null;
  chunk_id: string;
  doc_id: string;
  section: string;
  filing_date: string;
  similarity: number;
  text: string;
}

export interface AskResponse {
  question: string;
  path: RoutePath;
  path_label: string;
  classified: string | null;
  confidence: number | null;
  answer: string;
  cited: string[];
  citations_valid: boolean;
  attempts: number;
  uncited_claims: string[];
  sources: Source[];
  graph: GraphView;
  passages: Passage[];
  latency_s: number;
  demo: boolean;
}

export interface CorpusSchema {
  preset: string;
  entities: string[];
  relations: [string, string, string][];
}

export interface Corpus {
  id: string;
  name: string;
  description: string;
  kind: "documents" | "sec_insiders";
  schema: CorpusSchema;
  owner: string | null;
  created_at: string;
  documents: number;
  ready: number;
}

export interface Preset {
  label: string;
  description: string;
  preset: string;
  entities: string[];
  relations: [string, string, string][];
}

export interface CorpusProfile {
  corpus: Corpus;
  profile: {
    types: { type: string; n: number }[];
    top_entities: { name: string; type: string; degree: number }[];
    relations: { relation: string; n: number }[];
  } | null;
  examples: Example[];
}

export interface CorpusInput {
  name: string;
  description: string;
  preset?: string;
  entities?: string[];
  relations?: [string, string, string][];
}

export type DocumentStatus = "uploaded" | "queued" | "ingesting" | "ready" | "failed";

export interface DocumentInfo {
  id: string;
  corpus_id: string | null;
  filename: string;
  title: string;
  kind: "pdf" | "text" | "html";
  size_bytes: number;
  status: DocumentStatus;
  progress: number | null;
  message: string | null;
  chunks: number | null;
  uploaded_by: string | null;
  uploaded_at: string;
  ingested_at: string | null;
}

/** A document's text for the viewer (GET /api/documents/{id}). */
export interface DocumentContent {
  document: Pick<DocumentInfo, "id" | "corpus_id" | "title" | "filename" | "kind" | "size_bytes" | "status" | "chunks"
    | "uploaded_at" | "ingested_at">;
  text: string;
  chars: number;
  truncated: boolean;
}

export interface Example {
  label: string;
  path: RoutePath;
  question: string;
}

export type AccountStatus = "pending" | "approved" | "rejected" | "disabled";

export interface User {
  email: string;
  name: string;
  picture?: string | null;
  role: "user" | "admin";
  status: AccountStatus;
  updates_opt_in: boolean;
  account_id?: string | null;
}

export interface Account extends User {
  created_at: string;
  decided_at: string | null;
  decided_by: string | null;
}

export interface Meta {
  demo: boolean;
  auth: { enabled: boolean; client_id: string | null; consent_text: string };
  examples: Example[];
  tickers: string[];
  corpus?: { label: string; detail: string }[];
  models: { router: string; answer: string; embed: string };
  billing?: { currency: string; priced_as: string | null };
  /** Set on a sign-in-off server that acts as one client user (KGRAG_AS_USER). */
  acting_as?: { email: string; name: string; account: string | null } | null;
}

export interface ServiceStatus {
  ok: boolean;
  detail: string;
}

export interface Health {
  ok: boolean;
  demo: boolean;
  services: Record<"neo4j" | "postgres" | "ollama", ServiceStatus>;
  models: Meta["models"];
}

export interface BenchmarkCategory {
  key: string;
  label: string;
  graphrag: number;
  vector: number;
  n: number;
}

export interface Benchmark {
  source: "published" | "results-file";
  categories: BenchmarkCategory[];
  overall: { graphrag: number; vector: number; n: number };
  latency: { graphrag_median_s: number | null; vector_median_s: number | null };
  costs: { label: string; vector: string | null; graphrag: string | null }[];
}

// ---- Client accounts and usage (server/usage.py) ----

export type CreditStatus = "ok" | "at_risk" | "over" | "no_credit";

export interface Credit {
  monthly_credit_eur: number;
  spent_eur: number;
  remaining_eur: number;
  projected_eur: number;
  used_pct: number | null;
  status: CreditStatus;
  month: string;
  resets_at: string;
}

export interface Billing {
  currency: string;
  priced_as: string | null;
  eur_per_1m_in: number;
  eur_per_1m_out: number;
  usd_to_eur: number;
  default_monthly_credit_eur: number;
}

export interface ClientAccount {
  id: string;
  name: string;
  industry: string;
  monthly_credit_eur: number;
  status: "active" | "suspended";
  members: number;
  created_at: string;
}

export interface AccountOverview {
  id: string;
  name: string;
  industry: string;
  status: "active" | "suspended";
  members: number;
  credit: Credit;
  last_7d: { questions: number; requests: number; cost_eur: number; active_users: number;
    p95_ms: number | null; error_pct: number | null };
}

export interface StageUsage {
  stage: string;
  calls: number;
  tokens_in: number;
  tokens_out: number;
  cost_eur: number;
  cost_share_pct: number;
  p50_ms: number | null;
  p95_ms: number | null;
}

export interface UserUsage {
  user: string;
  questions: number;
  ingests: number;
  requests: number;
  cost_eur: number;
  tokens: number;
  p95_ms: number | null;
  errors: number;
  last_seen: string | null;
  cost_share_pct: number;
}

export interface DayUsage {
  date: string;
  questions: number;
  ingests: number;
  requests: number;
  cost_eur: number;
  active_users: number;
}

export interface UsageReport {
  account: ClientAccount;
  billing: Billing;
  window: { days: number; since: string; until: string };
  credit: Credit;
  totals: {
    questions: number; ingests: number; requests: number; documents_chunks: number;
    cost_eur: number; ask_cost_eur: number; ingest_cost_eur: number; tokens_in: number; tokens_out: number;
    cost_per_question_eur: number | null; cost_per_answer_eur: number | null; cost_per_request_eur: number | null;
    active_users: number; active_users_7d: number; estimated_cost_share_pct: number;
  };
  latency: { p50_ms: number | null; p95_ms: number | null; p99_ms: number | null };
  quality: Record<"answered_pct" | "no_evidence_pct" | "error_pct" | "retry_pct" | "citation_fail_pct"
    | "low_confidence_pct" | "ingest_failed_pct", number | null>;
  paths: Record<string, number>;
  stages: StageUsage[];
  users: UserUsage[];
  daily: DayUsage[];
}

