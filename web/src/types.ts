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
