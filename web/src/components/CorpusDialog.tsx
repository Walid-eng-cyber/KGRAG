import { useEffect, useMemo, useState } from "react";
import { api, ApiError } from "../api";
import { typeLabel } from "../lib/geometry";
import type { Corpus, Preset } from "../types";
import { Glyph } from "./Shape";

type Rel = [string, string, string];

interface Props {
  presets: Record<string, Preset>;
  editing: Corpus | null; // null = create
  onClose: () => void;
  onSaved: (c: Corpus) => void;
  onDeleted: (id: string) => void;
}

const norm = (s: string) => s.trim().toUpperCase().replace(/[^A-Z0-9_]+/g, "_").replace(/^_+|_+$/g, "");

/** Create or edit a corpus: name, description and what to extract (preset or custom schema). */
export function CorpusDialog({ presets, editing, onClose, onSaved, onDeleted }: Props) {
  const fixed = editing?.kind === "sec_insiders";
  const [name, setName] = useState(editing?.name ?? "");
  const [description, setDescription] = useState(editing?.description ?? "");
  const [preset, setPreset] = useState<string>(editing ? editing.schema.preset : "general");
  const [entities, setEntities] = useState<string[]>(editing?.schema.entities ?? presets.general?.entities ?? []);
  const [relations, setRelations] = useState<Rel[]>(editing?.schema.relations ?? presets.general?.relations ?? []);
  const [newType, setNewType] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const choosePreset = (key: string) => {
    setPreset(key);
    if (key !== "custom" && presets[key]) {
      setEntities(presets[key].entities);
      setRelations(presets[key].relations);
    }
  };
  const customize = () => setPreset("custom");
  const schemaChanged = useMemo(() => editing && !fixed && (
    JSON.stringify(entities) !== JSON.stringify(editing.schema.entities) ||
    JSON.stringify(relations) !== JSON.stringify(editing.schema.relations)), [editing, fixed, entities, relations]);

  const addType = () => {
    const t = norm(newType);
    if (t && !entities.includes(t)) { setEntities([...entities, t]); customize(); }
    setNewType("");
  };
  const removeType = (t: string) => {
    setEntities(entities.filter((x) => x !== t));
    setRelations(relations.filter((r) => r[0] !== t && r[2] !== t));
    customize();
  };
  const setRel = (i: number, j: 0 | 1 | 2, v: string) => {
    const next = relations.map((r) => [...r] as Rel);
    next[i][j] = j === 1 ? v : v;
    setRelations(next);
    customize();
  };

  const save = async () => {
    setError(null);
    setSaving(true);
    try {
      const rels = relations.map((r) => [r[0], norm(r[1]), r[2]] as Rel).filter((r) => r[1]);
      const schemaPart = preset === "custom" || (editing && schemaChanged)
        ? { entities, relations: rels }
        : editing ? {} : { preset };
      const res = editing
        ? await api.updateCorpus(editing.id, { name, description, ...(fixed ? {} : schemaPart) })
        : await api.createCorpus({ name, description, ...schemaPart });
      onSaved(res.corpus);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  };

  const remove = async () => {
    if (!editing) return;
    if (!window.confirm(`Delete “${editing.name}” and all ${editing.documents} of its documents? This can't be undone.`)) return;
    try {
      await api.deleteCorpus(editing.id);
      onDeleted(editing.id);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    }
  };

  return (
    <div className="modal-backdrop" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <section className="modal" role="dialog" aria-modal="true" aria-labelledby="corpus-dialog-title">
        <div className="modal-h">
          <h2 id="corpus-dialog-title">{editing ? "Corpus settings" : "New corpus"}</h2>
          <button className="modal-x" aria-label="Close" onClick={onClose}>×</button>
        </div>
        <div className="modal-b">
          <p className="muted" style={{ margin: 0 }}>Corpora are private: only you can see them, their documents and their answers.</p>
          <label className="field">
            <span>Name</span>
            <input value={name} onChange={(e) => setName(e.target.value)} maxLength={80} placeholder="e.g. Walid – career" autoFocus />
          </label>
          <label className="field">
            <span>Description <i>(helps extraction: what are these documents about?)</i></span>
            <textarea value={description} onChange={(e) => setDescription(e.target.value)} maxLength={500} rows={2}
              placeholder="e.g. My CV, transcripts and certificates" />
          </label>

          {fixed ? (
            <p className="muted">This corpus is a structured data set (SEC Forms 3/4/5); its schema is fixed.</p>
          ) : (
            <>
              <div className="field">
                <span>What should be extracted?</span>
                <div className="presets">
                  {Object.entries(presets).map(([key, p]) => (
                    <button key={key} type="button" className={`preset${preset === key ? " on" : ""}`} onClick={() => choosePreset(key)}>
                      <b>{p.label}</b><span>{p.description}</span>
                    </button>
                  ))}
                  <button type="button" className={`preset${preset === "custom" ? " on" : ""}`} onClick={customize}>
                    <b>Custom</b><span>Start from the types below and edit them freely.</span>
                  </button>
                </div>
              </div>

              <div className="field">
                <span>Entity types</span>
                <div className="type-chips">
                  {entities.map((t) => (
                    <span className="type-chip" key={t}>
                      <Glyph type={t} size={14} />{typeLabel(t)}
                      <button aria-label={`Remove ${t}`} onClick={() => removeType(t)}>×</button>
                    </span>
                  ))}
                  <input className="type-add" value={newType} onChange={(e) => setNewType(e.target.value)}
                    onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); addType(); } }}
                    placeholder="Add a type…" maxLength={32} />
                </div>
              </div>

              <div className="field">
                <span>Relations <i>(subject · relation · object)</i></span>
                <div className="rels">
                  {relations.map((r, i) => (
                    <div className="rel-row" key={i}>
                      <select value={r[0]} onChange={(e) => setRel(i, 0, e.target.value)}>
                        {entities.map((t) => <option key={t} value={t}>{typeLabel(t)}</option>)}
                      </select>
                      <input value={r[1]} onChange={(e) => setRel(i, 1, e.target.value.toUpperCase())} placeholder="RELATION" maxLength={32} />
                      <select value={r[2]} onChange={(e) => setRel(i, 2, e.target.value)}>
                        {entities.map((t) => <option key={t} value={t}>{typeLabel(t)}</option>)}
                      </select>
                      <button className="modal-x" aria-label="Remove relation" onClick={() => { setRelations(relations.filter((_, j) => j !== i)); customize(); }}>×</button>
                    </div>
                  ))}
                  <button className="btn ghost small" type="button" disabled={!entities.length}
                    onClick={() => { setRelations([...relations, [entities[0], "", entities[0]]]); customize(); }}>
                    + Add relation
                  </button>
                </div>
              </div>
              {schemaChanged && (
                <p className="notice-inline">The new schema applies to documents ingested from now on. Re-ingest existing documents to apply it to them.</p>
              )}
            </>
          )}
          {error && <div className="docs-error" role="alert" style={{ margin: 0 }}>{error}</div>}
        </div>
        <div className="modal-f">
          {editing && !fixed ? <button className="btn danger" onClick={() => void remove()}>Delete corpus</button> : <span />}
          <div style={{ display: "flex", gap: 8 }}>
            <button className="btn ghost" onClick={onClose}>Cancel</button>
            <button className="btn" onClick={() => void save()} disabled={saving || !name.trim()}>
              {saving ? "Saving…" : editing ? "Save" : "Create corpus"}
            </button>
          </div>
        </div>
      </section>
    </div>
  );
}
