import { useState, useCallback } from 'react';
import { Loader2, FlaskConical, ChevronDown, ChevronUp, CheckCircle2, XCircle, Minus, X, ZoomIn } from 'lucide-react';
import { evalTextSearch } from '../api';
import type { EvalSearchResponse, MetricAtK, EvalResultItem } from '../types';

interface EvalPreset {
  label: string;
  query: string;
  tag: string;
}

const PRESETS: EvalPreset[] = [
  { label: 'Crosswalk', query: 'crosswalk on road', tag: 'crosswalk' },
  { label: 'Stop line', query: 'stop line on road', tag: 'stopline' },
  { label: 'Cones / Barriers', query: 'traffic cones on road', tag: 'cones' },
  { label: 'Lane marking', query: 'lane markings on road', tag: 'lane_marking' },
  { label: 'Road edge', query: 'road edge curb', tag: 'road_edge' },
];

const K_VALUES = [5, 10, 20, 50, 100, 200];

function MetricBar({ value, max = 1.0, color }: { value: number; max?: number; color: string }) {
  const pct = Math.min((value / max) * 100, 100);
  return (
    <div className="w-full h-2 rounded-full bg-gray-800 overflow-hidden">
      <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
    </div>
  );
}

function MetricsTable({ metrics, totalRelevant }: { metrics: MetricAtK[]; totalRelevant: number }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-gray-400 border-b border-gray-800">
            <th className="text-left py-2 px-3 font-medium">K</th>
            <th className="text-left py-2 px-3 font-medium">Precision@K</th>
            <th className="text-left py-2 px-3 font-medium">Recall@K</th>
            <th className="text-left py-2 px-3 font-medium">nDCG@K</th>
            <th className="text-right py-2 px-3 font-medium">Relevant / K</th>
          </tr>
        </thead>
        <tbody>
          {metrics.map((m) => (
            <tr key={m.k} className="border-b border-gray-800/50 hover:bg-gray-900/50">
              <td className="py-2 px-3 font-mono text-gray-300">{m.k}</td>
              <td className="py-2 px-3">
                <div className="flex items-center gap-2">
                  <span className="font-mono w-14 text-right">{(m.precision * 100).toFixed(1)}%</span>
                  <div className="flex-1 max-w-[120px]">
                    <MetricBar value={m.precision} color="bg-blue-500" />
                  </div>
                </div>
              </td>
              <td className="py-2 px-3">
                <div className="flex items-center gap-2">
                  <span className="font-mono w-14 text-right">{(m.recall * 100).toFixed(1)}%</span>
                  <div className="flex-1 max-w-[120px]">
                    <MetricBar value={m.recall} color="bg-emerald-500" />
                  </div>
                </div>
              </td>
              <td className="py-2 px-3">
                <div className="flex items-center gap-2">
                  <span className="font-mono w-14 text-right">{(m.ndcg * 100).toFixed(1)}%</span>
                  <div className="flex-1 max-w-[120px]">
                    <MetricBar value={m.ndcg} color="bg-amber-500" />
                  </div>
                </div>
              </td>
              <td className="py-2 px-3 text-right font-mono text-gray-400">
                {m.relevant_count} / {m.total}
              </td>
            </tr>
          ))}
        </tbody>
        <tfoot>
          <tr className="text-gray-500 text-xs">
            <td colSpan={5} className="pt-2 px-3">
              Total relevant in index: {totalRelevant}
            </td>
          </tr>
        </tfoot>
      </table>
    </div>
  );
}

function ImageLightbox({
  item,
  rank,
  relevantTag,
  onClose,
  onPrev,
  onNext,
}: {
  item: EvalResultItem;
  rank: number;
  relevantTag: string;
  onClose: () => void;
  onPrev: (() => void) | null;
  onNext: (() => void) | null;
}) {
  const hasGt = item.gt_tags.length > 0;
  return (
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center bg-black/85 backdrop-blur-sm"
      onClick={onClose}
      onKeyDown={(e) => {
        if (e.key === 'Escape') onClose();
        if (e.key === 'ArrowLeft' && onPrev) onPrev();
        if (e.key === 'ArrowRight' && onNext) onNext();
      }}
      tabIndex={0}
      role="dialog"
    >
      <div
        className="relative max-w-[90vw] max-h-[90vh] flex flex-col bg-gray-950 rounded-2xl border border-gray-800 overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3 border-b border-gray-800">
          <div className="flex items-center gap-3">
            <span className="font-mono text-gray-500 text-lg">#{rank}</span>
            <div
              className={`flex items-center gap-2 px-3 py-1 rounded-full text-sm font-medium ${
                item.is_relevant
                  ? 'bg-emerald-900/60 text-emerald-300'
                  : hasGt
                  ? 'bg-red-900/40 text-red-300'
                  : 'bg-gray-800 text-gray-400'
              }`}
            >
              {item.is_relevant ? (
                <><CheckCircle2 size={14} /> Relevant</>
              ) : hasGt ? (
                <><XCircle size={14} /> Not relevant</>
              ) : (
                <><Minus size={14} /> No ground truth</>
              )}
            </div>
            <span className="font-mono text-amber-400 text-sm">{(item.score * 100).toFixed(1)}%</span>
          </div>
          <button onClick={onClose} className="p-1.5 rounded-lg hover:bg-gray-800 text-gray-400 hover:text-white transition-colors">
            <X size={20} />
          </button>
        </div>

        {/* Image */}
        <div className="flex-1 overflow-auto p-4 flex items-center justify-center bg-gray-950">
          <img
            src={`/api/static/${item.uri}`}
            alt=""
            className="max-w-full max-h-[65vh] object-contain rounded-lg"
          />
        </div>

        {/* Footer with metadata */}
        <div className="px-5 py-3 border-t border-gray-800 space-y-2">
          <div className="flex items-center gap-3 flex-wrap">
            {item.use_case && (
              <span className="px-2 py-1 text-xs rounded-md bg-blue-900/40 text-blue-300 font-medium">
                {item.use_case}
              </span>
            )}
            <span className="text-xs text-gray-500 font-mono truncate">{item.image_id}</span>
          </div>
          {hasGt && (
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-xs text-gray-500">Ground truth:</span>
              {item.gt_tags.map((t) => (
                <span
                  key={t}
                  className={`px-2 py-0.5 text-xs rounded-md font-medium ${
                    t === relevantTag
                      ? item.is_relevant
                        ? 'bg-emerald-900 text-emerald-300 ring-1 ring-emerald-500'
                        : 'bg-gray-800 text-gray-400'
                      : 'bg-gray-800 text-gray-500'
                  }`}
                >
                  {t}
                </span>
              ))}
            </div>
          )}
          {!hasGt && (
            <p className="text-xs text-gray-600">This image has no label annotations (unlabeled).</p>
          )}
          {/* Nav arrows */}
          <div className="flex items-center justify-between pt-1">
            <button
              onClick={(e) => { e.stopPropagation(); onPrev?.(); }}
              disabled={!onPrev}
              className="px-3 py-1.5 text-xs rounded-lg bg-gray-800 text-gray-400 hover:text-white hover:bg-gray-700 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
            >
              ← Previous
            </button>
            <button
              onClick={(e) => { e.stopPropagation(); onNext?.(); }}
              disabled={!onNext}
              className="px-3 py-1.5 text-xs rounded-lg bg-gray-800 text-gray-400 hover:text-white hover:bg-gray-700 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
            >
              Next →
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

function ResultRow({ item, rank, onClick }: { item: EvalResultItem; rank: number; onClick: () => void }) {
  const hasGt = item.gt_tags.length > 0;
  return (
    <div
      className={`flex items-center gap-3 px-3 py-2 rounded-lg border text-sm cursor-pointer hover:ring-1 hover:ring-gray-600 transition-all ${
        item.is_relevant
          ? 'border-emerald-700 bg-emerald-950/30'
          : hasGt
          ? 'border-red-800/50 bg-red-950/20'
          : 'border-gray-800/50 bg-gray-900/30'
      }`}
      onClick={onClick}
    >
      <span className="font-mono text-gray-500 w-8 text-right shrink-0">#{rank}</span>
      <div className="relative w-10 h-10 rounded overflow-hidden bg-gray-800 shrink-0 group">
        <img
          src={`/api/static/${item.uri}`}
          alt=""
          className="w-full h-full object-cover"
          loading="lazy"
        />
        <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
          <ZoomIn size={14} className="text-white" />
        </div>
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          {item.is_relevant ? (
            <CheckCircle2 size={14} className="text-emerald-400 shrink-0" />
          ) : hasGt ? (
            <XCircle size={14} className="text-red-400 shrink-0" />
          ) : (
            <Minus size={14} className="text-gray-600 shrink-0" />
          )}
          <span className="truncate text-gray-300">{item.image_id.slice(0, 20)}...</span>
          {item.use_case && (
            <span className="px-1.5 py-0.5 text-[10px] rounded bg-gray-800 text-gray-400 shrink-0">
              {item.use_case}
            </span>
          )}
        </div>
        {hasGt && (
          <div className="flex gap-1 mt-0.5 flex-wrap">
            {item.gt_tags.map((t) => (
              <span
                key={t}
                className={`px-1.5 py-0.5 text-[10px] rounded font-medium ${
                  item.is_relevant && t === item.gt_tags.find(() => true)
                    ? 'bg-emerald-900 text-emerald-300'
                    : 'bg-gray-800 text-gray-500'
                }`}
              >
                {t}
              </span>
            ))}
          </div>
        )}
      </div>
      <span className="font-mono text-gray-500 shrink-0">{(item.score * 100).toFixed(1)}%</span>
    </div>
  );
}

export function EvaluatePage() {
  const [query, setQuery] = useState('');
  const [relevantTag, setRelevantTag] = useState('crosswalk');
  const [topK, setTopK] = useState(200);
  const [response, setResponse] = useState<EvalSearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showAll, setShowAll] = useState(false);
  const [batchResults, setBatchResults] = useState<EvalSearchResponse[]>([]);
  const [batchLoading, setBatchLoading] = useState(false);
  const [lightboxIdx, setLightboxIdx] = useState<number | null>(null);

  const handleEval = useCallback(async () => {
    if (!query.trim() || !relevantTag.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await evalTextSearch(query.trim(), relevantTag.trim(), topK, K_VALUES);
      setResponse(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Eval failed');
    } finally {
      setLoading(false);
    }
  }, [query, relevantTag, topK]);

  const handlePreset = useCallback(
    async (preset: EvalPreset) => {
      setQuery(preset.query);
      setRelevantTag(preset.tag);
      setLoading(true);
      setError(null);
      try {
        const res = await evalTextSearch(preset.query, preset.tag, topK, K_VALUES);
        setResponse(res);
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Eval failed');
      } finally {
        setLoading(false);
      }
    },
    [topK],
  );

  const handleBatchAll = useCallback(async () => {
    setBatchLoading(true);
    setBatchResults([]);
    setError(null);
    const results: EvalSearchResponse[] = [];
    for (const preset of PRESETS) {
      try {
        const res = await evalTextSearch(preset.query, preset.tag, topK, K_VALUES);
        results.push(res);
        setBatchResults([...results]);
      } catch {
        // skip failed
      }
    }
    setBatchLoading(false);
  }, [topK]);

  const displayedResults = response
    ? showAll
      ? response.results
      : response.results.filter((r) => r.gt_tags.length > 0)
    : [];

  return (
    <div className="max-w-[1600px] mx-auto px-4 py-6 space-y-6">
      <div className="flex items-center gap-3">
        <FlaskConical size={24} className="text-amber-400" />
        <div>
          <h1 className="text-xl font-bold">Text-to-Image Search Evaluation</h1>
          <p className="text-sm text-gray-500">
            Measure Precision@K, Recall@K, and nDCG using label-derived ground truth from annotated images
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* Left: controls */}
        <div className="lg:col-span-2 space-y-4">
          <div className="space-y-3">
            <div>
              <label className="block text-xs text-gray-400 mb-1">Text Query</label>
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleEval()}
                placeholder="e.g. crosswalk on road"
                className="w-full px-3 py-2 rounded-lg bg-gray-900 border border-gray-700 text-sm focus:outline-none focus:border-amber-500"
              />
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-xs text-gray-400 mb-1">Ground Truth Tag</label>
                <select
                  value={relevantTag}
                  onChange={(e) => setRelevantTag(e.target.value)}
                  className="w-full px-3 py-2 rounded-lg bg-gray-900 border border-gray-700 text-sm focus:outline-none focus:border-amber-500"
                >
                  <option value="crosswalk">crosswalk</option>
                  <option value="stopline">stopline</option>
                  <option value="cones">cones</option>
                  <option value="lane_marking">lane_marking</option>
                  <option value="road_edge">road_edge</option>
                </select>
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">Top K</label>
                <input
                  type="number"
                  value={topK}
                  onChange={(e) => setTopK(Number(e.target.value))}
                  min={10}
                  max={500}
                  className="w-full px-3 py-2 rounded-lg bg-gray-900 border border-gray-700 text-sm focus:outline-none focus:border-amber-500"
                />
              </div>
            </div>
          </div>

          <button
            onClick={handleEval}
            disabled={!query.trim() || loading}
            className="w-full py-2.5 rounded-lg bg-amber-600 hover:bg-amber-500 disabled:opacity-40 disabled:cursor-not-allowed font-medium text-sm transition-colors flex items-center justify-center gap-2"
          >
            {loading ? <Loader2 size={16} className="animate-spin" /> : <FlaskConical size={16} />}
            {loading ? 'Evaluating...' : 'Run Evaluation'}
          </button>

          <div className="space-y-2">
            <p className="text-xs text-gray-500 font-medium uppercase tracking-wider">Presets</p>
            <div className="flex flex-wrap gap-1.5">
              {PRESETS.map((p) => (
                <button
                  key={p.tag}
                  onClick={() => handlePreset(p)}
                  disabled={loading}
                  className="px-3 py-1.5 rounded-full text-xs bg-gray-800 border border-gray-700 text-gray-400 hover:text-white hover:border-amber-600 transition-colors disabled:opacity-40"
                >
                  {p.label}
                </button>
              ))}
            </div>
            <button
              onClick={handleBatchAll}
              disabled={batchLoading || loading}
              className="w-full py-2 rounded-lg bg-gray-800 hover:bg-gray-700 border border-gray-700 text-sm text-gray-300 transition-colors disabled:opacity-40 flex items-center justify-center gap-2"
            >
              {batchLoading ? <Loader2 size={14} className="animate-spin" /> : null}
              {batchLoading ? 'Running all...' : 'Run All Presets'}
            </button>
          </div>

          {error && (
            <div className="p-3 rounded-lg bg-red-950 border border-red-800 text-red-300 text-sm">
              {error}
            </div>
          )}

          {/* Batch summary */}
          {batchResults.length > 0 && (
            <div className="space-y-2">
              <h3 className="text-sm font-semibold text-gray-300">Batch Summary</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="text-gray-500 border-b border-gray-800">
                      <th className="text-left py-1.5 px-2">Tag</th>
                      <th className="text-right py-1.5 px-2">P@10</th>
                      <th className="text-right py-1.5 px-2">R@50</th>
                      <th className="text-right py-1.5 px-2">nDCG@50</th>
                      <th className="text-right py-1.5 px-2">DB</th>
                    </tr>
                  </thead>
                  <tbody>
                    {batchResults.map((r) => {
                      const p10 = r.metrics.find((m) => m.k === 10);
                      const r50 = r.metrics.find((m) => m.k === 50);
                      const n50 = r.metrics.find((m) => m.k === 50);
                      return (
                        <tr
                          key={r.relevant_tag}
                          className="border-b border-gray-800/50 hover:bg-gray-900/50 cursor-pointer"
                          onClick={() => {
                            setResponse(r);
                            setQuery(r.query);
                            setRelevantTag(r.relevant_tag);
                          }}
                        >
                          <td className="py-1.5 px-2 font-medium text-gray-300">{r.relevant_tag}</td>
                          <td className="py-1.5 px-2 text-right font-mono">
                            {p10 ? (p10.precision * 100).toFixed(1) : '-'}%
                          </td>
                          <td className="py-1.5 px-2 text-right font-mono">
                            {r50 ? (r50.recall * 100).toFixed(1) : '-'}%
                          </td>
                          <td className="py-1.5 px-2 text-right font-mono">
                            {n50 ? (n50.ndcg * 100).toFixed(1) : '-'}%
                          </td>
                          <td className="py-1.5 px-2 text-right text-gray-500">
                            {r.total_relevant_in_db}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>

        {/* Right: metrics + results */}
        <div className="lg:col-span-3 space-y-4 min-w-0">
          {response && (
            <>
              <div className="p-4 rounded-xl border border-gray-800 bg-gray-900/50 space-y-3">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="font-semibold text-gray-200">
                      &ldquo;{response.query}&rdquo;{' '}
                      <span className="text-amber-400">→ {response.relevant_tag}</span>
                    </h3>
                    <p className="text-xs text-gray-500 mt-0.5">
                      {response.total_relevant_in_db} relevant images in DB &middot;{' '}
                      {response.results_count} results returned
                    </p>
                  </div>
                </div>
                <MetricsTable metrics={response.metrics} totalRelevant={response.total_relevant_in_db} />
              </div>

              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-gray-300">
                    Results
                    <span className="ml-2 text-xs font-normal text-gray-500">
                      {showAll
                        ? `All ${response.results.length}`
                        : `${displayedResults.length} labeled only`}
                    </span>
                  </h3>
                  <button
                    onClick={() => setShowAll(!showAll)}
                    className="flex items-center gap-1 text-xs text-gray-500 hover:text-white transition-colors"
                  >
                    {showAll ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                    {showAll ? 'Show labeled only' : 'Show all'}
                  </button>
                </div>

                <div className="flex items-center gap-4 text-[10px] text-gray-500">
                  <span className="flex items-center gap-1">
                    <CheckCircle2 size={10} className="text-emerald-400" /> Relevant
                  </span>
                  <span className="flex items-center gap-1">
                    <XCircle size={10} className="text-red-400" /> Labeled but not relevant
                  </span>
                  <span className="flex items-center gap-1">
                    <Minus size={10} className="text-gray-600" /> No ground truth
                  </span>
                </div>

                <div className="space-y-1 max-h-[600px] overflow-y-auto pr-1">
                  {displayedResults.map((r, i) => (
                    <ResultRow
                      key={r.image_id}
                      item={r}
                      rank={r.position}
                      onClick={() => setLightboxIdx(i)}
                    />
                  ))}
                  {displayedResults.length === 0 && (
                    <p className="text-sm text-gray-500 py-4 text-center">
                      No {showAll ? '' : 'labeled '}results found
                    </p>
                  )}
                </div>
              </div>
            </>
          )}

          {!response && !loading && (
            <div className="flex flex-col items-center justify-center py-16 text-gray-600">
              <FlaskConical size={48} strokeWidth={1} />
              <p className="mt-4 text-sm">Select a preset or enter a custom query to evaluate</p>
              <p className="text-xs text-gray-700 mt-1">
                Uses label-derived ground truth from annotated ADAS images
              </p>
            </div>
          )}
        </div>
      </div>

      {lightboxIdx !== null && displayedResults[lightboxIdx] && (
        <ImageLightbox
          item={displayedResults[lightboxIdx]}
          rank={displayedResults[lightboxIdx].position}
          relevantTag={relevantTag}
          onClose={() => setLightboxIdx(null)}
          onPrev={lightboxIdx > 0 ? () => setLightboxIdx(lightboxIdx - 1) : null}
          onNext={lightboxIdx < displayedResults.length - 1 ? () => setLightboxIdx(lightboxIdx + 1) : null}
        />
      )}
    </div>
  );
}
