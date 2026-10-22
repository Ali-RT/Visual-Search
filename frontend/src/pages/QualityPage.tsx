import { useState, useCallback } from 'react';
import { Loader2, AlertTriangle, Eye, Sun, Contrast, Cpu, ChevronDown, ChevronUp } from 'lucide-react';
import { getWorstQuality } from '../api';
import type { QualityItem, QualityWorstResponse } from '../types';

function scoreColor(score: number): string {
  if (score >= 70) return 'text-emerald-400';
  if (score >= 45) return 'text-yellow-400';
  return 'text-red-400';
}

function scoreBg(score: number): string {
  if (score >= 70) return 'bg-emerald-500';
  if (score >= 45) return 'bg-yellow-500';
  return 'bg-red-500';
}

function labelBadge(label: string): string {
  switch (label) {
    case 'good': case 'sharp': return 'bg-emerald-900/50 text-emerald-300 border-emerald-700/50';
    case 'fair': case 'moderate': case 'low': return 'bg-yellow-900/50 text-yellow-300 border-yellow-700/50';
    case 'poor': case 'blurry': case 'very_low': case 'overexposed': case 'underexposed': case 'clipped':
      return 'bg-red-900/50 text-red-300 border-red-700/50';
    default: return 'bg-gray-800 text-gray-300 border-gray-700';
  }
}

function MetricRow({ icon: Icon, label, value, badge, barPct }: {
  icon: typeof Eye; label: string; value: string; badge: string; barPct: number;
}) {
  return (
    <div className="flex items-center gap-2">
      <Icon size={12} className="text-gray-500 shrink-0" />
      <span className="w-16 text-[10px] text-gray-500">{label}</span>
      <div className="flex-1 h-2 bg-gray-800 rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${scoreBg(barPct * 100)}`} style={{ width: `${Math.max(barPct * 100, 2)}%` }} />
      </div>
      <span className="w-12 text-[10px] text-gray-400 font-mono text-right">{value}</span>
      <span className={`px-1.5 py-0.5 rounded text-[9px] font-medium border ${labelBadge(badge)}`}>{badge}</span>
    </div>
  );
}

function QualityCard({ item, rank }: { item: QualityItem; rank: number }) {
  const [expanded, setExpanded] = useState(false);
  const m = item.metrics;

  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900/50 overflow-hidden">
      <div className="flex gap-4 p-3">
        {/* Thumbnail */}
        <div className="shrink-0 w-48 h-28 rounded-lg overflow-hidden bg-black relative">
          <img
            src={`/api/static/${item.uri}`}
            alt={item.filename}
            className="w-full h-full object-cover"
            loading="lazy"
          />
          <div className={`absolute top-1.5 left-1.5 px-1.5 py-0.5 rounded text-[10px] font-bold ${scoreBg(m.overall_score)} text-white`}>
            #{rank}
          </div>
        </div>

        {/* Summary */}
        <div className="flex-1 min-w-0 space-y-2">
          <div className="flex items-center justify-between">
            <p className="text-xs text-gray-400 truncate">{item.filename}</p>
            <div className="flex items-center gap-2">
              <span className={`text-lg font-bold font-mono ${scoreColor(m.overall_score)}`}>
                {m.overall_score.toFixed(0)}
              </span>
              <span className={`px-2 py-0.5 rounded-full text-[10px] font-medium border ${labelBadge(m.overall_label)}`}>
                {m.overall_label}
              </span>
            </div>
          </div>

          {/* Metric bars */}
          <div className="space-y-1.5">
            <MetricRow icon={Eye} label="Blur" value={m.blur_score.toFixed(0)} badge={m.blur_label} barPct={Math.min(m.blur_score / 300, 1)} />
            <MetricRow icon={Sun} label="Exposure" value={`${(m.exposure_score * 100).toFixed(0)}%`} badge={m.exposure_label} barPct={m.exposure_score} />
            <MetricRow icon={Contrast} label="Contrast" value={m.contrast_score.toFixed(0)} badge={m.contrast_label} barPct={Math.min(m.contrast_score / 60, 1)} />
            <MetricRow icon={Cpu} label="CLIP IQA" value={`${(m.clip_quality * 100).toFixed(0)}%`} badge={m.clip_label} barPct={m.clip_quality} />
          </div>

          {/* Issues */}
          {m.issues.length > 0 && (
            <button
              onClick={() => setExpanded(!expanded)}
              className="flex items-center gap-1 text-[10px] text-yellow-500 hover:text-yellow-400"
            >
              <AlertTriangle size={10} />
              {m.issues.length} issue{m.issues.length !== 1 ? 's' : ''}
              {expanded ? <ChevronUp size={10} /> : <ChevronDown size={10} />}
            </button>
          )}
          {expanded && m.issues.length > 0 && (
            <ul className="space-y-0.5">
              {m.issues.map((issue, i) => (
                <li key={i} className="text-[10px] text-yellow-600 flex items-start gap-1">
                  <span className="mt-0.5">•</span> {issue}
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
}

export function QualityPage() {
  const [data, setData] = useState<QualityWorstResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [count, setCount] = useState(20);
  const [selected, setSelected] = useState<QualityItem | null>(null);

  const handleScan = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await getWorstQuality(count);
      setData(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Quality scan failed');
    } finally {
      setLoading(false);
    }
  }, [count]);

  const poorCount = data?.items.filter((i) => i.metrics.overall_label === 'poor').length ?? 0;
  const fairCount = data?.items.filter((i) => i.metrics.overall_label === 'fair').length ?? 0;

  return (
    <div className="max-w-[1600px] mx-auto px-4 py-6 space-y-6">
      <div>
        <h2 className="text-lg font-semibold">Image Quality Assessment</h2>
        <p className="text-sm text-gray-400 mt-1">
          Scans all images using blur detection (Laplacian), exposure analysis, contrast measurement,
          and CLIP-IQA. Returns the N lowest quality images.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* Left panel: controls + selected detail */}
        <div className="lg:col-span-2 space-y-4">
          <div className="space-y-3 p-4 rounded-xl border border-gray-800 bg-gray-900">
            <div>
              <label className="block text-xs text-gray-400 mb-1">Number of worst images to return</label>
              <input
                type="number"
                value={count}
                onChange={(e) => setCount(Math.max(1, Math.min(100, Number(e.target.value))))}
                min={1}
                max={100}
                className="w-full px-3 py-1.5 rounded-md bg-gray-800 border border-gray-700 text-sm focus:outline-none focus:border-blue-500"
              />
            </div>

            <button
              onClick={handleScan}
              disabled={loading}
              className="w-full py-2.5 rounded-lg bg-rose-600 hover:bg-rose-500 disabled:opacity-40 disabled:cursor-not-allowed font-medium text-sm transition-colors flex items-center justify-center gap-2"
            >
              {loading ? <Loader2 size={16} className="animate-spin" /> : <AlertTriangle size={16} />}
              {loading ? 'Scanning all images...' : 'Find Low Quality Images'}
            </button>
          </div>

          {error && (
            <div className="p-3 rounded-lg bg-red-950 border border-red-800 text-red-300 text-sm">
              {error}
            </div>
          )}

          {data && (
            <div className="p-4 rounded-xl border border-gray-800 bg-gray-900 space-y-3">
              <h3 className="text-sm font-semibold">Scan Summary</h3>
              <div className="grid grid-cols-2 gap-3">
                <div className="text-center p-3 rounded-lg bg-gray-800">
                  <p className="text-2xl font-bold">{data.total_scanned}</p>
                  <p className="text-[10px] text-gray-500">Total scanned</p>
                </div>
                <div className="text-center p-3 rounded-lg bg-gray-800">
                  <p className="text-2xl font-bold">{data.returned}</p>
                  <p className="text-[10px] text-gray-500">Returned</p>
                </div>
                <div className="text-center p-3 rounded-lg bg-red-950/50 border border-red-900/50">
                  <p className="text-2xl font-bold text-red-400">{poorCount}</p>
                  <p className="text-[10px] text-red-600">Poor quality</p>
                </div>
                <div className="text-center p-3 rounded-lg bg-yellow-950/50 border border-yellow-900/50">
                  <p className="text-2xl font-bold text-yellow-400">{fairCount}</p>
                  <p className="text-[10px] text-yellow-600">Fair quality</p>
                </div>
              </div>
            </div>
          )}

          {/* Selected image detail */}
          {selected && (
            <div className="p-4 rounded-xl border border-gray-800 bg-gray-900 space-y-3">
              <h3 className="text-sm font-semibold">Detail</h3>
              <img
                src={`/api/static/${selected.uri}`}
                alt={selected.filename}
                className="w-full max-h-[400px] object-contain rounded-lg border border-gray-700 bg-black"
              />
              <p className="text-xs text-gray-400 truncate">{selected.filename}</p>
              <p className="text-xs text-gray-500">{selected.width}×{selected.height}</p>
              {selected.metrics.issues.length > 0 && (
                <div className="space-y-1">
                  {selected.metrics.issues.map((issue, i) => (
                    <p key={i} className="text-xs text-yellow-500 flex items-start gap-1">
                      <AlertTriangle size={10} className="mt-0.5 shrink-0" /> {issue}
                    </p>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Right panel: worst images list */}
        <div className="lg:col-span-3 space-y-3">
          {data && (
            <p className="text-xs text-gray-500 mb-2">
              Showing {data.returned} worst quality images (sorted by score, lowest first)
            </p>
          )}
          {data?.items.map((item, i) => (
            <button
              key={item.uri}
              onClick={() => setSelected(item)}
              className="w-full text-left hover:ring-1 hover:ring-blue-500/50 rounded-xl transition-all"
            >
              <QualityCard item={item} rank={i + 1} />
            </button>
          ))}

          {!data && !loading && (
            <div className="text-center py-16 text-gray-600">
              <AlertTriangle size={32} className="mx-auto mb-3 opacity-30" />
              <p className="text-sm">Click "Find Low Quality Images" to scan your dataset</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
