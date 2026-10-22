import { AlertTriangle, Cpu, ScanSearch, Layers } from 'lucide-react';
import type { HybridTagItem, HybridTagResponse } from '../types';

interface Props {
  data: HybridTagResponse;
}

const SOURCE_CONFIG = {
  clip: { icon: Cpu, color: 'text-purple-400', bg: 'bg-purple-900/30', border: 'border-purple-800', label: 'CLIP' },
  detection: { icon: ScanSearch, color: 'text-amber-400', bg: 'bg-amber-900/30', border: 'border-amber-800', label: 'Detection' },
  segmentation: { icon: Layers, color: 'text-teal-400', bg: 'bg-teal-900/30', border: 'border-teal-800', label: 'Segmentation' },
};

function confidenceColor(confidence: number): string {
  if (confidence >= 0.8) return 'text-emerald-400';
  if (confidence >= 0.6) return 'text-yellow-400';
  return 'text-red-400';
}

function confidenceBarColor(confidence: number): string {
  if (confidence >= 0.8) return 'bg-emerald-500';
  if (confidence >= 0.6) return 'bg-yellow-500';
  return 'bg-red-500';
}

function TagCard({ tag }: { tag: HybridTagItem }) {
  const src = SOURCE_CONFIG[tag.source];
  const Icon = src.icon;
  const scores = tag.details?.scores as Record<string, number> | undefined;

  return (
    <div className={`rounded-lg border ${src.border} ${src.bg} p-3 space-y-2`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <h4 className="text-xs font-semibold text-gray-300 uppercase tracking-wide">
            {tag.group.replace(/_/g, ' ')}
          </h4>
          {tag.uncertain && (
            <span title="Low confidence — may be inaccurate">
              <AlertTriangle size={12} className="text-yellow-500" />
            </span>
          )}
        </div>
        <div className="flex items-center gap-1.5">
          <Icon size={11} className={src.color} />
          <span className={`text-[10px] ${src.color}`}>{src.label}</span>
        </div>
      </div>

      <div className="flex items-center gap-2">
        <span className="text-sm font-medium text-white">{tag.label}</span>
        <span className={`text-xs font-mono ${confidenceColor(tag.confidence)}`}>
          {(tag.confidence * 100).toFixed(0)}%
        </span>
      </div>

      {/* Show bar chart for CLIP scores */}
      {scores && (
        <div className="space-y-1 pt-1">
          {Object.entries(scores)
            .sort(([, a], [, b]) => b - a)
            .map(([label, score]) => (
              <div key={label} className="flex items-center gap-2">
                <span className="w-16 text-[10px] text-gray-500 text-right truncate">{label}</span>
                <div className="flex-1 h-3 bg-gray-800 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all ${
                      label === tag.label ? confidenceBarColor(score) : 'bg-gray-600'
                    }`}
                    style={{ width: `${Math.max(score * 100, 1)}%` }}
                  />
                </div>
                <span className="w-10 text-[10px] text-gray-500 font-mono">{(score * 100).toFixed(0)}%</span>
              </div>
            ))}
        </div>
      )}

      {/* Show count/detail for detection */}
      {tag.source === 'detection' && tag.details && !scores && (
        <div className="flex flex-wrap gap-1.5 pt-1">
          {Object.entries(tag.details).map(([k, v]) => {
            if (typeof v === 'object' && v !== null) {
              return Object.entries(v as Record<string, number>).map(([label, count]) => (
                <span key={label} className="px-1.5 py-0.5 rounded text-[10px] bg-gray-800 text-gray-300">
                  {label}: {count}
                </span>
              ));
            }
            return (
              <span key={k} className="px-1.5 py-0.5 rounded text-[10px] bg-gray-800 text-gray-300">
                {k.replace(/_/g, ' ')}: {String(v)}
              </span>
            );
          })}
        </div>
      )}

      {/* Show percentage for segmentation */}
      {tag.source === 'segmentation' && tag.details && !scores && (
        <div className="flex flex-wrap gap-1.5 pt-1">
          {Object.entries(tag.details).map(([k, v]) => (
            <span key={k} className="px-1.5 py-0.5 rounded text-[10px] bg-gray-800 text-gray-300">
              {k.replace(/_/g, ' ')}: {typeof v === 'number' ? `${v}%` : String(v)}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function TagSection({ title, icon: Icon, color, tags }: { title: string; icon: typeof Cpu; color: string; tags: HybridTagItem[] }) {
  if (tags.length === 0) return null;
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <Icon size={16} className={color} />
        <h3 className="text-sm font-semibold">{title}</h3>
        <span className="text-[10px] text-gray-600">{tags.length} tags</span>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {tags.map((t) => (
          <TagCard key={`${t.source}-${t.group}`} tag={t} />
        ))}
      </div>
    </div>
  );
}

export function HybridTagResults({ data }: Props) {
  const allTags = [...data.scene_tags, ...data.object_tags, ...data.layout_tags];
  const uncertainCount = allTags.filter((t) => t.uncertain).length;

  return (
    <div className="space-y-6">
      {/* Summary chips */}
      <div className="space-y-2">
        <div className="flex flex-wrap gap-1.5">
          {Object.entries(data.summary).map(([group, label]) => {
            const tag = allTags.find((t) => t.group === group);
            const src = tag ? SOURCE_CONFIG[tag.source] : SOURCE_CONFIG.clip;
            return (
              <span
                key={group}
                className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-medium border ${src.border} ${src.bg}`}
              >
                <span className="text-gray-400">{group.replace(/_/g, ' ')}:</span>
                <span className="text-white">{label}</span>
                {tag?.uncertain && <AlertTriangle size={10} className="text-yellow-500" />}
              </span>
            );
          })}
        </div>
        {uncertainCount > 0 && (
          <p className="text-[10px] text-yellow-600 flex items-center gap-1">
            <AlertTriangle size={10} />
            {uncertainCount} tag{uncertainCount !== 1 ? 's' : ''} have low confidence and may be inaccurate
          </p>
        )}
      </div>

      {/* Sections by source */}
      <TagSection title="Scene Understanding" icon={Cpu} color="text-purple-400" tags={data.scene_tags} />
      <TagSection title="Object Detection" icon={ScanSearch} color="text-amber-400" tags={data.object_tags} />
      <TagSection title="Scene Layout" icon={Layers} color="text-teal-400" tags={data.layout_tags} />
    </div>
  );
}
