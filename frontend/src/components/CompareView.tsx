import { useState } from 'react';
import { X, ChevronDown, ChevronUp, Layers, ScanSearch, Loader2 } from 'lucide-react';
import { detectObjects, segmentImage } from '../api';
import type { SearchResultItem, DetectionItem, SegmentationResponse } from '../types';

interface Props {
  items: SearchResultItem[];
  onRemove: (item: SearchResultItem) => void;
  onClose: () => void;
}

interface CardState {
  segmentation: SegmentationResponse | null;
  detections: DetectionItem[] | null;
  imgSize: { w: number; h: number } | null;
  loadingSeg: boolean;
  loadingDet: boolean;
}

function CompareCard({
  item,
  onRemove,
}: {
  item: SearchResultItem;
  onRemove: () => void;
}) {
  const [state, setState] = useState<CardState>({
    segmentation: null,
    detections: null,
    imgSize: null,
    loadingSeg: false,
    loadingDet: false,
  });
  const [expanded, setExpanded] = useState(true);
  const imageUrl = `/api/static/${item.uri}`;

  const handleSegment = async () => {
    setState((s) => ({ ...s, loadingSeg: true }));
    try {
      const res = await segmentImage(imageUrl);
      setState((s) => ({ ...s, segmentation: res, loadingSeg: false }));
    } catch {
      setState((s) => ({ ...s, loadingSeg: false }));
    }
  };

  const handleDetect = async () => {
    setState((s) => ({ ...s, loadingDet: true }));
    try {
      const res = await detectObjects(imageUrl);
      setState((s) => ({
        ...s,
        detections: res.detections,
        imgSize: { w: res.width, h: res.height },
        loadingDet: false,
      }));
    } catch {
      setState((s) => ({ ...s, loadingDet: false }));
    }
  };

  const showSrc = state.segmentation
    ? `data:image/jpeg;base64,${state.segmentation.colored_mask_b64}`
    : imageUrl;

  return (
    <div className="flex-1 min-w-0 border border-gray-800 rounded-xl bg-gray-900/50 overflow-hidden flex flex-col">
      {/* Header */}
      <div className="flex items-center gap-2 px-3 py-2 border-b border-gray-800 bg-gray-900">
        <span className="font-mono text-blue-300 text-xs">#{item.position}</span>
        <span className="text-xs text-gray-400 truncate flex-1">{item.uri.split('/').pop()}</span>
        <span className="text-xs text-emerald-400 font-medium">{(item.score * 100).toFixed(1)}%</span>
        <button onClick={() => setExpanded(!expanded)} className="p-0.5 text-gray-500 hover:text-white">
          {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </button>
        <button onClick={onRemove} className="p-0.5 text-gray-500 hover:text-red-400">
          <X size={14} />
        </button>
      </div>

      {expanded && (
        <>
          {/* Image */}
          <div className="relative">
            <img src={showSrc} alt={item.image_id} className="w-full object-contain max-h-[400px] bg-black" />
          </div>

          {/* Metadata */}
          <div className="px-3 py-2 space-y-2 text-xs">
            <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-gray-400">
              <span>Score</span>
              <span className="text-white font-medium">{(item.score * 100).toFixed(2)}%</span>
              <span>Position</span>
              <span className="text-white">#{item.position}</span>
              {item.use_case && (
                <>
                  <span>Use case</span>
                  <span className="text-blue-300">{item.use_case}</span>
                </>
              )}
              {item.label && (
                <>
                  <span>Label</span>
                  <span className="text-purple-300">{item.label}</span>
                </>
              )}
            </div>

            {/* Segmentation chips */}
            {state.segmentation && (
              <div className="flex flex-wrap gap-1">
                {state.segmentation.segments
                  .filter((s) => s.percentage >= 1)
                  .map((s) => (
                    <span
                      key={s.class_id}
                      className="px-1.5 py-0.5 rounded-full text-[9px] font-medium text-white"
                      style={{ backgroundColor: `rgb(${s.color[0]},${s.color[1]},${s.color[2]})` }}
                    >
                      {s.label} {s.percentage.toFixed(0)}%
                    </span>
                  ))}
              </div>
            )}

            {/* Detection chips */}
            {state.detections && (
              <div className="flex flex-wrap gap-1">
                {[...new Set(state.detections.map((d) => d.label))].map((label) => {
                  const count = state.detections!.filter((d) => d.label === label).length;
                  return (
                    <span
                      key={label}
                      className="px-1.5 py-0.5 rounded-full text-[9px] font-medium bg-amber-900/50 text-amber-300 border border-amber-700/50"
                    >
                      {label} ×{count}
                    </span>
                  );
                })}
              </div>
            )}

            {/* Action buttons */}
            <div className="flex gap-1.5 pt-1">
              <button
                onClick={handleSegment}
                disabled={state.loadingSeg}
                className="flex items-center gap-1 px-2 py-1 rounded text-[10px] font-medium bg-teal-900/50 text-teal-300 border border-teal-700/50 hover:bg-teal-800/50 disabled:opacity-40 transition-colors"
              >
                {state.loadingSeg ? <Loader2 size={10} className="animate-spin" /> : <Layers size={10} />}
                Segment
              </button>
              <button
                onClick={handleDetect}
                disabled={state.loadingDet}
                className="flex items-center gap-1 px-2 py-1 rounded text-[10px] font-medium bg-amber-900/50 text-amber-300 border border-amber-700/50 hover:bg-amber-800/50 disabled:opacity-40 transition-colors"
              >
                {state.loadingDet ? <Loader2 size={10} className="animate-spin" /> : <ScanSearch size={10} />}
                Detect
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

export function CompareView({ items, onRemove, onClose }: Props) {
  if (items.length === 0) return null;

  return (
    <div className="fixed inset-0 z-50 bg-gray-950/95 overflow-auto">
      {/* Top bar */}
      <div className="sticky top-0 z-10 flex items-center justify-between px-6 py-3 border-b border-gray-800 bg-gray-950/90 backdrop-blur-sm">
        <div className="flex items-center gap-3">
          <h2 className="text-lg font-semibold">Compare</h2>
          <span className="text-sm text-gray-500">{items.length} images</span>
        </div>
        <button
          onClick={onClose}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm bg-gray-800 text-gray-300 hover:text-white hover:bg-gray-700 transition-colors"
        >
          <X size={14} />
          Close
        </button>
      </div>

      {/* Cards row */}
      <div className="p-6">
        <div className={`flex gap-4 ${items.length <= 3 ? 'justify-center' : ''}`}>
          {items.map((item) => (
            <CompareCard key={item.image_id} item={item} onRemove={() => onRemove(item)} />
          ))}
        </div>
      </div>
    </div>
  );
}
