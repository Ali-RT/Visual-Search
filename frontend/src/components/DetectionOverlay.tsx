import { useState, useCallback, useRef, useEffect } from 'react';
import { Loader2, ScanSearch, Layers, X, Link2 } from 'lucide-react';
import { detectObjects, segmentImage } from '../api';
import type { DetectionItem, SegmentationResponse } from '../types';

const COLORS = [
  '#3b82f6', '#ef4444', '#22c55e', '#f59e0b', '#a855f7',
  '#ec4899', '#06b6d4', '#f97316', '#14b8a6', '#8b5cf6',
];

function colorForLabel(label: string): string {
  let hash = 0;
  for (let i = 0; i < label.length; i++) hash = label.charCodeAt(i) + ((hash << 5) - hash);
  return COLORS[Math.abs(hash) % COLORS.length];
}

// Cityscapes classes that are detectable objects (not background/stuff)
const OBJECT_CLASSES = new Set([
  'pole', 'traffic light', 'traffic sign',
  'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle',
]);

function buildPromptFromSegments(seg: SegmentationResponse): string | undefined {
  const found = seg.segments
    .filter((s) => OBJECT_CLASSES.has(s.label) && s.percentage >= 0.1)
    .map((s) => s.label);

  if (found.length === 0) return undefined;
  return found.map((l) => `${l}.`).join(' ');
}

interface Props {
  imageUrl: string;
}

export function DetectionOverlay({ imageUrl }: Props) {
  const [detections, setDetections] = useState<DetectionItem[] | null>(null);
  const [segmentation, setSegmentation] = useState<SegmentationResponse | null>(null);
  const [imgSize, setImgSize] = useState<{ w: number; h: number } | null>(null);
  const [loadingDet, setLoadingDet] = useState(false);
  const [loadingSeg, setLoadingSeg] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [usedPrompt, setUsedPrompt] = useState<string | null>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [displayRect, setDisplayRect] = useState<DOMRect | null>(null);

  const updateRect = useCallback(() => {
    if (imgRef.current) setDisplayRect(imgRef.current.getBoundingClientRect());
  }, []);

  useEffect(() => {
    window.addEventListener('resize', updateRect);
    return () => window.removeEventListener('resize', updateRect);
  }, [updateRect]);

  useEffect(() => {
    setDetections(null);
    setSegmentation(null);
    setError(null);
    setUsedPrompt(null);
  }, [imageUrl]);

  const handleSegment = useCallback(async () => {
    setLoadingSeg(true);
    setError(null);
    try {
      const res = await segmentImage(imageUrl);
      setSegmentation(res);
      setDetections(null);
      setUsedPrompt(null);
      updateRect();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Segmentation failed');
    } finally {
      setLoadingSeg(false);
    }
  }, [imageUrl, updateRect]);

  const handleDetect = useCallback(async () => {
    setLoadingDet(true);
    setError(null);
    try {
      const prompt = segmentation ? buildPromptFromSegments(segmentation) : undefined;
      setUsedPrompt(prompt ?? null);
      const res = await detectObjects(imageUrl, prompt);
      setDetections(res.detections);
      setImgSize({ w: res.width, h: res.height });
      updateRect();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Detection failed');
    } finally {
      setLoadingDet(false);
    }
  }, [imageUrl, segmentation, updateRect]);

  const handleClear = useCallback(() => {
    setDetections(null);
    setSegmentation(null);
    setUsedPrompt(null);
  }, []);

  const scaleBox = (box: [number, number, number, number]) => {
    if (!imgSize || !displayRect || !containerRef.current) return null;
    const containerRect = containerRef.current.getBoundingClientRect();
    const sx = displayRect.width / imgSize.w;
    const sy = displayRect.height / imgSize.h;
    const offsetX = displayRect.left - containerRect.left;
    const offsetY = displayRect.top - containerRect.top;
    return {
      left: offsetX + box[0] * sx,
      top: offsetY + box[1] * sy,
      width: (box[2] - box[0]) * sx,
      height: (box[3] - box[1]) * sy,
    };
  };

  const showImage = segmentation ? `data:image/jpeg;base64,${segmentation.colored_mask_b64}` : imageUrl;
  const hasResults = detections || segmentation;
  const uniqueLabels = detections ? [...new Set(detections.map((d) => d.label))] : [];
  const isGuided = segmentation !== null;

  return (
    <div className="space-y-3">
      <div ref={containerRef} className="relative inline-block w-full">
        <img
          ref={imgRef}
          src={showImage}
          alt="Analysis"
          className="w-full max-h-[560px] object-contain rounded-xl border border-gray-700 bg-gray-900"
          onLoad={updateRect}
          draggable={false}
        />

        {detections && detections.map((d, i) => {
          const rect = scaleBox(d.box);
          if (!rect) return null;
          const color = colorForLabel(d.label);
          return (
            <div key={i} className="absolute pointer-events-none" style={rect}>
              <div
                className="absolute inset-0 rounded-sm"
                style={{ border: `2px solid ${color}`, background: `${color}15` }}
              />
              <span
                className="absolute -top-5 left-0 px-1.5 py-0.5 text-[10px] font-medium rounded whitespace-nowrap"
                style={{ backgroundColor: color, color: '#fff' }}
              >
                {d.label} {(d.score * 100).toFixed(0)}%
              </span>
            </div>
          );
        })}
      </div>

      {/* Action buttons */}
      <div className="flex items-center gap-2 flex-wrap">
        <button
          onClick={handleSegment}
          disabled={loadingSeg}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors disabled:opacity-40 ${
            segmentation
              ? 'bg-teal-800 text-teal-200 border border-teal-600'
              : 'bg-teal-600 hover:bg-teal-500'
          }`}
        >
          {loadingSeg ? <Loader2 size={14} className="animate-spin" /> : <Layers size={14} />}
          {loadingSeg ? 'Segmenting...' : segmentation ? 'Re-Segment' : '1. Segment'}
        </button>

        <button
          onClick={handleDetect}
          disabled={loadingDet}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium bg-amber-600 hover:bg-amber-500 disabled:opacity-40 transition-colors"
        >
          {loadingDet ? <Loader2 size={14} className="animate-spin" /> : <ScanSearch size={14} />}
          {loadingDet
            ? 'Detecting...'
            : isGuided
              ? '2. Detect (guided)'
              : 'Detect Objects'}
        </button>

        {hasResults && (
          <button
            onClick={handleClear}
            className="flex items-center gap-1 px-2 py-1.5 rounded-md bg-gray-800 text-gray-400 hover:text-white text-xs transition-colors"
          >
            <X size={12} />
            Clear
          </button>
        )}

        {error && <span className="text-xs text-red-400">{error}</span>}
      </div>

      {/* Guided detection info */}
      {usedPrompt && (
        <div className="flex items-start gap-1.5 text-[11px] text-teal-400 bg-teal-950/40 border border-teal-900/50 rounded-md px-2.5 py-1.5">
          <Link2 size={12} className="mt-0.5 shrink-0" />
          <span>
            <strong>Segmentation-guided:</strong> detection targeted{' '}
            <span className="text-teal-300">{usedPrompt.replace(/\./g, ',').replace(/,\s*$/, '')}</span>
          </span>
        </div>
      )}

      {/* Segmentation results */}
      {segmentation && (
        <div className="space-y-1.5">
          <div className="flex flex-wrap gap-1.5">
            {segmentation.segments
              .filter((s) => s.percentage >= 1)
              .map((s) => {
                const c = `rgb(${s.color[0]},${s.color[1]},${s.color[2]})`;
                const isObj = OBJECT_CLASSES.has(s.label);
                return (
                  <span
                    key={s.class_id}
                    className={`px-2 py-0.5 rounded-full text-[10px] font-medium text-white ${isObj ? 'ring-1 ring-white/40' : ''}`}
                    style={{ backgroundColor: c }}
                    title={isObj ? 'Object class → will guide detection' : 'Background class'}
                  >
                    {s.label} {s.percentage.toFixed(1)}%
                    {isObj && ' ⬆'}
                  </span>
                );
              })}
          </div>
          <p className="text-[10px] text-gray-600">
            Classes marked with ⬆ guide the detection prompt
          </p>
        </div>
      )}

      {/* Detection results */}
      {detections && uniqueLabels.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {uniqueLabels.map((label) => {
            const count = detections.filter((d) => d.label === label).length;
            const color = colorForLabel(label);
            return (
              <span
                key={label}
                className="px-2 py-0.5 rounded-full text-[10px] font-medium"
                style={{ backgroundColor: `${color}20`, color, border: `1px solid ${color}40` }}
              >
                {label} ×{count}
              </span>
            );
          })}
        </div>
      )}

      {detections && (
        <span className="text-[11px] text-gray-500">
          {detections.length} object{detections.length !== 1 ? 's' : ''} detected
        </span>
      )}
    </div>
  );
}
