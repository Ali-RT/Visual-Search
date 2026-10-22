import { useRef, useState, useCallback, useEffect } from 'react';
import { Crop, RotateCcw } from 'lucide-react';

export interface CropRect {
  x: number;
  y: number;
  w: number;
  h: number;
}

interface Props {
  imageSrc: string;
  crop: CropRect | null;
  onChange: (crop: CropRect | null) => void;
}

export function CropSelector({ imageSrc, crop, onChange }: Props) {
  const imgRef = useRef<HTMLImageElement>(null);
  const [drawing, setDrawing] = useState(false);
  const [startPx, setStartPx] = useState({ x: 0, y: 0 });
  const [currentPx, setCurrentPx] = useState({ x: 0, y: 0 });
  const [enabled, setEnabled] = useState(false);
  const [naturalSize, setNaturalSize] = useState({ w: 1, h: 1 });

  useEffect(() => {
    setEnabled(false);
    onChange(null);
  }, [imageSrc, onChange]);

  const pxToImage = useCallback(
    (clientX: number, clientY: number) => {
      const img = imgRef.current;
      if (!img) return { x: 0, y: 0 };
      const r = img.getBoundingClientRect();
      const sx = naturalSize.w / r.width;
      const sy = naturalSize.h / r.height;
      return {
        x: Math.round(Math.max(0, Math.min(naturalSize.w, (clientX - r.left) * sx))),
        y: Math.round(Math.max(0, Math.min(naturalSize.h, (clientY - r.top) * sy))),
      };
    },
    [naturalSize],
  );

  const imgToPercent = useCallback(
    (ix: number, iy: number, iw: number, ih: number) => ({
      left: `${(ix / naturalSize.w) * 100}%`,
      top: `${(iy / naturalSize.h) * 100}%`,
      width: `${(iw / naturalSize.w) * 100}%`,
      height: `${(ih / naturalSize.h) * 100}%`,
    }),
    [naturalSize],
  );

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (!enabled) return;
      e.preventDefault();
      const pt = pxToImage(e.clientX, e.clientY);
      setStartPx(pt);
      setCurrentPx(pt);
      setDrawing(true);
      onChange(null);
    },
    [enabled, pxToImage, onChange],
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (!drawing) return;
      setCurrentPx(pxToImage(e.clientX, e.clientY));
    },
    [drawing, pxToImage],
  );

  const handleMouseUp = useCallback(() => {
    if (!drawing) return;
    setDrawing(false);
    const x = Math.min(startPx.x, currentPx.x);
    const y = Math.min(startPx.y, currentPx.y);
    const w = Math.abs(currentPx.x - startPx.x);
    const h = Math.abs(currentPx.y - startPx.y);
    if (w > 10 && h > 10) {
      onChange({
        x: Math.max(0, x),
        y: Math.max(0, y),
        w: Math.min(w, naturalSize.w - x),
        h: Math.min(h, naturalSize.h - y),
      });
    }
  }, [drawing, startPx, currentPx, naturalSize, onChange]);

  const activeRect = (() => {
    const src = drawing ? { x1: startPx.x, y1: startPx.y, x2: currentPx.x, y2: currentPx.y } : crop ? { x1: crop.x, y1: crop.y, x2: crop.x + crop.w, y2: crop.y + crop.h } : null;
    if (!src) return null;
    const x = Math.min(src.x1, src.x2);
    const y = Math.min(src.y1, src.y2);
    const w = Math.abs(src.x2 - src.x1);
    const h = Math.abs(src.y2 - src.y1);
    if (w < 2 && h < 2) return null;
    return { x, y, w, h };
  })();

  const overlayStrips = activeRect
    ? {
        top: imgToPercent(0, 0, naturalSize.w, activeRect.y),
        bottom: imgToPercent(0, activeRect.y + activeRect.h, naturalSize.w, naturalSize.h - activeRect.y - activeRect.h),
        left: imgToPercent(0, activeRect.y, activeRect.x, activeRect.h),
        right: imgToPercent(activeRect.x + activeRect.w, activeRect.y, naturalSize.w - activeRect.x - activeRect.w, activeRect.h),
      }
    : null;

  return (
    <div className="space-y-2">
      {/* Image + overlay container — positioned relative to the <img> */}
      <div
        className="relative inline-block w-full select-none"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        style={{ cursor: enabled ? 'crosshair' : 'default' }}
      >
        <img
          ref={imgRef}
          src={imageSrc}
          alt="Query"
          className="w-full max-h-[560px] object-contain rounded-xl border border-gray-700 bg-gray-900"
          draggable={false}
          onLoad={() => {
            const img = imgRef.current;
            if (img) setNaturalSize({ w: img.naturalWidth, h: img.naturalHeight });
          }}
        />

        {/* Dark overlay strips (everything outside selection) */}
        {enabled && overlayStrips && (
          <>
            <div className="absolute bg-black/50 pointer-events-none" style={overlayStrips.top} />
            <div className="absolute bg-black/50 pointer-events-none" style={overlayStrips.bottom} />
            <div className="absolute bg-black/50 pointer-events-none" style={overlayStrips.left} />
            <div className="absolute bg-black/50 pointer-events-none" style={overlayStrips.right} />
          </>
        )}

        {/* Selection border */}
        {enabled && activeRect && (
          <div
            className="absolute border-2 border-blue-400 pointer-events-none rounded-sm"
            style={imgToPercent(activeRect.x, activeRect.y, activeRect.w, activeRect.h)}
          />
        )}
      </div>

      {/* Crop controls */}
      <div className="flex items-center gap-2">
        <button
          onClick={() => {
            setEnabled(!enabled);
            if (enabled) onChange(null);
          }}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
            enabled
              ? 'bg-blue-600 text-white'
              : 'bg-gray-800 text-gray-400 hover:text-white hover:bg-gray-700'
          }`}
        >
          <Crop size={14} />
          {enabled ? 'Crop ON' : 'Crop Region'}
        </button>

        {crop && (
          <>
            <button
              onClick={() => onChange(null)}
              className="flex items-center gap-1 px-2 py-1.5 rounded-md bg-gray-800 text-gray-400 hover:text-white text-xs transition-colors"
            >
              <RotateCcw size={12} />
              Reset
            </button>
            <span className="text-[10px] text-gray-500 font-mono">
              {crop.w}×{crop.h}px at ({crop.x}, {crop.y})
            </span>
          </>
        )}
      </div>
    </div>
  );
}
