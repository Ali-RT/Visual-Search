import { useState, useCallback } from 'react';
import { Loader2, Cpu, Zap } from 'lucide-react';
import { ImageDropzone } from '../components/ImageDropzone';
import { TagResults } from '../components/TagResults';
import { HybridTagResults } from '../components/HybridTagResults';
import { tagImage, tagImageHybrid } from '../api';
import type { TagResponse, HybridTagResponse } from '../types';

type Mode = 'hybrid' | 'clip';

export function TagPage() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [mode, setMode] = useState<Mode>('hybrid');
  const [clipTags, setClipTags] = useState<TagResponse | null>(null);
  const [hybridTags, setHybridTags] = useState<HybridTagResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileSelect = useCallback((f: File) => {
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setClipTags(null);
    setHybridTags(null);
    setError(null);
  }, []);

  const handleClear = useCallback(() => {
    setFile(null);
    setPreview(null);
    setClipTags(null);
    setHybridTags(null);
  }, []);

  const handleTag = useCallback(async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      if (mode === 'hybrid') {
        const res = await tagImageHybrid(file);
        setHybridTags(res);
        setClipTags(null);
      } else {
        const res = await tagImage(file);
        setClipTags(res);
        setHybridTags(null);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Tagging failed');
    } finally {
      setLoading(false);
    }
  }, [file, mode]);

  return (
    <div className="max-w-[1600px] mx-auto px-4 py-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold">Image Tagging</h2>
          <p className="text-sm text-gray-400 mt-1">
            {mode === 'hybrid'
              ? 'Three-stage analysis: CLIP scene understanding + Grounding DINO object detection + SegFormer scene layout.'
              : 'SigLIP2 zero-shot classification for scene properties.'}
          </p>
        </div>

        {/* Mode toggle */}
        <div className="flex rounded-lg border border-gray-700 overflow-hidden">
          <button
            onClick={() => setMode('hybrid')}
            className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium transition-colors ${
              mode === 'hybrid' ? 'bg-emerald-600 text-white' : 'bg-gray-900 text-gray-400 hover:text-white'
            }`}
          >
            <Zap size={12} />
            Hybrid
          </button>
          <button
            onClick={() => setMode('clip')}
            className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium transition-colors ${
              mode === 'clip' ? 'bg-purple-600 text-white' : 'bg-gray-900 text-gray-400 hover:text-white'
            }`}
          >
            <Cpu size={12} />
            CLIP Only
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* Left: image preview + controls */}
        <div className="lg:col-span-2 space-y-4">
          <ImageDropzone onFileSelect={handleFileSelect} preview={preview} onClear={handleClear} />
          <button
            onClick={handleTag}
            disabled={!file || loading}
            className={`w-full py-2.5 rounded-lg disabled:opacity-40 disabled:cursor-not-allowed font-medium text-sm transition-colors flex items-center justify-center gap-2 ${
              mode === 'hybrid'
                ? 'bg-emerald-600 hover:bg-emerald-500'
                : 'bg-purple-600 hover:bg-purple-500'
            }`}
          >
            {loading ? <Loader2 size={16} className="animate-spin" /> : null}
            {loading
              ? mode === 'hybrid' ? 'Running 3-stage analysis...' : 'Analyzing...'
              : mode === 'hybrid' ? 'Hybrid Analysis' : 'CLIP Tags'}
          </button>

          {error && (
            <div className="p-3 rounded-lg bg-red-950 border border-red-800 text-red-300 text-sm">
              {error}
            </div>
          )}
        </div>

        {/* Right: tag results */}
        <div className="lg:col-span-3 space-y-4">
          {hybridTags && <HybridTagResults data={hybridTags} />}

          {clipTags && (
            <>
              <div className="flex flex-wrap gap-2">
                {Object.entries(clipTags.flat_labels).map(([group, label]) => (
                  <span
                    key={group}
                    className="px-2.5 py-1 rounded-full text-xs font-medium bg-gray-800 border border-gray-700"
                  >
                    {group.replace(/_/g, ' ')}: <span className="text-white">{label}</span>
                  </span>
                ))}
              </div>
              <TagResults data={clipTags} />
            </>
          )}
        </div>
      </div>
    </div>
  );
}
