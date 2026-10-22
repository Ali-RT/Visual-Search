import { useState, useCallback } from 'react';
import { Loader2, SlidersHorizontal, X, Type, Columns2 } from 'lucide-react';
import { ImageDropzone } from '../components/ImageDropzone';
import { CropSelector, type CropRect } from '../components/CropSelector';
import { ResultsGrid } from '../components/ResultsGrid';
import { DetectionOverlay } from '../components/DetectionOverlay';
import { CompareView } from '../components/CompareView';
import { searchImages, searchHybrid, searchSimilar } from '../api';
import type { SearchResponse, SearchResultItem } from '../types';

const USE_CASES = ['', 'crosswalk', 'stopline', 'cones'];

export function SearchPage() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [crop, setCrop] = useState<CropRect | null>(null);
  const [textQuery, setTextQuery] = useState('');
  const [imageWeight, setImageWeight] = useState(0.7);
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [topK, setTopK] = useState(50);
  const [useCase, setUseCase] = useState('');
  const [selected, setSelected] = useState<SearchResultItem | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [compareItems, setCompareItems] = useState<SearchResultItem[]>([]);
  const [showCompare, setShowCompare] = useState(false);

  const handleFileSelect = useCallback((f: File) => {
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResponse(null);
    setSelected(null);
    setError(null);
    setCrop(null);
  }, []);

  const handleClear = useCallback(() => {
    setFile(null);
    setPreview(null);
    setResponse(null);
    setSelected(null);
    setCrop(null);
  }, []);

  const isHybrid = textQuery.trim().length > 0;

  const handleSearch = useCallback(async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const res = isHybrid
        ? await searchHybrid(file, textQuery.trim(), imageWeight, topK, useCase || undefined, crop)
        : await searchImages(file, topK, 200, useCase || undefined, crop);
      setResponse(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Search failed');
    } finally {
      setLoading(false);
    }
  }, [file, topK, useCase, crop, isHybrid, textQuery, imageWeight]);

  const handleFindSimilar = useCallback(async (item: SearchResultItem) => {
    setLoading(true);
    setError(null);
    setSelected(null);
    setPreview(`/api/static/${item.uri}`);
    setFile(null);
    setCrop(null);
    try {
      const res = await searchSimilar(item.image_id, topK, useCase || undefined);
      setResponse(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Similar search failed');
    } finally {
      setLoading(false);
    }
  }, [topK, useCase]);

  const handleToggleCompare = useCallback((item: SearchResultItem) => {
    setCompareItems((prev) => {
      const exists = prev.some((c) => c.image_id === item.image_id);
      if (exists) return prev.filter((c) => c.image_id !== item.image_id);
      if (prev.length >= 6) return prev;
      return [...prev, item];
    });
  }, []);

  const handleRemoveCompare = useCallback((item: SearchResultItem) => {
    setCompareItems((prev) => prev.filter((c) => c.image_id !== item.image_id));
  }, []);

  const searchLabel = loading
    ? 'Searching...'
    : isHybrid
      ? 'Hybrid Search'
      : crop
        ? 'Search Cropped Region'
        : 'Search';

  return (
    <div className="max-w-[1600px] mx-auto px-4 py-6 space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* Left panel: query + controls */}
        <div className="lg:col-span-2 space-y-4">
          <h2 className="text-lg font-semibold">Query Image</h2>

          {preview ? (
            <div className="space-y-2">
              <div className="relative">
                <CropSelector imageSrc={preview} crop={crop} onChange={setCrop} />
                <button
                  onClick={handleClear}
                  className="absolute top-2 right-2 z-10 p-1.5 bg-gray-900/80 rounded-full hover:bg-red-600 transition-colors"
                >
                  <X size={16} />
                </button>
              </div>
            </div>
          ) : (
            <ImageDropzone onFileSelect={handleFileSelect} />
          )}

          {/* Text refinement */}
          <div className="space-y-2">
            <label className="flex items-center gap-1.5 text-xs text-gray-400">
              <Type size={12} />
              Text refinement <span className="text-gray-600">(optional)</span>
            </label>
            <input
              type="text"
              value={textQuery}
              onChange={(e) => setTextQuery(e.target.value)}
              placeholder='e.g. "at night", "with heavy traffic"'
              className="w-full px-3 py-2 rounded-lg bg-gray-900 border border-gray-700 text-sm focus:outline-none focus:border-blue-500 placeholder:text-gray-600"
            />
            {isHybrid && (
              <div className="space-y-1">
                <div className="flex items-center justify-between text-[11px] text-gray-500">
                  <span>Image {Math.round(imageWeight * 100)}%</span>
                  <span>Text {Math.round((1 - imageWeight) * 100)}%</span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={100}
                  value={imageWeight * 100}
                  onChange={(e) => setImageWeight(Number(e.target.value) / 100)}
                  className="w-full h-1.5 rounded-full appearance-none bg-gray-700 accent-blue-500"
                />
              </div>
            )}
          </div>

          <button
            onClick={() => setShowSettings(!showSettings)}
            className="flex items-center gap-1.5 text-sm text-gray-400 hover:text-white transition-colors"
          >
            <SlidersHorizontal size={14} />
            Settings
          </button>

          {showSettings && (
            <div className="space-y-3 p-3 rounded-lg border border-gray-800 bg-gray-900">
              <div>
                <label className="block text-xs text-gray-400 mb-1">Top K</label>
                <input
                  type="number"
                  value={topK}
                  onChange={(e) => setTopK(Number(e.target.value))}
                  min={1}
                  max={200}
                  className="w-full px-3 py-1.5 rounded-md bg-gray-800 border border-gray-700 text-sm focus:outline-none focus:border-blue-500"
                />
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">Filter by Use Case</label>
                <select
                  value={useCase}
                  onChange={(e) => setUseCase(e.target.value)}
                  className="w-full px-3 py-1.5 rounded-md bg-gray-800 border border-gray-700 text-sm focus:outline-none focus:border-blue-500"
                >
                  {USE_CASES.map((uc) => (
                    <option key={uc} value={uc}>
                      {uc || 'All use cases'}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          )}

          <button
            onClick={handleSearch}
            disabled={!file || loading}
            className="w-full py-2.5 rounded-lg bg-blue-600 hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed font-medium text-sm transition-colors flex items-center justify-center gap-2"
          >
            {loading ? <Loader2 size={16} className="animate-spin" /> : null}
            {searchLabel}
          </button>

          {crop && (
            <p className="text-xs text-blue-400">
              Searching cropped region: {crop.w}×{crop.h}px at ({crop.x}, {crop.y})
            </p>
          )}

          {error && (
            <div className="p-3 rounded-lg bg-red-950 border border-red-800 text-red-300 text-sm">
              {error}
            </div>
          )}

          {response && (
            <div className="text-xs text-gray-500 space-y-1">
              <p>{response.results.length} results</p>
              <p>Model: {response.model_version}</p>
              <p>Index: {response.index_version}</p>
            </div>
          )}

          {selected && (
            <div className="p-3 rounded-lg border border-blue-800 bg-blue-950/30 space-y-2">
              <h3 className="text-sm font-semibold text-blue-300">Selected</h3>
              <DetectionOverlay imageUrl={`/api/static/${selected.uri}`} />
              <div className="text-xs text-gray-400 space-y-0.5">
                <p>Score: {(selected.score * 100).toFixed(2)}%</p>
                <p>Position: #{selected.position}</p>
                {selected.use_case && <p>Use case: {selected.use_case}</p>}
                <p className="truncate">ID: {selected.image_id}</p>
              </div>
              <button
                onClick={() => handleFindSimilar(selected)}
                disabled={loading}
                className="w-full py-1.5 rounded-md bg-blue-600 hover:bg-blue-500 disabled:opacity-40 text-xs font-medium transition-colors"
              >
                Find Similar
              </button>
            </div>
          )}
        </div>

        {/* Right panel: results grid */}
        <div className="lg:col-span-3 min-w-0">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold">
              Results
              {response && (
                <span className="ml-2 text-sm font-normal text-gray-500">
                  Top {response.results.length}
                </span>
              )}
            </h2>

            {compareItems.length > 0 && (
              <button
                onClick={() => setShowCompare(true)}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-emerald-600 hover:bg-emerald-500 text-white transition-colors"
              >
                <Columns2 size={14} />
                Compare ({compareItems.length})
              </button>
            )}
          </div>

          <ResultsGrid
            results={response?.results ?? []}
            onSelect={setSelected}
            onFindSimilar={handleFindSimilar}
            compareItems={compareItems}
            onToggleCompare={handleToggleCompare}
          />
        </div>
      </div>

      {showCompare && (
        <CompareView
          items={compareItems}
          onRemove={handleRemoveCompare}
          onClose={() => setShowCompare(false)}
        />
      )}
    </div>
  );
}
