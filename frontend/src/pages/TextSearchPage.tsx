import { useState, useCallback } from 'react';
import { Loader2, SlidersHorizontal, Type, Columns2 } from 'lucide-react';
import { ResultsGrid } from '../components/ResultsGrid';
import { DetectionOverlay } from '../components/DetectionOverlay';
import { CompareView } from '../components/CompareView';
import { searchByText, searchSimilar } from '../api';
import type { SearchResponse, SearchResultItem } from '../types';

const USE_CASES = ['', 'crosswalk', 'stopline', 'cones'];

const EXAMPLE_QUERIES = [
  'a road with a crosswalk',
  'highway at night with headlights',
  'urban intersection with traffic',
  'rural road with trees',
  'snowy road conditions',
  'cars stopped at a red light',
];

export function TextSearchPage() {
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [topK, setTopK] = useState(50);
  const [useCase, setUseCase] = useState('');
  const [selected, setSelected] = useState<SearchResultItem | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [compareItems, setCompareItems] = useState<SearchResultItem[]>([]);
  const [showCompare, setShowCompare] = useState(false);

  const handleSearch = useCallback(async () => {
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await searchByText(query.trim(), topK, useCase || undefined);
      setResponse(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Search failed');
    } finally {
      setLoading(false);
    }
  }, [query, topK, useCase]);

  const handleFindSimilar = useCallback(async (item: SearchResultItem) => {
    setLoading(true);
    setError(null);
    setSelected(null);
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

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSearch();
      }
    },
    [handleSearch],
  );

  return (
    <div className="max-w-[1600px] mx-auto px-4 py-6 space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* Left panel */}
        <div className="lg:col-span-2 space-y-4">
          <h2 className="text-lg font-semibold">Text Query</h2>

          <div className="space-y-3">
            <div className="relative">
              <Type size={16} className="absolute left-3 top-3.5 text-gray-500" />
              <textarea
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Describe what you're looking for..."
                rows={3}
                className="w-full pl-10 pr-4 py-3 rounded-xl bg-gray-900 border border-gray-700 text-sm focus:outline-none focus:border-blue-500 resize-none placeholder:text-gray-500"
              />
            </div>

            <div className="flex flex-wrap gap-1.5">
              {EXAMPLE_QUERIES.map((eq) => (
                <button
                  key={eq}
                  onClick={() => setQuery(eq)}
                  className="px-2.5 py-1 rounded-full text-[11px] bg-gray-800 border border-gray-700 text-gray-400 hover:text-white hover:border-gray-500 transition-colors"
                >
                  {eq}
                </button>
              ))}
            </div>
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
            disabled={!query.trim() || loading}
            className="w-full py-2.5 rounded-lg bg-purple-600 hover:bg-purple-500 disabled:opacity-40 disabled:cursor-not-allowed font-medium text-sm transition-colors flex items-center justify-center gap-2"
          >
            {loading ? <Loader2 size={16} className="animate-spin" /> : null}
            {loading ? 'Searching...' : 'Search by Text'}
          </button>

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
            <div className="p-3 rounded-lg border border-purple-800 bg-purple-950/30 space-y-2">
              <h3 className="text-sm font-semibold text-purple-300">Selected</h3>
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
                className="w-full py-1.5 rounded-md bg-purple-600 hover:bg-purple-500 disabled:opacity-40 text-xs font-medium transition-colors"
              >
                Find Similar
              </button>
            </div>
          )}
        </div>

        {/* Right panel: results */}
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
