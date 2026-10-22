import { Search, CheckSquare, Square } from 'lucide-react';
import type { SearchResultItem } from '../types';

interface Props {
  results: SearchResultItem[];
  onSelect?: (item: SearchResultItem) => void;
  onFindSimilar?: (item: SearchResultItem) => void;
  compareItems?: SearchResultItem[];
  onToggleCompare?: (item: SearchResultItem) => void;
}

export function ResultsGrid({ results, onSelect, onFindSimilar, compareItems, onToggleCompare }: Props) {
  if (results.length === 0) {
    return <p className="text-gray-500 text-center py-12">No results yet. Upload an image to search.</p>;
  }

  const compareIds = new Set(compareItems?.map((c) => c.image_id) ?? []);

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3">
      {results.map((item) => {
        const inCompare = compareIds.has(item.image_id);
        return (
          <div
            key={`${item.image_id}-${item.position}`}
            className={`group relative rounded-lg overflow-hidden border bg-gray-900 transition-all hover:shadow-lg ${
              inCompare
                ? 'border-emerald-500 ring-1 ring-emerald-500/40 shadow-emerald-500/10'
                : 'border-gray-800 hover:border-blue-500 hover:shadow-blue-500/10'
            }`}
          >
            <button onClick={() => onSelect?.(item)} className="w-full">
              <img
                src={`/api/static/${item.uri}`}
                alt={item.image_id}
                className="w-full aspect-square object-cover"
                loading="lazy"
                onError={(e) => {
                  (e.target as HTMLImageElement).src = `data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='200' height='200'><rect fill='%23374151' width='200' height='200'/><text x='50%25' y='50%25' fill='%239CA3AF' text-anchor='middle' dy='.3em' font-size='12'>No preview</text></svg>`;
                }}
              />
            </button>

            <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/80 to-transparent p-2">
              <div className="flex items-center justify-between text-xs">
                <span className="font-mono text-blue-300">#{item.position}</span>
                <span className="text-gray-300">{(item.score * 100).toFixed(1)}%</span>
              </div>
              {item.use_case && (
                <span className="inline-block mt-1 px-1.5 py-0.5 text-[10px] rounded bg-blue-900/60 text-blue-300">
                  {item.use_case}
                </span>
              )}
            </div>

            {/* Compare checkbox */}
            {onToggleCompare && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onToggleCompare(item);
                }}
                className={`absolute top-2 left-2 p-1 rounded transition-all ${
                  inCompare
                    ? 'bg-emerald-600 text-white opacity-100'
                    : 'bg-gray-900/80 text-gray-400 opacity-0 group-hover:opacity-100 hover:text-emerald-400'
                }`}
                title={inCompare ? 'Remove from compare' : 'Add to compare'}
              >
                {inCompare ? <CheckSquare size={14} /> : <Square size={14} />}
              </button>
            )}

            {onFindSimilar && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onFindSimilar(item);
                }}
                className="absolute top-2 right-2 p-1.5 rounded-full bg-gray-900/80 text-gray-400 opacity-0 group-hover:opacity-100 hover:bg-blue-600 hover:text-white transition-all"
                title="Find similar"
              >
                <Search size={14} />
              </button>
            )}
          </div>
        );
      })}
    </div>
  );
}
