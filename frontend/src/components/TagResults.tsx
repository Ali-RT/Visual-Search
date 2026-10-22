import type { TagResponse } from '../types';

interface Props {
  data: TagResponse;
}

const GROUP_COLORS: Record<string, string> = {
  time_of_day: 'bg-amber-500',
  scene_type: 'bg-emerald-500',
  weather: 'bg-sky-500',
  road_condition: 'bg-rose-500',
};

export function TagResults({ data }: Props) {
  return (
    <div className="space-y-5">
      {Object.entries(data.tags).map(([group, info]) => (
        <div key={group} className="rounded-lg border border-gray-800 bg-gray-900 p-4">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wide">
              {group.replace(/_/g, ' ')}
            </h3>
            <span className="text-sm font-mono px-2 py-0.5 rounded bg-gray-800 text-white">
              {info.label}
            </span>
          </div>
          <div className="space-y-2">
            {Object.entries(info.scores)
              .sort(([, a], [, b]) => b - a)
              .map(([label, score]) => (
                <div key={label} className="flex items-center gap-3">
                  <span className="w-20 text-xs text-gray-400 text-right">{label}</span>
                  <div className="flex-1 h-5 bg-gray-800 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all ${
                        label === info.label
                          ? GROUP_COLORS[group] || 'bg-blue-500'
                          : 'bg-gray-600'
                      }`}
                      style={{ width: `${Math.max(score * 100, 1)}%` }}
                    />
                  </div>
                  <span className="w-12 text-xs text-gray-400 font-mono">
                    {(score * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
          </div>
        </div>
      ))}
    </div>
  );
}
