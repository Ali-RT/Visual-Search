import { Search, Tag, Eye, Type, ShieldAlert, FlaskConical } from 'lucide-react';

type Page = 'search' | 'text-search' | 'tag' | 'quality' | 'evaluate';

interface Props {
  current: Page;
  onChange: (page: Page) => void;
}

const tabs: { id: Page; label: string; icon: React.ReactNode }[] = [
  { id: 'search', label: 'Visual Search', icon: <Search size={16} /> },
  { id: 'text-search', label: 'Text Search', icon: <Type size={16} /> },
  { id: 'tag', label: 'Image Tagging', icon: <Tag size={16} /> },
  { id: 'quality', label: 'Quality', icon: <ShieldAlert size={16} /> },
  { id: 'evaluate', label: 'Evaluate', icon: <FlaskConical size={16} /> },
];

export function Navbar({ current, onChange }: Props) {
  return (
    <nav className="border-b border-gray-800 bg-gray-950/80 backdrop-blur-sm sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 flex items-center h-14 gap-6">
        <div className="flex items-center gap-2 mr-6">
          <Eye size={22} className="text-blue-400" />
          <span className="font-bold text-lg tracking-tight">Visual Search</span>
        </div>
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => onChange(tab.id)}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
              current === tab.id
                ? 'bg-blue-600 text-white'
                : 'text-gray-400 hover:text-white hover:bg-gray-800'
            }`}
          >
            {tab.icon}
            {tab.label}
          </button>
        ))}
      </div>
    </nav>
  );
}
