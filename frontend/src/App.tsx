import { useState } from 'react';
import { Navbar } from './components/Navbar';
import { SearchPage } from './pages/SearchPage';
import { TextSearchPage } from './pages/TextSearchPage';
import { TagPage } from './pages/TagPage';
import { QualityPage } from './pages/QualityPage';
import { EvaluatePage } from './pages/EvaluatePage';

type Page = 'search' | 'text-search' | 'tag' | 'quality' | 'evaluate';

export default function App() {
  const [page, setPage] = useState<Page>('search');

  return (
    <div className="min-h-screen bg-gray-950">
      <Navbar current={page} onChange={setPage} />
      {page === 'search' && <SearchPage />}
      {page === 'text-search' && <TextSearchPage />}
      {page === 'tag' && <TagPage />}
      {page === 'quality' && <QualityPage />}
      {page === 'evaluate' && <EvaluatePage />}
    </div>
  );
}
