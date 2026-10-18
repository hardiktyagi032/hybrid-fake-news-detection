import { createContext, useContext, useMemo, useState } from 'react';

const AppContext = createContext(null);
const DEFAULT_PAGE = 'fact-check';
const VALID_PAGES = new Set(['fact-check', 'result', 'history', 'dashboard', 'model-audit', 'sources', 'settings', 'profile']);

function initialPage() {
  const requested = new URLSearchParams(window.location.search).get('view');
  return VALID_PAGES.has(requested) ? requested : DEFAULT_PAGE;
}

export function AppProvider({ children }) {
  const [page, setPageState] = useState(initialPage);
  const [theme, setTheme] = useState('light');
  const [analysis, setAnalysis] = useState('ready');
  const [sourceFilter, setSourceFilter] = useState('all');
  const [technical, setTechnical] = useState(false);
  const setPage = (nextPage) => {
    setPageState(nextPage);
    const url = new URL(window.location.href);
    if (nextPage === DEFAULT_PAGE) url.searchParams.delete('view');
    else url.searchParams.set('view', nextPage);
    window.history.replaceState({}, '', url);
  };
  const value = useMemo(() => ({ page, setPage, theme, setTheme, analysis, setAnalysis, sourceFilter, setSourceFilter, technical, setTechnical }), [page, theme, analysis, sourceFilter, technical]);
  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export const useApp = () => useContext(AppContext);
