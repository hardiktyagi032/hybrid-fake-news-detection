import { createContext, useContext, useMemo, useState } from 'react';

const AppContext = createContext(null);

export function AppProvider({ children }) {
  const [page, setPage] = useState('fact-check');
  const [theme, setTheme] = useState('light');
  const [analysis, setAnalysis] = useState('ready');
  const [sourceFilter, setSourceFilter] = useState('all');
  const [technical, setTechnical] = useState(false);
  const value = useMemo(() => ({ page, setPage, theme, setTheme, analysis, setAnalysis, sourceFilter, setSourceFilter, technical, setTechnical }), [page, theme, analysis, sourceFilter, technical]);
  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export const useApp = () => useContext(AppContext);
