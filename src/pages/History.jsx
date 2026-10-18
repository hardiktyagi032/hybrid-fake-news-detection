import { useState } from 'react';
import { useApp } from '../context/AppContext';
import { history } from '../mock/mockAnalysis';
import HistorySearch from '../components/history/HistorySearch';
import HistoryItem from '../components/history/HistoryItem';
export default function History() { const { setPage } = useApp(); const [query, setQuery] = useState(''); const items = history.filter(item => item.title.toLowerCase().includes(query.toLowerCase())); return <main className="page history-page"><div className="history-toolbar"><HistorySearch value={query} onChange={setQuery} /><button className="filter-button">Filter ▾</button></div><section className="history-list panel"><div className="history-group-title">Today <span>1 analysis</span></div>{items.filter((_, index) => index === 0).map(item => <HistoryItem item={item} key={item.title} onOpen={() => setPage('result')} />)}<div className="history-group-title">Earlier <span>{Math.max(items.length - 1, 0)} analyses</span></div>{items.slice(1).map(item => <HistoryItem item={item} key={item.title} onOpen={() => setPage('result')} />)}</section></main>; }
