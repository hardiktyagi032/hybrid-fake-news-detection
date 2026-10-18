import { useApp } from '../../context/AppContext';

const nav = [
  ['fact-check', '✦', 'Fact check'], ['dashboard', '▦', 'Dashboard'], ['history', '◷', 'History'], ['sources', '⌕', 'Sources'],
];

export default function Sidebar() {
  const { page, setPage } = useApp();
  return <aside className="sidebar"><button className="brand" onClick={() => setPage('fact-check')}><span className="brand-mark">V</span><span>veritas</span></button><div className="sidebar-new"><button onClick={() => setPage('fact-check')}><span>＋</span> New fact check</button></div><nav>{nav.map(([id, icon, label]) => <button key={id} className={page === id ? 'nav-link active' : 'nav-link'} onClick={() => setPage(id)}><span>{icon}</span>{label}</button>)}</nav><div className="sidebar-bottom"><button className={page === 'settings' ? 'nav-link active' : 'nav-link'} onClick={() => setPage('settings')}><span>⚙</span>Settings</button><button className={page === 'profile' ? 'nav-link active' : 'nav-link'} onClick={() => setPage('profile')}><span>◉</span>Profile</button><div className="sidebar-note"><span className="live-dot" /> Evidence-first analysis<br /><small>Hybrid ML + verification</small></div></div></aside>;
}
