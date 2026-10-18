import { useApp } from '../../context/AppContext';

const titles = { 'fact-check': ['New fact check', 'Verify news with transparent, source-backed analysis'], result: ['Analysis result', 'Evidence-backed decision'], history: ['Analysis history', 'Your saved investigations'], dashboard: ['Dashboard', 'Your verification activity'], sources: ['Source library', 'Evidence used across your analyses'], settings: ['Settings', 'Control your workspace'], profile: ['Profile', 'Your account and preferences'] };

export default function Header() {
  const { page, theme, setTheme } = useApp();
  const [title, subtitle] = titles[page] || ['Not found', ''];
  return <header className="header"><div><p className="eyebrow">VERITAS / FACT CHECKING</p><h1>{title}</h1><p className="header-subtitle">{subtitle}</p></div><div className="header-actions"><button className="icon-button" onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')} aria-label="Toggle colour mode">{theme === 'dark' ? '☀' : '☾'}</button><button className="avatar" aria-label="Open profile">AC</button></div></header>;
}
