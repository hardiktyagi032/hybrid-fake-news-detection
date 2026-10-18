import { useApp } from '../../context/AppContext';
import Sidebar from './Sidebar';
import Header from './Header';

export default function AppLayout({ children }) {
  const { theme } = useApp();
  return <div className={`app ${theme === 'dark' ? 'theme-dark' : ''}`}><Sidebar /><div className="app-main"><Header />{children}</div></div>;
}
