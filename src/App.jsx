import { AppProvider, useApp } from './context/AppContext';
import AppLayout from './components/layout/AppLayout';
import NewAnalysis from './pages/NewAnalysis';
import AnalysisResult from './pages/AnalysisResult';
import History from './pages/History';
import Dashboard from './pages/Dashboard';
import Sources from './pages/Sources';
import Settings from './pages/Settings';
import Profile from './pages/Profile';
import NotFound from './pages/NotFound';

const screens = { 'fact-check': NewAnalysis, result: AnalysisResult, history: History, dashboard: Dashboard, sources: Sources, settings: Settings, profile: Profile };

function Router() {
  const { page } = useApp();
  const Screen = screens[page] || NotFound;
  return <AppLayout><Screen /></AppLayout>;
}

export default function App() {
  return <AppProvider><Router /></AppProvider>;
}
