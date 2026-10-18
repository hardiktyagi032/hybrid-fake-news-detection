import { useApp } from '../context/AppContext';
import Button from '../components/common/Button';
export default function NotFound() { const { setPage } = useApp(); return <main className="page empty-page"><p className="eyebrow">404</p><h2>This view is unavailable.</h2><Button onClick={() => setPage('fact-check')}>Return to fact check</Button></main>; }
