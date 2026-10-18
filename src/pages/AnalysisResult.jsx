import { useApp } from '../context/AppContext';
import Button from '../components/common/Button';
import ResultSummary from '../components/results/ResultSummary';
import KeyFindings from '../components/results/KeyFindings';
import ExplanationPanel from '../components/results/ExplanationPanel';
import EvidenceList from '../components/evidence/EvidenceList';
import ModelPredictions from '../components/ml-details/ModelPredictions';
import FollowUpInput from '../components/chat/FollowUpInput';

export default function AnalysisResult() { const { setPage, technical, setTechnical } = useApp(); return <main className="page result-page"><div className="breadcrumb"><button onClick={() => setPage('fact-check')}>← New fact check</button><div><button title="Copy analysis">⧉</button><button title="Export report">⇩</button><button title="Share">↗</button></div></div><div className="result-grid"><div><ResultSummary /><ExplanationPanel /><section className="evidence-section"><div className="section-heading"><div><span className="eyebrow">RETRIEVED EVIDENCE</span><h3>What sources say</h3></div><Button variant="quiet" onClick={() => setPage('sources')}>View all sources →</Button></div><EvidenceList limit={2} /></section><button className="technical-toggle" onClick={() => setTechnical(!technical)}><span>⌘</span> {technical ? 'Hide' : 'Show'} technical analysis <b>{technical ? '↑' : '↓'}</b></button>{technical && <ModelPredictions />}</div><aside className="result-aside"><KeyFindings /><FollowUpInput /></aside></div></main>; }
