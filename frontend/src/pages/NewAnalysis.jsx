import { useState } from 'react';
import { useApp } from '../context/AppContext';
import ChatWindow from '../components/chat/ChatWindow';
import ChatInput from '../components/chat/ChatInput';
import AnalysisProgress from '../components/analysis/AnalysisProgress';

const starter = 'A social media post claims that routine vaccines can alter a person’s DNA and cause long-term genetic changes.';

export default function NewAnalysis() {
  const { setPage, analysis, setAnalysis } = useApp();
  const [value, setValue] = useState(starter);
  const submit = () => { setAnalysis('running'); setTimeout(() => { setAnalysis('complete'); setPage('result'); }, 1300); };
  return <main className="page new-analysis"><div className="new-analysis__intro"><div className="intro-orb">✦</div><span className="eyebrow">EVIDENCE-FIRST VERIFICATION</span><h2>Is this claim <em>actually</em> true?</h2><p>Veritas combines language models, live evidence retrieval, and claim-level verification to make its reasoning inspectable.</p></div><div className="analysis-workspace"><div className="chat-column"><ChatWindow /><ChatInput value={value} onChange={setValue} onSubmit={submit} analysing={analysis === 'running'} /></div><AnalysisProgress active={analysis === 'running'} /></div></main>;
}
