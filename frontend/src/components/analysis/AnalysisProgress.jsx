import PipelineStages from './PipelineStages';

export default function AnalysisProgress({ active }) {
  return <section className="analysis-progress panel"><div className="progress-heading"><div><span className="eyebrow">HYBRID PIPELINE</span><h3>{active ? 'Investigating your claim' : 'How Veritas reaches a decision'}</h3></div><span className={active ? 'progress-pill active' : 'progress-pill'}>{active ? 'In progress' : 'Complete'}</span></div>{active && <div className="progress-track"><span /></div>}<PipelineStages compact /></section>;
}
