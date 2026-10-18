import { stages } from '../../mock/mockAnalysis';

export default function PipelineStages({ compact = false }) {
  return <div className={`pipeline ${compact ? 'pipeline--compact' : ''}`}>{stages.map(([title, description, state], index) => <div className="pipeline-stage" key={title}><div className="stage-rail"><span className={`stage-icon ${state}`}>{state === 'done' ? '✓' : index + 1}</span>{index < stages.length - 1 && <i />}</div><div><strong>{title}</strong><p>{description}</p></div></div>)}</div>;
}
