import { sources } from '../../mock/mockAnalysis';
import EvidenceCard from './EvidenceCard';
export default function EvidenceList({ limit, filter = 'all' }) {
  const items = filter === 'all' ? sources : sources.filter(item => item.verdict.toLowerCase() === filter);
  return <div className="evidence-list">{items.slice(0, limit || items.length).map(item => <EvidenceCard item={item} key={item.id} />)}</div>;
}
