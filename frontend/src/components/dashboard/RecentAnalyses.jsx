import { history } from '../../mock/mockAnalysis';
import HistoryItem from '../history/HistoryItem';
export default function RecentAnalyses({ onOpen }) { return history.slice(0, 3).map(item => <HistoryItem key={item.title} item={item} onOpen={onOpen} />); }
