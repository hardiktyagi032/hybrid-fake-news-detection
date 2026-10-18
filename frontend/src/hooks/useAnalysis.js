import { useState } from 'react';
import { startAnalysis } from '../services/analysisService';
export function useAnalysis() { const [loading, setLoading] = useState(false); const [error, setError] = useState(null); const submit = async payload => { setLoading(true); setError(null); try { return await startAnalysis(payload); } catch (err) { setError(err); throw err; } finally { setLoading(false); } }; return { submit, loading, error }; }
