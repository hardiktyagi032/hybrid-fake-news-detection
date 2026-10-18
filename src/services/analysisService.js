import { request } from './apiClient';
export const startAnalysis = (payload) => request('/analyses', { method: 'POST', body: JSON.stringify(payload) });
export const getAnalysis = (id) => request(`/analyses/${id}`);
