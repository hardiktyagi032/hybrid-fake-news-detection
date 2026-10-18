import { request } from './apiClient';
export const getHistory = () => request('/history');
export const deleteHistoryItem = (id) => request(`/history/${id}`, { method: 'DELETE' });
