import { request } from './apiClient';
export const saveSettings = (settings) => request('/settings', { method: 'PUT', body: JSON.stringify(settings) });
