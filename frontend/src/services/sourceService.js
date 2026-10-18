import { request } from './apiClient';
export const getSources = (params = '') => request(`/sources${params}`);
