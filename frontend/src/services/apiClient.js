// Replace this mock transport with the FastAPI base URL when the backend is available.
export async function request(path, options = {}) {
  const baseUrl = import.meta.env.VITE_API_BASE_URL || '/api';
  const response = await fetch(`${baseUrl}${path}`, { headers: { 'Content-Type': 'application/json', ...options.headers }, ...options });
  if (!response.ok) throw new Error(`Request failed: ${response.status}`);
  return response.json();
}
