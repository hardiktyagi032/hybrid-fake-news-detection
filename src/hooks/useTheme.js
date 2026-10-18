import { useApp } from '../context/AppContext';
export const useTheme = () => { const { theme, setTheme } = useApp(); return { theme, setTheme }; };
