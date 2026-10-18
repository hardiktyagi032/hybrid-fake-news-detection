export const formatDate = (value) => new Intl.DateTimeFormat('en', { dateStyle: 'medium' }).format(new Date(value));
