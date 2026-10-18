export const hasAnalysisInput = (value) => Boolean(value && value.trim().length > 4);
export const isUrl = (value) => /^https?:\/\//i.test(value.trim());
