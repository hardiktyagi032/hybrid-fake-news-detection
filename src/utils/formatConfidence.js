export const formatConfidence = (value) => `${Math.round(Number(value) * (value <= 1 ? 100 : 1))}%`;
