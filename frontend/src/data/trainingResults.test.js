import assert from 'node:assert/strict';
import test from 'node:test';
import { trainingAudit } from './trainingResults.js';

test('training audit declares exactly the three approved models', () => {
  assert.deepEqual(
    trainingAudit.models.map(model => model.id),
    ['logistic_regression', 'linear_svm', 'random_forest'],
  );
});

test('selected model is present and wins source-held validation', () => {
  const selected = trainingAudit.models.find(model => model.id === trainingAudit.selectedModel);
  assert.ok(selected);
  assert.equal(selected.selected, true);
  assert.equal(
    selected.validationMeanF1,
    Math.max(...trainingAudit.models.map(model => model.validationMeanF1)),
  );
});

test('all reported rates and thresholds are valid probabilities', () => {
  for (const model of trainingAudit.models) {
    for (const key of [
      'threshold', 'validationMeanF1', 'validationWorstF1', 'testMeanAccuracy',
      'testMeanBalancedAccuracy', 'testMeanF1', 'testWorstF1', 'meanRocAuc',
      'meanBrier', 'contemporaryF1', 'contemporaryAccuracy',
    ]) assert.ok(model[key] >= 0 && model[key] <= 1, `${model.id}.${key}`);
  }
});

test('confusion matrices match the declared contemporary sample size', () => {
  for (const model of trainingAudit.models) {
    const records = model.confusion.flat().reduce((total, value) => total + value, 0);
    assert.equal(records, trainingAudit.records.contemporary);
  }
});
