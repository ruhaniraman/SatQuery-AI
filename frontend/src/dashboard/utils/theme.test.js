import test from 'node:test';
import assert from 'node:assert/strict';
import { DEFAULT_THEME, THEME_KEY, nextTheme, readTheme, saveTheme } from './theme.js';

const memoryStorage = (initial = {}) => {
  const data = { ...initial };
  return { getItem: (k) => (k in data ? data[k] : null), setItem: (k, v) => { data[k] = String(v); }, data };
};

test('dark is the default and unknown or corrupt values fall back to it', () => {
  assert.equal(DEFAULT_THEME, 'dark');
  assert.equal(readTheme(memoryStorage()), 'dark');
  assert.equal(readTheme(memoryStorage({ [THEME_KEY]: 'sepia' })), 'dark');
  assert.equal(readTheme(memoryStorage({ [THEME_KEY]: '' })), 'dark');
  assert.equal(readTheme(null), 'dark');
});

test('a saved choice is read back', () => {
  const storage = memoryStorage();
  assert.equal(saveTheme('light', storage), true);
  assert.equal(storage.data[THEME_KEY], 'light');
  assert.equal(readTheme(storage), 'light');
  assert.equal(saveTheme('dark', storage) && readTheme(storage), 'dark');
});

test('only real themes are saved', () => {
  const storage = memoryStorage();
  assert.equal(saveTheme('neon', storage), false);
  assert.equal(saveTheme(undefined, storage), false);
  assert.deepEqual(storage.data, {});
});

test('blocked or throwing storage never breaks the page', () => {
  const throwing = { getItem() { throw new Error('denied'); }, setItem() { throw new Error('denied'); } };
  assert.equal(readTheme(throwing), 'dark');
  assert.equal(saveTheme('light', throwing), false);
});

test('toggling flips between the two themes', () => {
  assert.equal(nextTheme('dark'), 'light');
  assert.equal(nextTheme('light'), 'dark');
  assert.equal(nextTheme('anything else'), 'light');
});
