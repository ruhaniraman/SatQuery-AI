import test from 'node:test';
import assert from 'node:assert/strict';
import { interpretSearch, searchPlaces, shortName } from './geocode.js';

test('coordinates are recognised in the usual forms', () => {
  assert.deepEqual(interpretSearch('23.75, 86.42'), { kind: 'coordinates', lat: 23.75, lng: 86.42 });
  assert.equal(interpretSearch('23°45′N 86°25′E').kind, 'coordinates');
});

test('names are places, including ones made only of coordinate letters', () => {
  assert.deepEqual(interpretSearch('Chandigarh'), { kind: 'place', query: 'Chandigarh' });
  assert.equal(interpretSearch('Lagos').kind, 'place');           // l-a-g-o-s all look like coordinate letters, but there is no digit
  assert.equal(interpretSearch('Route 66, Arizona').kind, 'place');
  assert.equal(interpretSearch('  ').kind, 'empty');
});

test('numbers that are not a valid coordinate keep the coordinate error', () => {
  const bad = interpretSearch('123.4, 200');
  assert.equal(bad.kind, 'bad-coordinates');
  assert.match(bad.error, /Latitude/);
});

test('searchPlaces parses Nominatim rows, drops unusable ones and shortens names', async () => {
  let url;
  const rows = [
    { lat: '30.73', lon: '76.77', display_name: 'Chandigarh, Chandigarh Sub-Division, Chandigarh, 160001, India', boundingbox: ['30.6', '30.8', '76.7', '76.9'] },
    { lat: 'x', lon: '1', display_name: 'broken' },
    { lat: '1', lon: '2', display_name: 'Tiny', boundingbox: ['1', '0', '2', '3'] },     // north < south: no bounds
  ];
  const fetchImpl = async (u) => { url = u; return { ok: true, status: 200, json: async () => rows }; };
  const found = await searchPlaces('Chandigarh', { fetchImpl, baseUrl: 'http://geo/search' });
  assert.match(url, /^http:\/\/geo\/search\?q=Chandigarh&format=jsonv2&limit=5/);
  assert.equal(found.length, 2);
  assert.deepEqual(found[0], { lat: 30.73, lng: 76.77, name: 'Chandigarh, Chandigarh Sub-Division, India', fullName: rows[0].display_name, bounds: [[30.6, 76.7], [30.8, 76.9]] });
  assert.equal(found[1].bounds, null);
});

test('searchPlaces reports network and HTTP failures in plain words', async () => {
  await assert.rejects(searchPlaces('x', { fetchImpl: async () => { throw new TypeError('Failed'); } }), /Could not reach/);
  await assert.rejects(searchPlaces('x', { fetchImpl: async () => ({ ok: false, status: 429 }) }), /429/);
  await assert.rejects(searchPlaces('x', { fetchImpl: async () => ({ ok: true, json: async () => { throw new Error('bad'); } }) }), /unreadable/);
  const abort = Object.assign(new Error('aborted'), { name: 'AbortError' });
  await assert.rejects(searchPlaces('x', { fetchImpl: async () => { throw abort; } }), (e) => e.name === 'AbortError');
});

test('shortName keeps the start and the country', () => {
  assert.equal(shortName('Paris, Île-de-France, France'), 'Paris, Île-de-France, France');
  assert.equal(shortName('A, B, C, D, E'), 'A, B, E');
});
