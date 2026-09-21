import test from 'node:test';
import assert from 'node:assert/strict';
import { parseCoordinates } from './coordinates.js';

const near = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-6, `${actual} ~ ${expected}`);
const ok = (text, lat, lng) => {
  const r = parseCoordinates(text);
  assert.equal(r.ok, true, `${text} -> ${JSON.stringify(r)}`);
  near(r.lat, lat);
  near(r.lng, lng);
};
const bad = (text, pattern) => {
  const r = parseCoordinates(text);
  assert.equal(r.ok, false, `${text} should be rejected`);
  assert.match(r.error, pattern, text);
};

test('plain decimal pairs, latitude first', () => {
  ok('23.75, 86.42', 23.75, 86.42);
  ok('23.75 86.42', 23.75, 86.42);
  ok('  23.75,86.42  ', 23.75, 86.42);
  ok('-23.55, -46.63', -23.55, -46.63);
  ok('+23.5, +86.4', 23.5, 86.4);
  ok('0, 0', 0, 0);
  ok('lat 23.75 lon 86.42', 23.75, 86.42);
  ok('Latitude: 23.75, Longitude: 86.42', 23.75, 86.42);
  ok('−23.55, −46.63', -23.55, -46.63);                                  // typographic minus signs
});

test('hemisphere letters set the sign and the order', () => {
  ok('23.75N 86.42E', 23.75, 86.42);
  ok('23.75° N, 86.42° E', 23.75, 86.42);
  ok('23.75S 46.6W', -23.75, -46.6);
  ok('N 23.75 E 86.42', 23.75, 86.42);
  ok('S23.75, W46.6', -23.75, -46.6);
  ok('86.42E 23.75N', 23.75, 86.42);                                      // longitude first is fine with letters
  ok('e 86.42 n 23.75', 23.75, 86.42);
});

test('degrees, minutes and seconds', () => {
  ok('23°45\'N 86°25\'E', 23 + 45 / 60, 86 + 25 / 60);
  ok('23°47′30″N 86°25′12″E', 23 + 47 / 60 + 30 / 3600, 86 + 25 / 60 + 12 / 3600);
  ok('23 47 30 N 86 25 12 E', 23 + 47 / 60 + 30 / 3600, 86 + 25 / 60 + 12 / 3600);
  ok('33°51′54″S 151°12′36″E', -(33 + 51 / 60 + 54 / 3600), 151 + 12 / 60 + 36 / 3600);
  ok('40 26 46 N, 79 58 56 W', 40 + 26 / 60 + 46 / 3600, -(79 + 58 / 60 + 56 / 3600));
});

test('a negative degree with minutes stays negative overall', () => {
  ok('-23 30, -46 30', -23.5, -46.5);
});

test('out-of-range values are rejected with a reason', () => {
  bad('91, 10', /Latitude must be between/);
  bad('-91, 10', /Latitude must be between/);
  bad('10, 181', /Longitude must be between/);
  bad('10, -181', /Longitude must be between/);
  bad('23 75 10 N 86 25 12 E', /Minutes/);
  bad('23 45 75 N 86 25 12 E', /Seconds/);
});

test('boundary values are allowed', () => {
  ok('90, 180', 90, 180);
  ok('-90, -180', -90, -180);
});

test('nonsense gets a helpful message, never a throw', () => {
  bad('', /Enter a latitude and longitude/);
  bad('   ', /Enter a latitude and longitude/);
  bad('hello world', /Only numbers/);
  bad('23.75', /Separate latitude and longitude|exactly one/);
  bad('23.75 86.42 10.0', /Separate latitude and longitude/);
  bad('23N 45N', /one of N\/S/);
  bad('23E 45E', /one of N\/S/);
  bad('23.75N 86.42', /letter/);
  bad('N', /No numbers/);
  bad('1,2,3', /Separate latitude and longitude/);
  bad('N 1 2 3 4 E 5', /Expected exactly one/);
  assert.equal(parseCoordinates(undefined).ok, false);
  assert.equal(parseCoordinates(null).ok, false);
  assert.equal(parseCoordinates(42).ok, false);
});
