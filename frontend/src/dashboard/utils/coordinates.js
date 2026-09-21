// Turns what a person types or pastes into a latitude/longitude. Pure (no DOM): tested in plain Node.
//
// Accepted (latitude first unless N/S/E/W letters say otherwise):
//   23.75, 86.42        23.75 86.42        -23.75,-46.6       lat 23.75 lon 86.42
//   23.75N 86.42E       N 23.75  E 86.42   86.42E 23.75N      (letters decide which is which)
//   23°45'N 86°25'E     23 45 30 N 86 25 12 E                  (degrees, minutes, seconds)

export const COORDINATE_HELP = 'Try “23.75, 86.42” or “23°45′N 86°25′E”.';

const fail = (error) => ({ ok: false, error });

// One coordinate from its numbers: [degrees], [degrees, minutes] or [degrees, minutes, seconds]
function toDecimal(numbers, hemisphere) {
  const [deg, min = 0, sec = 0] = numbers;
  if (numbers.length > 1 && (min < 0 || min >= 60)) return { error: 'Minutes must be between 0 and 60.' };
  if (numbers.length > 2 && (sec < 0 || sec >= 60)) return { error: 'Seconds must be between 0 and 60.' };
  const magnitude = Math.abs(deg) + min / 60 + sec / 3600;
  const negative = deg < 0 || Object.is(deg, -0) || hemisphere === 'S' || hemisphere === 'W';
  return { value: negative ? -magnitude : magnitude };
}

// Split a token list into coordinate groups: [{ numbers, letter }]
function group(tokens) {
  const isLetter = (t) => /^[NSEW]$/.test(t);
  const groups = [];
  if (isLetter(tokens[0])) {                                   // prefix style: N 23.75 E 86.42
    tokens.forEach((t) => {
      if (isLetter(t)) groups.push({ letter: t, numbers: [] });
      else groups[groups.length - 1].numbers.push(Number(t));
    });
  } else {                                                     // suffix style: 23.75N 86.42E
    let current = { letter: null, numbers: [] };
    tokens.forEach((t) => {
      if (isLetter(t)) {
        current.letter = t;
        groups.push(current);
        current = { letter: null, numbers: [] };
      } else current.numbers.push(Number(t));
    });
    if (current.numbers.length) groups.push(current);          // trailing numbers without a letter
  }
  return groups;
}

export function parseCoordinates(input) {
  const text = String(input ?? '').trim();
  if (!text) return fail('Enter a latitude and longitude. ' + COORDINATE_HELP);

  const cleaned = text
    .toUpperCase()
    .replace(/LATITUDE|LAT\b|LONGITUDE|LONG\b|LON\b|LNG\b/g, ' ')
    .replace(/[°º′’'″”":]/g, ' ')
    .replace(/[−–—]/g, '-');
  if (/[^0-9NSEW\s.,;+-]/.test(cleaned)) return fail('Only numbers and the letters N, S, E, W are allowed. ' + COORDINATE_HELP);

  const hasLetters = /[NSEW]/.test(cleaned);
  const tokens = cleaned.replace(/[,;]/g, ' ').match(/[NSEW]|[+-]?\d+(?:\.\d+)?|[+-]?\.\d+/g) || [];
  if (!tokens.some((t) => /\d/.test(t))) return fail('No numbers found. ' + COORDINATE_HELP);

  let groups;
  if (hasLetters) {
    groups = group(tokens);
  } else if (/[,;]/.test(cleaned) && cleaned.split(/[,;]/).length === 2) {
    groups = cleaned.split(/[,;]/).map((part) => ({ letter: null, numbers: (part.match(/[+-]?\d+(?:\.\d+)?|[+-]?\.\d+/g) || []).map(Number) }));
  } else if (tokens.length === 2) {
    groups = tokens.map((t) => ({ letter: null, numbers: [Number(t)] }));
  } else {
    return fail('Separate latitude and longitude with a comma, or add N/S and E/W letters. ' + COORDINATE_HELP);
  }

  if (groups.length !== 2 || groups.some((g) => g.numbers.length < 1 || g.numbers.length > 3 || g.numbers.some((n) => !Number.isFinite(n)))) {
    return fail('Expected exactly one latitude and one longitude. ' + COORDINATE_HELP);
  }

  let latGroup = groups[0];
  let lngGroup = groups[1];
  const letters = groups.map((g) => g.letter);
  if (letters.every(Boolean)) {
    const vertical = (l) => l === 'N' || l === 'S';
    if (vertical(letters[0]) === vertical(letters[1])) return fail('Give one of N/S (latitude) and one of E/W (longitude).');
    if (!vertical(letters[0])) [latGroup, lngGroup] = [groups[1], groups[0]];
  } else if (letters.some(Boolean)) {
    return fail('Put a letter (N/S/E/W) on both coordinates, or on neither.');
  }
  if (latGroup.letter && !/^[NS]$/.test(latGroup.letter)) return fail('Latitude takes N or S.');
  if (lngGroup.letter && !/^[EW]$/.test(lngGroup.letter)) return fail('Longitude takes E or W.');

  const lat = toDecimal(latGroup.numbers, latGroup.letter);
  const lng = toDecimal(lngGroup.numbers, lngGroup.letter);
  if (lat.error || lng.error) return fail(lat.error || lng.error);
  if (Math.abs(lat.value) > 90) return fail('Latitude must be between −90 and 90.');
  if (Math.abs(lng.value) > 180) return fail('Longitude must be between −180 and 180.');
  return { ok: true, lat: lat.value, lng: lng.value };
}
