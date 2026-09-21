// Reads the backend's comparison answer into parts so the panel can lay it out (warning banners, one
// card per changed area with Before / After / Measured rows) instead of printing a wall of text.
// Pure: tested in plain Node. The format it reads is produced by change_report.compose_change_answer:
//
//   WARNING: ...
//
//   Changed areas
//   - Likely change, centre (~6% of the scene). Also flagged by the deforestation scan.
//       Before: ...
//       After: ...
//       Measured: ...
//   - No changed area was found.
//
//   Your question (...) is not answered separately in comparison mode. ...
//
// Anything that does not look like that (an older, model-written answer) comes back as { legacy: true }
// so the caller can show it as plain paragraphs.

const KINDS = { 'Major change': 'major', 'Likely change': 'likely', 'Possible smaller change': 'possible' };
const HEAD = /^- (Major change|Likely change|Possible smaller change), (.+?) \(~(\d+)% of the scene\)\.(?: Also flagged by (.+)\.)?\s*$/;
const FIELD = /^\s+(Before|After|Measured):\s*(.*)$/;

export function parseChangeAnswer(text) {
  const lines = String(text ?? '').replace(/\r/g, '').split('\n');
  const heading = lines.findIndex((l) => l.trim() === 'Changed areas');
  if (heading === -1) return { legacy: true, warnings: [], areas: [], notes: [], footer: [] };

  const warnings = lines.slice(0, heading).map((l) => l.trim()).filter((l) => l.startsWith('WARNING: ')).map((l) => l.slice(9));
  const areas = [];
  const notes = [];
  const footer = [];
  let current = null;

  for (const line of lines.slice(heading + 1)) {
    if (!line.trim()) { current = null; continue; }
    const head = line.match(HEAD);
    const field = line.match(FIELD);
    if (head) {
      current = {
        kind: KINDS[head[1]], label: head[1], where: head[2], sizePercent: Number(head[3]),
        scans: head[4] ? head[4].replace(/^the /, '').replace(/ scans?$/, '').split(/, | and /).map((s) => s.replace(/^the /, '')) : [],
        before: '', after: '', measured: '',
      };
      areas.push(current);
    } else if (field && current) {
      current[field[1].toLowerCase()] = field[2].trim();
    } else if (line.startsWith('- ')) {
      notes.push(line.slice(2).trim());                        // e.g. "No changed area was found."
      current = null;
    } else {
      footer.push(line.trim());
      current = null;
    }
  }
  return { legacy: false, warnings, areas, notes, footer };
}

export const KIND_TONE = { major: 'red', likely: 'amber', possible: 'slate' };
