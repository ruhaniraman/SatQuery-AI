// Pure.

// What the header pill says, from the backend's health and whether an analysis is running.
export function describeStatus(health, executing) {
  switch (health.state) {
    case 'offline': return { tone: 'error', text: 'Backend offline' };
    case 'model-error': return { tone: 'error', text: 'Model unavailable' };
    case 'loading': return { tone: 'warn', text: 'Model loading' };
    case 'ready': return executing ? { tone: 'busy', text: 'Analysing' } : { tone: 'ready', text: 'Agent ready' };
    default: return executing ? { tone: 'busy', text: 'Analysing' } : { tone: 'idle', text: 'Checking' };
  }
}
