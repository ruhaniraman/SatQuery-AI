import { FileText, Layers, MessageSquare, ScanSearch, GitCompare } from 'lucide-react';
import ImageryPanel from './components/panels/ImageryPanel';
import AssistantPanel from './components/panels/AssistantPanel';
import ScansPanel from './components/panels/ScansPanel';
import ComparePanel from './components/panels/ComparePanel';
import ReportPanel from './components/panels/ReportPanel';

// The left navigation is built from this list. To add a feature: write a panel component that takes
// `{ ws, goTo }` (ws = the shared workspace from useWorkspace, goTo(id) switches feature) and add an
// entry here. Nothing else needs to change.
//
//   id       unique key
//   label    short name under the nav icon
//   title    heading of the panel
//   summary  one line under the heading
//   icon     lucide icon
//   Panel    the component
//   status   optional (ws) => 'busy' | 'ready' | null, shown as a badge on the nav item
export const FEATURES = [
  {
    id: 'imagery',
    label: 'Imagery',
    title: 'Imagery',
    summary: 'Upload the images to analyse and tag their sensor type.',
    icon: Layers,
    Panel: ImageryPanel,
    status: (ws) => (ws.images.length > 0 ? 'ready' : null),
  },
  {
    id: 'assistant',
    label: 'Assistant',
    title: 'Assistant',
    summary: 'Ask questions about your imagery in plain language.',
    icon: MessageSquare,
    Panel: AssistantPanel,
    status: (ws) => (ws.isExecuting && ws.busy === 'chat' ? 'busy' : null),
  },
  {
    id: 'scans',
    label: 'Scans',
    title: 'Feature scans',
    summary: 'Find mining, agriculture or deforestation in Image A.',
    icon: ScanSearch,
    Panel: ScansPanel,
    status: (ws) => (ws.isExecuting && String(ws.busy).startsWith('scan:') ? 'busy' : null),
  },
  {
    id: 'compare',
    label: 'Compare',
    title: 'Compare',
    summary: 'Detect change between two images, or fuse optical with SAR.',
    icon: GitCompare,
    Panel: ComparePanel,
    status: (ws) => (ws.isExecuting && ws.busy === 'compare' ? 'busy' : null),
  },
  {
    id: 'report',
    label: 'Report',
    title: 'Report',
    summary: 'Exactly what ran, plus the audit PDF.',
    icon: FileText,
    Panel: ReportPanel,
    status: (ws) => (ws.result ? 'ready' : null),
  },
];

export const featureById = (id) => FEATURES.find((f) => f.id === id) || FEATURES[0];
