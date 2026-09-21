import { FileText, GitCompareArrows, Layers, Merge, MessageSquare, ScanSearch } from 'lucide-react';
import ImageryPanel from './components/panels/ImageryPanel';
import AssistantPanel from './components/panels/AssistantPanel';
import ScansPanel from './components/panels/ScansPanel';
import ChangePanel from './components/panels/ChangePanel';
import FusionPanel from './components/panels/FusionPanel';
import ReportPanel from './components/panels/ReportPanel';

// The left navigation is built from this list. To add a feature: write a panel component that takes
// `{ ws, goTo }` (ws = the shared workspace from useWorkspace, goTo(id) switches feature) and add an
// entry here. Nothing else needs to change.
//
//   id           unique key
//   group        'single' | 'pair' | 'output': the nav shows a divider (and caption) between groups
//   label        short name under the nav icon
//   title        heading of the panel
//   summary      one line under the heading
//   icon         lucide icon
//   Panel        the component
//   status       optional (ws) => 'busy' | 'ready' | null, shown as a badge on the nav item
//   layers       (ws, evidenceRun) => [{ value, label, disabled, empty }]: what the Inputs viewer can show for this
//                feature. A slot layer with no image yet is `empty` (still clickable: it shows an upload prompt);
//                `disabled` means there is nothing to show or do (no evidence yet, swipe without both images)
//   uploadTargets (ws) => [{ id, label }]: the image slots this feature owns, so the Inputs viewer can upload into them
//   evidenceRun  (ws) => the run whose evidence image the viewer's "Evidence" layer shows
const SENSOR = { optical: 'Optical', sar: 'SAR' };

const slotLayer = (ws, id, label) => ({ value: id, label, empty: !ws.slots[id].file });
const evidenceLayer = (run, label = 'Evidence') => ({ value: 'evidence', label, disabled: !run?.data?.visual_evidence_url });
// A scan's evidence is the image with the findings drawn on it: offer it as Highlights, and a swipe against the original
const singleLayers = (ws, run) => [
  slotLayer(ws, 'a', `Image · ${SENSOR[ws.slots.a.modality]}`),
  evidenceLayer(run, run?.kind === 'scan' ? 'Highlights' : 'Evidence'),
  { value: 'compare', label: 'Compare', disabled: !(run?.kind === 'scan' && run.data?.visual_evidence_url && ws.slots.a.previewState === 'ready') },
];
const scanEvidence = (ws) => ws.latest.scan;
const singleTargets = () => [{ id: 'a', label: 'Image' }];

export const GROUP_LABELS = { single: 'Single', pair: 'Pairs', output: '' };

export const FEATURES = [
  {
    id: 'imagery',
    group: 'single',
    label: 'Imagery',
    title: 'Imagery',
    summary: 'Upload the image to analyse and tag its sensor type.',
    icon: Layers,
    Panel: ImageryPanel,
    status: (ws) => (ws.slots.a.file ? 'ready' : null),
    layers: singleLayers,
    evidenceRun: scanEvidence,
    uploadTargets: singleTargets,
  },
  {
    id: 'assistant',
    group: 'single',
    label: 'Assistant',
    title: 'Assistant',
    summary: 'Ask questions about your image in plain language.',
    icon: MessageSquare,
    Panel: AssistantPanel,
    status: (ws) => (ws.isExecuting && ws.busy === 'chat' ? 'busy' : null),
    layers: singleLayers,
    evidenceRun: scanEvidence,
    uploadTargets: singleTargets,
  },
  {
    id: 'scans',
    group: 'single',
    label: 'Scans',
    title: 'Feature scans',
    summary: 'Find mining, agriculture or deforestation in your image.',
    icon: ScanSearch,
    Panel: ScansPanel,
    status: (ws) => (ws.isExecuting && String(ws.busy).startsWith('scan:') ? 'busy' : null),
    layers: singleLayers,
    evidenceRun: scanEvidence,
    uploadTargets: singleTargets,
  },
  {
    id: 'change',
    group: 'pair',
    label: 'Change',
    title: 'Change detection',
    summary: 'Compare two images of the same sensor type for change.',
    icon: GitCompareArrows,
    Panel: ChangePanel,
    status: (ws) => (ws.isExecuting && ws.busy === 'change' ? 'busy' : ws.latest.change ? 'ready' : null),
    layers: (ws, run) => [
      slotLayer(ws, 'before', `Before · ${SENSOR[ws.slots.before.modality]}`),
      slotLayer(ws, 'after', `After · ${SENSOR[ws.slots.after.modality]}`),
      { value: 'swipe', label: 'Swipe', disabled: !(ws.slots.before.file && ws.slots.after.file) },
      evidenceLayer(run, 'Changes'),
    ],
    evidenceRun: (ws) => ws.latest.change,
    uploadTargets: () => [{ id: 'before', label: 'Before' }, { id: 'after', label: 'After' }],
  },
  {
    id: 'fusion',
    group: 'pair',
    label: 'Fusion',
    title: 'Optical–SAR fusion',
    summary: 'Fuse an optical image with a SAR image of the same place.',
    icon: Merge,
    Panel: FusionPanel,
    status: (ws) => (ws.isExecuting && ws.busy === 'fusion' ? 'busy' : ws.latest.fusion ? 'ready' : null),
    layers: (ws, run) => [slotLayer(ws, 'optical', 'Optical'), slotLayer(ws, 'sar', 'SAR'), evidenceLayer(run, 'Composite')],
    evidenceRun: (ws) => ws.latest.fusion,
    uploadTargets: () => [{ id: 'optical', label: 'Optical image' }, { id: 'sar', label: 'SAR image' }],
  },
  {
    id: 'report',
    group: 'output',
    label: 'Report',
    title: 'Report',
    summary: 'The evidence, exactly what ran, and the audit PDF.',
    icon: FileText,
    Panel: ReportPanel,
    status: (ws) => (ws.runs.length > 0 ? 'ready' : null),
    layers: (ws, run) => [evidenceLayer(run)],
    evidenceRun: (ws) => ws.reportRun,
    uploadTargets: () => [],
  },
];

export const featureById = (id) => FEATURES.find((f) => f.id === id) || FEATURES[0];

// The viewer layer to show: the one asked for if this feature has it (an empty slot layer is fine: it shows an
// upload prompt), else the first layer with content, else the first usable one. So switching from Change
// detection to Fusion never leaves the viewer on "Before", and an empty feature offers its first upload.
export function effectiveLayer(layers, wanted) {
  const usable = (layer) => layer && !layer.disabled;
  const exact = layers.find((l) => l.value === wanted);
  if (usable(exact)) return exact.value;
  return (layers.find((l) => usable(l) && !l.empty) ?? layers.find(usable))?.value ?? null;
}
