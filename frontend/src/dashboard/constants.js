import { Pickaxe, Sprout, Trees } from 'lucide-react';

// Shared dashboard constants (kept out of the .jsx component files so those only export components).

// What the file pickers accept.
export const ACCEPT = 'image/jpeg,image/png,image/webp,image/tiff,.tif,.tiff';

// The sensor-type choices for an image slot.
export const SENSORS = [
  { value: 'optical', label: 'Optical', title: 'Colour imagery' },
  { value: 'sar', label: 'SAR', title: 'Radar imagery: despeckled and described by brightness, not colour' },
];

// The three LoRA feature scans.
export const SCANS = [
  { id: 'mining', title: 'Mining', icon: Pickaxe, tone: 'amber', description: 'Surface extraction pits, quarries and open-cast workings.' },
  { id: 'agriculture', title: 'Agriculture', icon: Sprout, tone: 'emerald', description: 'Cultivated fields and crop rows.' },
  { id: 'deforestation', title: 'Deforestation', icon: Trees, tone: 'rose', description: 'Clear-cut land, active logging and canopy loss.' },
];
