import React from 'react';
import ImageSlot from '../ImageSlot';

// The single image that the Assistant and the Feature scans work on. Two-image work (change detection,
// optical-SAR fusion) has its own inputs in its own panel.
export default function ImageryPanel({ ws, goTo }) {
  const { slots, setImage, removeImage, setModality } = ws;
  const link = 'cursor-pointer font-semibold text-blue-300 underline';
  return (
    <div className="space-y-3">
      <ImageSlot id="a" title="Image" slot={slots.a} onFile={setImage} onRemove={removeImage} sensor="toggle" onSensor={setModality} />
      <p className="px-1 text-xs text-slate-400">
        Comparing two images? Use <button type="button" className={link} onClick={() => goTo('change')}>Change detection</button>
        {' '}or <button type="button" className={link} onClick={() => goTo('fusion')}>Optical–SAR fusion</button>.
      </p>
    </div>
  );
}
