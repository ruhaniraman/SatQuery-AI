import React, { useId, useState } from 'react';

// Small drawn previews for the three capability cards on the landing page. They are illustrations of
// what each feature shows, not real model output.

// A small aerial scene: fields, a river, a road and a village. `cleared` adds a bare-ground patch
// where a block of forest used to be (the "after" date of the change preview).
function MiniScene({ cleared = false }) {
  return (
    <>
      <rect width="320" height="170" fill="#3f5a2b" />
      <rect x="4" y="4" width="92" height="70" rx="3" fill="#566b34" />
      <rect x="100" y="4" width="96" height="52" rx="3" fill="#6b6a3a" />
      <rect x="4" y="78" width="70" height="88" rx="3" fill="#5d7236" />
      <rect x="236" y="96" width="80" height="70" rx="3" fill="#6d6b3d" />
      {/* forest block */}
      <rect x="200" y="4" width="116" height="84" rx="4" fill="#244a22" />
      {Array.from({ length: 18 }).map((_, i) => (
        <circle key={i} cx={210 + (i % 6) * 19} cy={16 + Math.floor(i / 6) * 25} r="9" fill="#2d5a29" />
      ))}
      {cleared && (
        <>
          <path d="M226 14 H300 V70 H250 L226 52 Z" fill="#8a6e4b" />
          <path d="M232 22 H292" stroke="#9c8058" strokeWidth="3" />
          <path d="M236 40 H290" stroke="#9c8058" strokeWidth="3" />
          <path d="M250 58 H290" stroke="#9c8058" strokeWidth="3" />
        </>
      )}
      <path d="M-5 128 C 60 110 110 150 170 126 S 260 104 330 118" stroke="#1d3b4d" strokeWidth="13" fill="none" />
      <path d="M-5 128 C 60 110 110 150 170 126 S 260 104 330 118" stroke="#2c5872" strokeWidth="7" fill="none" />
      <path d="M150 -5 Q 176 60 160 96 T 176 175" stroke="#2b2f33" strokeWidth="6" fill="none" />
      <rect x="92" y="62" width="50" height="36" rx="2" fill="#6f6e66" />
      {[0, 1, 2].map((k) => <rect key={k} x={97 + k * 15} y="68" width="10" height="12" fill="#a19d90" />)}
    </>
  );
}

export function AskPreview() {
  return (
    <div className="sql-pv sql-pv-ask">
      <p className="sql-pv-bubble sql-pv-q">Is there a water body in this image?</p>
      <div className="sql-pv-bubble sql-pv-a">
        <strong>Answer: Yes.</strong> Two small ponds near the centre.
        <span className="sql-pv-conf"><i />High confidence</span>
      </div>
    </div>
  );
}

export function ChangePreview() {
  const [pos, setPos] = useState(55);
  const clip = useId().replace(/:/g, '');
  return (
    <div className="sql-pv sql-pv-change">
      <svg viewBox="0 0 320 170" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
        <defs>
          <clipPath id={clip}><rect x={(pos / 100) * 320} y="0" width="320" height="170" /></clipPath>
        </defs>
        <MiniScene />
        <g clipPath={`url(#${clip})`}>
          <MiniScene cleared />
          <rect x="220" y="9" width="86" height="66" rx="6" fill="none" stroke="#f59e0b" strokeWidth="2" strokeDasharray="5 4" />
        </g>
        <line x1={(pos / 100) * 320} y1="0" x2={(pos / 100) * 320} y2="170" stroke="#fff" strokeWidth="2" />
      </svg>
      <span className="sql-pv-handle" style={{ left: `${pos}%` }} aria-hidden="true">‹ ›</span>
      <span className="sql-pv-tag" style={{ left: 8 }}>Before</span>
      <span className="sql-pv-tag" style={{ right: 8 }}>After</span>
      <input
        type="range" min="5" max="95" value={pos} onChange={(e) => setPos(Number(e.target.value))}
        aria-label="Drag to compare before and after" className="sql-pv-range"
      />
    </div>
  );
}

export function FusionPreview() {
  const id = useId().replace(/:/g, '');
  const bands = [
    { label: 'Optical', x: 0, filter: null },
    { label: 'Fused', x: 107, filter: `url(#${id}-fused)` },
    { label: 'SAR', x: 214, filter: `url(#${id}-sar)` },
  ];
  return (
    <div className="sql-pv sql-pv-fusion">
      <svg viewBox="0 0 320 170" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
        <defs>
          {/* SAR: grey backscatter with speckle */}
          <filter id={`${id}-sar`} x="0" y="0" width="100%" height="100%">
            <feColorMatrix type="saturate" values="0" result="grey" />
            <feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="1" seed="3" result="noise" />
            <feColorMatrix in="noise" type="saturate" values="0" result="speckle" />
            <feBlend in="grey" in2="speckle" mode="overlay" />
          </filter>
          {/* fused: optical colour, with the SAR-like structure lightly blended in */}
          <filter id={`${id}-fused`} x="0" y="0" width="100%" height="100%">
            <feColorMatrix type="saturate" values="0.75" result="soft" />
            <feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="1" seed="3" result="noise" />
            <feColorMatrix in="noise" type="matrix" values="0 0 0 0 0.5  0 0 0 0 0.5  0 0 0 0 0.5  0 0 0 0.35 0" result="faint" />
            <feBlend in="soft" in2="faint" mode="soft-light" />
          </filter>
          {bands.map((b) => (
            <clipPath key={b.label} id={`${id}-${b.label}`}><rect x={b.x} y="0" width="106" height="170" /></clipPath>
          ))}
        </defs>
        {bands.map((b) => (
          <g key={b.label} clipPath={`url(#${id}-${b.label})`}>
            <g filter={b.filter || undefined}><MiniScene /></g>
          </g>
        ))}
        <line x1="106.5" y1="0" x2="106.5" y2="170" stroke="rgba(3,7,18,0.9)" strokeWidth="2" />
        <line x1="213.5" y1="0" x2="213.5" y2="170" stroke="rgba(3,7,18,0.9)" strokeWidth="2" />
      </svg>
      <div className="sql-pv-bands">
        {bands.map((b) => <span key={b.label} className={b.label === 'Fused' ? 'is-on' : ''}>{b.label}</span>)}
      </div>
    </div>
  );
}
