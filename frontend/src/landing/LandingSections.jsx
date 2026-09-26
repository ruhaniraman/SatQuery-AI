import { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  ArrowLeftRight, ArrowRight, FileText, Gauge, Image, Layers, Map, MessageSquareText,
  ScanSearch, ShieldCheck, FileCheck2,
} from 'lucide-react';
import logoMark from '../assets/logo-mark.png';
import earth from '../assets/earth.jpeg';
import { AskPreview, ChangePreview, FusionPreview } from './CapabilityPreviews';
import './landing.css';

// The sections below the landing hero. Everything here describes what the product really does;
// the preview is a drawn illustration of the dashboard, not a screenshot.

const PREVIEW_STEPS = [
  {
    nav: 1,
    question: 'Is there a water body in this image?',
    answer: <><strong>Answer: Yes.</strong> Two small ponds sit near the centre of the view, between the fields and the road.</>,
    badge: 'High confidence',
    layer: 'Image',
  },
  {
    nav: 2,
    question: 'Run the agriculture scan',
    answer: <><strong>Likely cultivated fields in 1 area</strong>, covering about 19% of the view, in the upper left. Each tile is scored on its own, and neighbouring positives are merged into numbered findings.</>,
    badge: 'Likely',
    layer: 'Highlights',
  },
];

const CAPABILITIES = [
  {
    icon: MessageSquareText,
    title: 'Ask in plain language',
    text: 'Upload an optical or SAR image, or capture the live map, and ask what is there. Yes/no and rural/urban questions get a short answer from a remote-sensing adapter, with the explanation underneath.',
    Preview: AskPreview,
  },
  {
    icon: ArrowLeftRight,
    title: 'Change detection',
    text: 'Compare two dates of the same place. Changed areas are measured from the pixels, cross-checked by the specialist scans on both dates, and described region by region.',
    Preview: ChangePreview,
  },
  {
    icon: Layers,
    title: 'Optical + SAR fusion',
    text: 'Combine an optical image with a radar image of the same ground. The composite keeps optical colour and adds SAR structure, which sees through cloud.',
    Preview: FusionPreview,
  },
];

const FEATURES = [
  { icon: ScanSearch, title: 'Specialist scans', text: 'Mining, deforestation and agriculture adapters score the image tile by tile, then merge positive tiles into numbered findings with a plain headline and a warning when a scan over-reports.' },
  { icon: Map, title: 'Ask from the live map', text: 'Fly to any place or coordinate, zoom in, and ask about exactly what is on screen. Scan results stay pinned to the ground as you pan.' },
  { icon: Image, title: 'GeoTIFF and SAR ready', text: 'GeoTIFFs are read with their coordinate system, footprints are checked for overlap, and SAR is despeckled in the linear domain before display.' },
  { icon: Gauge, title: 'Confidence you can read', text: 'Short answers carry a High / Medium / Low badge derived from the model’s own probability, so you know when to double-check.' },
  { icon: FileCheck2, title: 'Audit-ready reports', text: 'Every run records which task ran, which model and adapter were used, and with what settings. Download it as JSON or as a PDF with the evidence image.' },
  { icon: ShieldCheck, title: 'Runs on your machine', text: 'The model runs locally on a single GPU. Imagery you analyse is never sent to a third-party AI service.' },
];

const STEPS = [
  { title: 'Bring an image', text: 'Drop in a GeoTIFF, PNG or JPEG, tag it optical or SAR, or capture the satellite map straight from the dashboard.' },
  { title: 'Ask or scan', text: 'Type a question, run a specialist scan, or add a second image to compare dates or fuse optical with radar.' },
  { title: 'Get evidence', text: 'Read the answer next to highlighted evidence, then export the full trace and report for your team.' },
];

const FACTS = [
  { value: '3', label: 'analysis modes: questions, change detection and optical–SAR fusion' },
  { value: '4', label: 'remote-sensing adapters: visual Q&A plus three specialist scans' },
  { value: '6 GB', label: 'of GPU memory is enough to run it all locally' },
  { value: '100%', label: 'of runs come with an execution trace and a downloadable report' },
];

function useReveal(rootRef) {
  useEffect(() => {
    const root = rootRef.current;
    if (!root) return undefined;
    const items = root.querySelectorAll('.sql-reveal');
    if (typeof IntersectionObserver === 'undefined') {
      items.forEach((el) => el.classList.add('is-in'));
      return undefined;
    }
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) {
          e.target.classList.add('is-in');
          io.unobserve(e.target);
        }
      });
    }, { threshold: 0.12, rootMargin: '0px 0px -40px 0px' });
    items.forEach((el) => io.observe(el));
    return () => io.disconnect();
  }, [rootRef]);
}

function prefersReducedMotion() {
  return typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
}

// Types the current question out, then shows the answer; cycles through PREVIEW_STEPS.
function usePreviewCycle() {
  const [step, setStep] = useState(0);
  const [typed, setTyped] = useState(PREVIEW_STEPS[0].question.length);
  const [answered, setAnswered] = useState(true);

  useEffect(() => {
    if (prefersReducedMotion()) return undefined;
    let current = 0;
    let timer;
    const typeQuestion = (i) => {
      setTyped(i);
      if (i < PREVIEW_STEPS[current].question.length) {
        timer = setTimeout(() => typeQuestion(i + 1), 45);
      } else {
        timer = setTimeout(() => {
          setAnswered(true);
          timer = setTimeout(next, 5200);
        }, 700);
      }
    };
    const next = () => {
      current = (current + 1) % PREVIEW_STEPS.length;
      setStep(current);
      setAnswered(false);
      setTyped(0);
      timer = setTimeout(() => typeQuestion(1), 600);
    };
    // The first example starts fully shown, so the preview is never empty.
    timer = setTimeout(next, 5200);
    return () => clearTimeout(timer);
  }, []);

  return { step: PREVIEW_STEPS[step], typed, answered };
}

// A drawn aerial scene: field patchwork, a river, a road and two ponds near the centre.
function AerialScene({ mode, answered }) {
  const fields = [
    [0, 0, 150, 120, '#3f5a2b'], [150, 0, 130, 120, '#566b34'], [280, 0, 170, 95, '#6b6a3a'], [450, 0, 150, 110, '#4a6130'],
    [0, 120, 110, 110, '#5d7236'], [110, 120, 170, 110, '#3b5328'], [450, 110, 150, 120, '#7a7043'],
    [0, 230, 150, 110, '#6d6b3d'], [0, 340, 150, 100, '#435c2c'], [150, 330, 160, 110, '#5a6e35'],
    [310, 320, 140, 120, '#3e5629'], [450, 230, 150, 210, '#546a33'],
  ];
  const showScan = mode === 'Highlights' && answered;
  const showPonds = mode === 'Image' && answered;
  const hot = [[0, 0, 0.9], [0, 1, 0.85], [1, 0, 0.82], [1, 1, 0.6], [0, 2, 0.55]];
  return (
    <svg viewBox="0 0 600 440" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
      <rect width="600" height="440" fill="#4b6230" />
      {fields.map(([x, y, w, h, c], i) => (
        <g key={i}>
          <rect x={x + 3} y={y + 3} width={w - 6} height={h - 6} fill={c} rx="3" />
          {i % 3 === 0 && Array.from({ length: Math.floor(w / 14) }).map((_, k) => (
            <line key={k} x1={x + 8 + k * 14} y1={y + 8} x2={x + 8 + k * 14} y2={y + h - 8} stroke="rgba(0,0,0,0.12)" strokeWidth="2" />
          ))}
        </g>
      ))}
      <path d="M280 95 Q 330 150 300 200 T 330 330 L 330 440" stroke="#2b2f33" strokeWidth="9" fill="none" />
      <path d="M280 95 Q 330 150 300 200 T 330 330 L 330 440" stroke="#8b8f93" strokeWidth="1.5" strokeDasharray="8 8" fill="none" />
      <path d="M-10 300 C 80 270 120 330 200 300 S 300 250 360 270 S 520 230 610 250" stroke="#1d3b4d" strokeWidth="16" fill="none" />
      <path d="M-10 300 C 80 270 120 330 200 300 S 300 250 360 270 S 520 230 610 250" stroke="#2c5872" strokeWidth="9" fill="none" />
      <rect x="160" y="245" width="100" height="70" fill="#6f6e66" rx="3" />
      {[0, 1, 2, 3].map((k) => <rect key={k} x={168 + k * 23} y="253" width="16" height="22" fill="#9a968a" />)}
      <ellipse cx="380" cy="175" rx="30" ry="20" fill="#1c4460" />
      <ellipse cx="425" cy="205" rx="20" ry="14" fill="#1f4a66" />
      <rect width="600" height="440" fill="url(#sqlVignette)" />
      <defs>
        <radialGradient id="sqlVignette" cx="50%" cy="50%" r="75%">
          <stop offset="60%" stopColor="#000" stopOpacity="0" />
          <stop offset="100%" stopColor="#000" stopOpacity="0.45" />
        </radialGradient>
      </defs>

      <g style={{ opacity: showPonds ? 1 : 0, transition: 'opacity 0.6s' }}>
        <ellipse cx="400" cy="190" rx="68" ry="48" fill="rgba(45,212,191,0.12)" stroke="#2dd4bf" strokeWidth="2" strokeDasharray="6 5" />
        <circle cx="462" cy="150" r="11" fill="#2dd4bf" />
        <text x="462" y="154" textAnchor="middle" fontSize="12" fontWeight="700" fill="#03201c">1</text>
      </g>

      <g style={{ opacity: showScan ? 1 : 0, transition: 'opacity 0.6s' }}>
        {hot.map(([r, c, p]) => (
          <rect key={`${r}-${c}`} x={c * 150} y={r * 110} width="150" height="110" fill={`rgba(245,158,11,${(p - 0.4) * 0.55})`} />
        ))}
        {Array.from({ length: 3 }).map((_, k) => (
          <g key={k}>
            <line x1={(k + 1) * 150} y1="0" x2={(k + 1) * 150} y2="440" stroke="rgba(255,255,255,0.14)" />
            <line x1="0" y1={(k + 1) * 110} x2="600" y2={(k + 1) * 110} stroke="rgba(255,255,255,0.14)" />
          </g>
        ))}
        <path d="M8 8 H292 V108 H148 V212 H8 Z" fill="none" stroke="#f59e0b" strokeWidth="2.5" strokeLinejoin="round" />
        <circle cx="120" cy="150" r="12" fill="#f59e0b" />
        <text x="120" y="154" textAnchor="middle" fontSize="12" fontWeight="700" fill="#1f1300">1</text>
      </g>
    </svg>
  );
}

function ProductPreview() {
  const { step, typed, answered } = usePreviewCycle();
  const railIcons = [Image, MessageSquareText, ScanSearch, ArrowLeftRight, Layers, FileText];
  const typing = typed < step.question.length;
  return (
    <div className="sql-preview sql-reveal">
      <div className="sql-window">
        <div className="sql-window-bar">
          <span className="sql-dot" /><span className="sql-dot" /><span className="sql-dot" />
          <span className="sql-window-title">SatQuery AI · Dashboard</span>
          <span className="sql-window-status"><i />Model ready</span>
        </div>
        <div className="sql-window-body">
          <div className="sql-rail">
            {railIcons.map((Icon, i) => (
              <span key={i} className={i === step.nav ? 'is-on' : ''}><Icon size={17} /></span>
            ))}
          </div>
          <div className="sql-chat">
            <p className="sql-chat-label">{step.nav === 2 ? 'Scans' : 'Assistant'}</p>
            {typed > 0 && (
              <p className={`sql-msg sql-msg-user${typing ? ' sql-typing' : ''}`}>
                {step.question.slice(0, typed)}
              </p>
            )}
            {answered && (
              <div className="sql-msg sql-msg-ai">
                {step.answer}
                <div><span className="sql-badge">● {step.badge}</span></div>
              </div>
            )}
            <div className="sql-chat-input">
              <span>Ask about this image…</span>
              <ArrowRight size={15} />
            </div>
          </div>
          <div className="sql-viewer">
            <AerialScene mode={step.layer} answered={answered} />
            {!answered && typed > 0 && <div className="sql-scanline" />}
            <span className="sql-viewer-chip" style={{ top: 12, right: 12 }}>{step.layer}</span>
            <span className="sql-viewer-meta">Optical · GeoTIFF · EPSG:32643</span>
          </div>
        </div>
      </div>
    </div>
  );
}

export function LandingNav({ shown, onTop }) {
  const navigate = useNavigate();
  return (
    <nav className={`sql-nav${shown ? ' is-shown' : ''}`} aria-hidden={!shown} style={{ '--sql-logo': `url(${logoMark})` }}>
      <button type="button" className="sql-brand" onClick={onTop} tabIndex={shown ? 0 : -1}>
        <span className="sql-brand-mark"><i /></span>
        <span className="sql-brand-name">SatQuery</span>
        <span className="sql-brand-tag">AI</span>
      </button>
      <div className="sql-nav-links">
        <a href="#product" tabIndex={shown ? 0 : -1}>Product</a>
        <a href="#capabilities" tabIndex={shown ? 0 : -1}>Capabilities</a>
        <a href="#features" tabIndex={shown ? 0 : -1}>Features</a>
        <a href="#how" tabIndex={shown ? 0 : -1}>How it works</a>
      </div>
      <button type="button" className="sql-btn sql-btn-primary sql-btn-sm" onClick={() => navigate('/dashboard')} tabIndex={shown ? 0 : -1}>
        Launch dashboard
      </button>
    </nav>
  );
}

export default function LandingSections() {
  const navigate = useNavigate();
  const rootRef = useRef(null);
  useReveal(rootRef);
  const open = () => navigate('/dashboard');

  return (
    <div ref={rootRef} className="sql-below" style={{ '--sql-logo': `url(${logoMark})` }}>
      <div className="sql-space" aria-hidden="true">
        <video autoPlay loop muted playsInline>
          <source src="/space-bg.mp4" type="video/mp4" />
        </video>
        <div className="sql-space-shade" />
      </div>
      <section id="product" className="sql-section" style={{ paddingTop: '8rem' }}>
        <div className="sql-glow" />
        <div className="sql-wrap sql-center">
          <p className="sql-eyebrow sql-reveal">✦ The product</p>
          <h2 className="sql-h2 sql-reveal"><b>Ask</b> the planet a <b>question</b></h2>
          <p className="sql-lead sql-reveal">
            SatQuery AI turns satellite imagery into answers you can check. Ask about an image, scan it for
            mining, deforestation or farmland, compare two dates, or fuse optical with radar, all from one dashboard.
          </p>
          <div className="sql-cta-actions sql-reveal">
            <button type="button" className="sql-btn sql-btn-primary" onClick={open}>Open the dashboard <ArrowRight size={16} /></button>
            <a className="sql-btn sql-btn-ghost" href="#how">See how it works</a>
          </div>
          <ProductPreview />
        </div>
      </section>

      <section id="capabilities" className="sql-section">
        <div className="sql-wrap">
          <p className="sql-eyebrow sql-reveal">✦ Capabilities</p>
          <h2 className="sql-h2 sql-reveal"><b>One</b> assistant for every <b>image</b></h2>
          <p className="sql-lead sql-reveal">
            The task follows what you upload: one image opens questions and scans, two dates open change
            detection, and an optical–SAR pair opens fusion.
          </p>
          <div className="sql-cards">
            {CAPABILITIES.map(({ icon: Icon, title, text, Preview }, i) => (
              <article key={title} className="sql-card sql-reveal" style={{ transitionDelay: `${i * 90}ms` }}>
                <div className="sql-icon"><Icon size={22} /></div>
                <h3>{title}</h3>
                <p>{text}</p>
                <div className="sql-card-foot"><Preview /></div>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section id="features" className="sql-section">
        <div className="sql-wrap sql-split">
          <div className="sql-sticky">
            <p className="sql-eyebrow sql-reveal">✦ Features</p>
            <h2 className="sql-h2 sql-reveal">Built for <b>analysts</b>, not <b>demos</b></h2>
            <p className="sql-lead sql-reveal">
              Every answer comes with the evidence behind it and a record of how it was produced, so results
              can be checked, shared and trusted.
            </p>
            <button type="button" className="sql-btn sql-btn-primary sql-reveal" onClick={open}>Get started <ArrowRight size={16} /></button>
          </div>
          <div className="sql-list">
            {FEATURES.map(({ icon: Icon, title, text }) => (
              <div key={title} className="sql-row sql-reveal">
                <div className="sql-icon"><Icon size={20} /></div>
                <div>
                  <h4>{title}</h4>
                  <p>{text}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section id="how" className="sql-section">
        <div className="sql-wrap">
          <p className="sql-eyebrow sql-reveal">✦ How it works</p>
          <h2 className="sql-h2 sql-reveal">From <b>pixels</b> to <b>answers</b> in three steps</h2>
          <div className="sql-steps">
            {STEPS.map(({ title, text }, i) => (
              <div key={title} className="sql-step sql-reveal" style={{ transitionDelay: `${i * 90}ms` }}>
                <p className="sql-step-num">0{i + 1}</p>
                <h4>{title}</h4>
                <p>{text}</p>
              </div>
            ))}
          </div>
          <div className="sql-facts sql-reveal">
            {FACTS.map(({ value, label }) => (
              <div key={value} className="sql-fact">
                <p className="sql-fact-value"><span>{value}</span></p>
                <p className="sql-fact-label">{label}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="sql-section">
        <div className="sql-wrap">
          <div className="sql-cta sql-reveal">
            <div className="sql-cta-bg" style={{ backgroundImage: `url(${earth})` }} />
            <p className="sql-eyebrow">✦ Ready when you are</p>
            <h2 className="sql-h2">Start <b>exploring</b> the Earth</h2>
            <p className="sql-lead">Sign in, drop in an image or pick a spot on the map, and ask your first question.</p>
            <div className="sql-cta-actions">
              <button type="button" className="sql-btn sql-btn-primary" onClick={open}>Launch dashboard <ArrowRight size={16} /></button>
            </div>
          </div>
        </div>
      </section>

      <footer className="sql-footer">
        <div className="sql-wrap">
          <div className="sql-footer-left">
            <span className="sql-brand" style={{ cursor: 'default' }}>
              <span className="sql-brand-mark"><i /></span>
              <span className="sql-brand-name">SatQuery AI</span>
            </span>
          </div>
          <div className="sql-footer-links">
            <a href="#capabilities">Assistant</a>
            <a href="#capabilities">Change detection</a>
            <a href="#capabilities">Optical + SAR</a>
            <a href="#features">Reports</a>
          </div>
        </div>
      </footer>
    </div>
  );
}
