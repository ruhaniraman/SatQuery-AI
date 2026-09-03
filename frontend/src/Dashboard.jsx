import React, { useState } from 'react';
import { 
  Upload, Layers, Play, MessageSquare, ShieldCheck, 
  Download, MapPin, Cpu, Eye 
} from 'lucide-react';
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';

export default function SatQueryDashboard() {
  const [activeLayer, setActiveLayer] = useState('fused');
  const [query, setQuery] = useState('');
  const [isExecuting, setIsExecuting] = useState(false);
  const [executionResult, setExecutionResult] = useState(null);

  const handleRunPipeline = (e) => {
    e.preventDefault();
    setIsExecuting(true);
    setTimeout(() => {
      setIsExecuting(false);
      setExecutionResult({
        task: "Change Detection & CD-VQA",
        model: "cdvqa_engine.py + optical_sar_fusion_model.py",
        confidence: "96.4%",
        summary: "Detected 14.2 sq km of new built-up infrastructure replacing agricultural land.",
      });
    }, 1500);
  };

  const position = [12.9716, 77.5946];

  return (
    <div className="flex flex-col h-screen bg-slate-950 text-slate-100 font-sans overflow-hidden">
      
      {/* Top Navbar */}
      <header className="flex items-center justify-between px-6 py-3 bg-slate-900 border-b border-slate-800 shrink-0">
        <div className="flex items-center space-x-3">
          <div className="bg-emerald-500 p-2 rounded-lg text-slate-950 font-bold">
            <Cpu size={20} />
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-wide">SatQuery-AI</h1>
            <p className="text-xs text-slate-400">Agentic Remote-Sensing Intelligence Platform</p>
          </div>
        </div>
        <div className="flex items-center space-x-4 text-sm">
          <span className="flex items-center text-emerald-400 bg-emerald-950/50 px-3 py-1 rounded-full border border-emerald-800/50 text-xs">
            <span className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse mr-2"></span>
            Agent Ready
          </span>
          <span className="text-slate-400 text-xs font-medium">ISRO/SAC Evaluation Mode</span>
        </div>
      </header>

      {/* Main Workspace Layout */}
      <div className="flex flex-1 overflow-hidden">
        
        {/* Left Sidebar: Data & Pipeline Controls */}
        <aside className="w-80 bg-slate-900 border-r border-slate-800 p-4 flex flex-col justify-between overflow-y-auto shrink-0">
          <div className="space-y-6">
            
            {/* Step 1: Data Input */}
            <div>
              <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3 flex items-center">
                <Upload size={14} className="mr-2 text-emerald-400" /> 1. Geospatial Inputs
              </h3>
              <div className="space-y-3">
                <div className="border-2 border-dashed border-slate-700 rounded-lg p-3 text-center hover:border-emerald-500 transition cursor-pointer bg-slate-950/50">
                  <p className="text-xs text-slate-300 font-medium">Upload Image A (Pre / Optical)</p>
                  <span className="text-[10px] text-slate-500">GeoTIFF / TIFF supported</span>
                </div>
                <div className="border-2 border-dashed border-slate-700 rounded-lg p-3 text-center hover:border-emerald-500 transition cursor-pointer bg-slate-950/50">
                  <p className="text-xs text-slate-300 font-medium">Upload Image B (Post / SAR)</p>
                  <span className="text-[10px] text-slate-500">Co-registered pairs</span>
                </div>
              </div>
            </div>

            {/* Step 2: Pipeline Configuration */}
            <div>
              <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3 flex items-center">
                <Layers size={14} className="mr-2 text-emerald-400" /> 2. Specialist Modules
              </h3>
              <div className="space-y-2 text-sm">
                <label className="flex items-center space-x-2 text-slate-300 cursor-pointer">
                  <input type="checkbox" defaultChecked className="rounded bg-slate-800 border-slate-700 text-emerald-500 focus:ring-0" />
                  <span className="text-xs">Spatial Alignment & Preprocessing</span>
                </label>
                <label className="flex items-center space-x-2 text-slate-300 cursor-pointer">
                  <input type="checkbox" defaultChecked className="rounded bg-slate-800 border-slate-700 text-emerald-500 focus:ring-0" />
                  <span className="text-xs">Optical-SAR Fusion Engine</span>
                </label>
                <label className="flex items-center space-x-2 text-slate-300 cursor-pointer">
                  <input type="checkbox" defaultChecked className="rounded bg-slate-800 border-slate-700 text-emerald-500 focus:ring-0" />
                  <span className="text-xs">Change Segmentation Mask</span>
                </label>
              </div>
            </div>

          </div>

          {/* Run Button */}
          <button 
            onClick={handleRunPipeline}
            disabled={isExecuting}
            className="w-full mt-4 bg-emerald-600 hover:bg-emerald-500 text-slate-950 font-semibold py-2.5 px-4 rounded-lg flex items-center justify-center space-x-2 transition disabled:opacity-50 cursor-pointer"
          >
            {isExecuting ? (
              <span className="animate-spin w-4 h-4 border-2 border-slate-950 border-t-transparent rounded-full"></span>
            ) : (
              <Play size={16} fill="currentColor" />
            )}
            <span className="text-xs">{isExecuting ? "Agent Executing..." : "Run Agentic Pipeline"}</span>
          </button>
        </aside>

        {/* Center Canvas: Interactive Map Viewer */}
        <main className="flex-1 bg-slate-950 relative flex flex-col p-4">
          <div className="flex-1 border border-slate-800 rounded-xl overflow-hidden relative shadow-inner flex flex-col">
            
            <MapContainer 
              center={position} 
              zoom={13} 
              style={{ width: '100%', height: '100%', background: '#020617' }}
              zoomControl={false}
            >
              <TileLayer
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              />
              <Marker position={position}>
                <Popup>
                  SatQuery-AI Target Region <br /> Analysis Area.
                </Popup>
              </Marker>
            </MapContainer>

            {/* Floating Top Bar on Map */}
            <div className="absolute top-4 left-4 right-4 z-[400] flex justify-between items-center bg-slate-900/90 backdrop-blur border border-slate-800 px-4 py-2 rounded-lg text-xs pointer-events-auto">
              <div className="flex items-center space-x-2 text-slate-300">
                <Eye size={14} className="text-emerald-400" />
                <span>Active Layer: <strong className="text-emerald-400 uppercase">{activeLayer}</strong></span>
              </div>
              <div className="flex space-x-2">
                <button 
                  onClick={() => setActiveLayer('optical')}
                  className={`px-2.5 py-1 rounded transition cursor-pointer ${activeLayer === 'optical' ? 'bg-emerald-600 text-slate-950 font-medium' : 'bg-slate-800 hover:bg-slate-700 text-slate-300'}`}
                >
                  Optical
                </button>
                <button 
                  onClick={() => setActiveLayer('sar')}
                  className={`px-2.5 py-1 rounded transition cursor-pointer ${activeLayer === 'sar' ? 'bg-emerald-600 text-slate-950 font-medium' : 'bg-slate-800 hover:bg-slate-700 text-slate-300'}`}
                >
                  SAR
                </button>
                <button 
                  onClick={() => setActiveLayer('fused')}
                  className={`px-2.5 py-1 rounded transition cursor-pointer ${activeLayer === 'fused' ? 'bg-emerald-600 text-slate-950 font-medium' : 'bg-slate-800 hover:bg-slate-700 text-slate-300'}`}
                >
                  Fused + Change Mask
                </button>
              </div>
            </div>

            {/* Floating Bottom Bar on Map */}
            <div className="absolute bottom-4 left-4 right-4 z-[400] flex justify-between items-center bg-slate-900/90 backdrop-blur border border-slate-800 px-4 py-2 rounded-lg text-xs pointer-events-auto">
              <span className="text-slate-400">Coordinates: <strong className="text-slate-200">12.9716° N, 77.5946° E</strong></span>
              <span className="text-slate-400">Sensor: <strong className="text-slate-200">Cartosat-2S / RISAT SAR</strong></span>
            </div>

          </div>
        </main>

        {/* Right Drawer: AI Agent & Audit Trace */}
        <section className="w-96 bg-slate-900 border-l border-slate-800 flex flex-col justify-between p-4 overflow-y-auto shrink-0">
          
          {/* Top: Chat / VQA Section */}
          <div className="space-y-4">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center">
              <MessageSquare size={14} className="mr-2 text-emerald-400" /> AI Assistant & VQA
            </h3>
            
            <div className="bg-slate-950 rounded-lg p-3 border border-slate-800 min-h-40 max-h-56 overflow-y-auto space-y-3 text-xs">
              <div className="bg-slate-900 p-2.5 rounded-lg border border-slate-800/60 text-slate-300">
                👋 Ask a natural language query about your remote-sensing imagery (e.g., "Has the built-up area increased?").
              </div>
              {executionResult && (
                <div className="bg-emerald-950/30 border border-emerald-800/40 p-2.5 rounded-lg text-emerald-200 space-y-1">
                  <p className="font-semibold">Answer:</p>
                  <p>{executionResult.summary}</p>
                </div>
              )}
            </div>

            {/* Query Input Box */}
            <div className="relative">
              <input 
                type="text" 
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Ask about changes, land cover..."
                className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-emerald-500 pr-10"
              />
              <button className="absolute right-2 top-2 text-emerald-400 hover:text-emerald-300 cursor-pointer">
                <Play size={14} />
              </button>
            </div>
          </div>

          {/* Bottom: Execution Trace & Reports */}
          <div className="space-y-4 mt-6 pt-4 border-t border-slate-800">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center">
              <ShieldCheck size={14} className="mr-2 text-emerald-400" /> Auditable Trace
            </h3>
            
            <div className="bg-slate-950 rounded-lg p-3 border border-slate-800 text-[11px] space-y-1.5 text-slate-400 font-mono">
              <p>Task: <span className="text-slate-200">{executionResult ? executionResult.task : 'Pending execution'}</span></p>
              <p>Model: <span className="text-slate-200">{executionResult ? executionResult.model : 'None selected'}</span></p>
              <p>Confidence: <span className="text-emerald-400">{executionResult ? executionResult.confidence : '-'}</span></p>
            </div>

            <button className="w-full bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium py-2 px-3 rounded-lg flex items-center justify-center space-x-2 transition border border-slate-700 cursor-pointer">
              <Download size={14} />
              <span>Generate PDF Report</span>
            </button>
          </div>

        </section>

      </div>
    </div>
  );
}