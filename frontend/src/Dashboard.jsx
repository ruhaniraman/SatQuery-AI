import React, { useState, useRef } from 'react';
import { 
  Upload, Play, MessageSquare, ShieldCheck, 
  Download, Cpu, Eye, Loader2 
} from 'lucide-react';

// Define your backend URL here
const BACKEND_URL = 'http://localhost:8000';

export default function SatQueryDashboard() {
  const [activeLayer, setActiveLayer] = useState('imageA');
  
  // Input State
  const [query, setQuery] = useState('');
  const [imageA, setImageA] = useState(null);
  const [imageB, setImageB] = useState(null);
  const [previewA, setPreviewA] = useState(null);
  const [previewB, setPreviewB] = useState(null);

  // Execution State
  const [isExecuting, setIsExecuting] = useState(false);
  const [executionResult, setExecutionResult] = useState(null);
  const [error, setError] = useState(null);

  // Refs for hidden file inputs
  const fileInputARef = useRef(null);
  const fileInputBRef = useRef(null);

  const handleImageUpload = (e, setImage, setPreview) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
      setPreview(URL.createObjectURL(file));
    }
  };

  const handleRunPipeline = async (e) => {
    e.preventDefault();
    if (!query || !imageA) {
      setError("Please provide at least Image A and a query.");
      return;
    }

    setIsExecuting(true);
    setError(null);

    const formData = new FormData();
    formData.append('query', query);
    formData.append('images', imageA);
    if (imageB) {
      formData.append('images', imageB);
    }

    try {
      const response = await fetch(`${BACKEND_URL}/analyze`, {
        method: 'POST',
        body: formData, // Browser automatically sets Content-Type to multipart/form-data
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Analysis failed');
      }

      const data = await response.json();
      setExecutionResult(data);
      setActiveLayer('evidence'); // Auto-switch to the result mask
    } catch (err) {
      setError(err.message);
    } finally {
      setIsExecuting(false);
    }
  };

  // Determine what to show in the center canvas
  const renderCenterCanvas = () => {
    if (activeLayer === 'evidence' && executionResult?.visual_evidence_url) {
      return <img src={`${BACKEND_URL}${executionResult.visual_evidence_url}`} alt="AI Evidence" className="object-contain w-full h-full" />;
    }
    if (activeLayer === 'imageB' && previewB) {
      return <img src={previewB} alt="Image B" className="object-contain w-full h-full" />;
    }
    if (previewA) {
      return <img src={previewA} alt="Image A" className="object-contain w-full h-full" />;
    }
    return <div className="flex items-center justify-center h-full text-slate-500">No image selected</div>;
  };

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
      </header>

      {/* Main Workspace Layout */}
      <div className="flex flex-1 overflow-hidden">
        
        {/* Left Sidebar: Data & Pipeline Controls */}
        <aside className="w-80 bg-slate-900 border-r border-slate-800 p-4 flex flex-col justify-between overflow-y-auto shrink-0">
          <div className="space-y-6">
            <div>
              <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3 flex items-center">
                <Upload size={14} className="mr-2 text-emerald-400" /> 1. Geospatial Inputs
              </h3>
              
              <div className="space-y-3">
                {/* Image A Upload */}
                <input 
                  type="file" 
                  accept="image/jpeg, image/png, image/webp, image/tiff" 
                  className="hidden" 
                  ref={fileInputARef} 
                  onChange={(e) => handleImageUpload(e, setImageA, setPreviewA)} 
                />
                <div 
                  onClick={() => fileInputARef.current.click()}
                  className={`border-2 border-dashed rounded-lg p-3 text-center transition cursor-pointer bg-slate-950/50 ${imageA ? 'border-emerald-500' : 'border-slate-700 hover:border-emerald-500'}`}
                >
                  <p className="text-xs text-slate-300 font-medium">
                    {imageA ? imageA.name : "Upload Image A (Required)"}
                  </p>
                  <span className="text-[10px] text-slate-500">GeoTIFF / PNG / JPG</span>
                </div>

                {/* Image B Upload */}
                <input 
                  type="file" 
                  accept="image/jpeg, image/png, image/webp, image/tiff" 
                  className="hidden" 
                  ref={fileInputBRef} 
                  onChange={(e) => handleImageUpload(e, setImageB, setPreviewB)} 
                />
                <div 
                  onClick={() => fileInputBRef.current.click()}
                  className={`border-2 border-dashed rounded-lg p-3 text-center transition cursor-pointer bg-slate-950/50 ${imageB ? 'border-emerald-500' : 'border-slate-700 hover:border-emerald-500'}`}
                >
                  <p className="text-xs text-slate-300 font-medium">
                    {imageB ? imageB.name : "Upload Image B (Optional)"}
                  </p>
                  <span className="text-[10px] text-slate-500">For Change Detection / Fusion</span>
                </div>
              </div>
            </div>
          </div>
        </aside>

        {/* Center Canvas: Image Viewer */}
        <main className="flex-1 bg-slate-950 relative flex flex-col p-4">
          <div className="flex-1 border border-slate-800 rounded-xl overflow-hidden relative shadow-inner flex flex-col bg-black">

            {/* Floating Top Bar */}
            <div className="absolute top-4 left-4 right-4 z-[400] flex justify-between items-center bg-slate-900/90 backdrop-blur border border-slate-800 px-4 py-2 rounded-lg text-xs pointer-events-auto">
              <div className="flex items-center space-x-2 text-slate-300">
                <Eye size={14} className="text-emerald-400" />
                <span>Active Layer: <strong className="text-emerald-400 uppercase">{activeLayer}</strong></span>
              </div>
              <div className="flex space-x-2">
                <button 
                  onClick={() => setActiveLayer('imageA')}
                  disabled={!previewA}
                  className={`px-2.5 py-1 rounded transition cursor-pointer disabled:opacity-50 ${activeLayer === 'imageA' ? 'bg-emerald-600 text-slate-950 font-medium' : 'bg-slate-800 hover:bg-slate-700 text-slate-300'}`}
                >
                  Image A
                </button>
                <button 
                  onClick={() => setActiveLayer('imageB')}
                  disabled={!previewB}
                  className={`px-2.5 py-1 rounded transition cursor-pointer disabled:opacity-50 ${activeLayer === 'imageB' ? 'bg-emerald-600 text-slate-950 font-medium' : 'bg-slate-800 hover:bg-slate-700 text-slate-300'}`}
                >
                  Image B
                </button>
                <button 
                  onClick={() => setActiveLayer('evidence')}
                  disabled={!executionResult}
                  className={`px-2.5 py-1 rounded transition cursor-pointer disabled:opacity-50 ${activeLayer === 'evidence' ? 'bg-emerald-600 text-slate-950 font-medium' : 'bg-slate-800 hover:bg-slate-700 text-slate-300'}`}
                >
                  Visual Evidence
                </button>
              </div>
            </div>

            {/* Image Renderer */}
            {renderCenterCanvas()}

          </div>
        </main>

        {/* Right Drawer: AI Agent & Audit Trace */}
        <section className="w-96 bg-slate-900 border-l border-slate-800 flex flex-col justify-between p-4 overflow-y-auto shrink-0">
          
          <div className="space-y-4">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center">
              <MessageSquare size={14} className="mr-2 text-emerald-400" /> AI Assistant & VQA
            </h3>
            
            <div className="bg-slate-950 rounded-lg p-3 border border-slate-800 min-h-40 max-h-64 overflow-y-auto space-y-3 text-xs">
              <div className="bg-slate-900 p-2.5 rounded-lg border border-slate-800/60 text-slate-300">
                 Ask a natural language query about your remote-sensing imagery.
              </div>
              
              {isExecuting && (
                <div className="flex items-center justify-center space-x-2 text-emerald-400 p-4">
                  <Loader2 size={16} className="animate-spin" />
                  <span>Processing pipeline...</span>
                </div>
              )}

              {error && (
                <div className="bg-red-950/30 border border-red-800/40 p-2.5 rounded-lg text-red-200">
                  <p className="font-semibold">Error:</p>
                  <p>{error}</p>
                </div>
              )}

              {executionResult && (
                <div className="bg-emerald-950/30 border border-emerald-800/40 p-2.5 rounded-lg text-emerald-200 space-y-1">
                  <p className="font-semibold">Answer:</p>
                  <p className="whitespace-pre-wrap">{executionResult.answer}</p>
                </div>
              )}
            </div>

            {/* Query Input Box */}
            <form onSubmit={handleRunPipeline} className="relative">
              <input 
                type="text" 
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Ask about changes, land cover..."
                className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-emerald-500 pr-10"
                disabled={isExecuting}
              />
              <button 
                type="submit"
                disabled={isExecuting || !query}
                className="absolute right-2 top-2 text-emerald-400 hover:text-emerald-300 cursor-pointer disabled:opacity-50"
              >
                <Play size={14} />
              </button>
            </form>
          </div>

          {/* Bottom: Execution Trace & Reports */}
          <div className="space-y-4 mt-6 pt-4 border-t border-slate-800">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center">
              <ShieldCheck size={14} className="mr-2 text-emerald-400" /> Auditable Trace
            </h3>
            
            <div className="bg-slate-950 rounded-lg p-3 border border-slate-800 text-[11px] space-y-1.5 text-slate-400 font-mono overflow-y-auto max-h-32">
              {executionResult?.agent_execution_trace ? (
                <>
                  <p className="text-emerald-400">Pipeline ID: {executionResult.agent_execution_trace.pipeline_id}</p>
                  <p>Nodes: {executionResult.agent_execution_trace.nodes_traversed.join(" -> ")}</p>
                  {Object.entries(executionResult.agent_execution_trace.telemetry).map(([key, val]) => (
                    <p key={key}>{key}: <span className="text-slate-200">{val}</span></p>
                  ))}
                </>
              ) : (
                <p>Awaiting execution telemetry...</p>
              )}
            </div>

            {executionResult?.report_download_url ? (
              <a 
                href={`${BACKEND_URL}${executionResult.report_download_url}`}
                download
                className="w-full bg-emerald-600 hover:bg-emerald-500 text-slate-950 text-xs font-bold py-2 px-3 rounded-lg flex items-center justify-center space-x-2 transition cursor-pointer"
              >
                <Download size={14} />
                <span>Download PDF Report</span>
              </a>
            ) : (
              <button disabled className="w-full bg-slate-800 text-slate-500 text-xs font-medium py-2 px-3 rounded-lg flex items-center justify-center space-x-2 border border-slate-700 cursor-not-allowed">
                <Download size={14} />
                <span>Report Pending</span>
              </button>
            )}
          </div>

        </section>
      </div>
    </div>
  );
}