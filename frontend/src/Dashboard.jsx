import React, { useState, useRef } from 'react';
import {
  Upload, MessageSquare, ShieldCheck,
  Download, Cpu, Eye, ImageIcon, Play, Loader2, User, Bot, Plus, X
} from 'lucide-react';
import earthBg from './assets/earth.jpeg'; 

const BACKEND_URL = 'http://localhost:8000';

export default function SatQueryDashboard() {
  const [activeLayer, setActiveLayer] = useState('imageA');
  
  // Input State
  const [query, setQuery] = useState('');
  const [imageA, setImageA] = useState(null);
  const [imageB, setImageB] = useState(null);
  const [previewA, setPreviewA] = useState(null);
  const [previewB, setPreviewB] = useState(null);
  const [showImageB, setShowImageB] = useState(false); 

  // Execution & Chat State
  const [isExecuting, setIsExecuting] = useState(false);
  const [executionResult, setExecutionResult] = useState(null);
  const [error, setError] = useState(null);
  const [chatHistory, setChatHistory] = useState([]); 

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

  const handleRemoveImageB = () => {
    setShowImageB(false);
    setImageB(null);
    setPreviewB(null);
    if (activeLayer === 'imageB' || activeLayer === 'evidence') {
      setActiveLayer('imageA');
    }
  };

  const handleRunPipeline = async (e) => {
    e.preventDefault();
    if (!query || !imageA) {
      setError("Please provide at least Image A and a query.");
      return;
    }

    const submittedQuery = query;
    setChatHistory(prev => [...prev, { role: 'user', content: submittedQuery }]);
    setQuery('');
    
    setIsExecuting(true);
    setError(null);

    const formData = new FormData();
    formData.append('query', submittedQuery);
    formData.append('images', imageA);
    if (imageB && showImageB) {
      formData.append('images', imageB);
    }

    try {
      const response = await fetch(`${BACKEND_URL}/analyze`, {
        method: 'POST',
        body: formData, 
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Analysis failed');
      }

      const data = await response.json();
      setExecutionResult(data);
      
      // Remove Markdown asterisks from the AI's answer
      const cleanAnswer = data.answer.replace(/\*/g, '');
      
      setChatHistory(prev => [...prev, { role: 'ai', content: cleanAnswer }]);
      setActiveLayer('evidence'); 
    } catch (err) {
      setError(err.message);
    } finally {
      setIsExecuting(false);
    }
  };

  const isTiff = (file) => file && file.name.toLowerCase().match(/\.tiff?$/);

  const renderCenterCanvas = () => {
    let content = null;

    if (activeLayer === 'evidence' && executionResult?.visual_evidence_url) {
      content = <img src={`${BACKEND_URL}${executionResult.visual_evidence_url}`} alt="AI Evidence" className="object-contain w-full h-full z-10 relative" />;
    } else if (activeLayer === 'imageB' && imageB) {
      content = isTiff(imageB) ? (
        <div className="flex flex-col items-center justify-center h-full z-10 relative text-blue-400/70">
          <ImageIcon size={48} className="mb-3" />
          <p className="font-semibold tracking-wide">GeoTIFF Loaded</p>
          <p className="text-xs text-slate-400 mt-1">{imageB.name}</p>
        </div>
      ) : previewB ? (
        <img src={previewB} alt="Image B" className="object-contain w-full h-full z-10 relative" />
      ) : null;
    } else if (activeLayer === 'imageA' && imageA) {
      content = isTiff(imageA) ? (
        <div className="flex flex-col items-center justify-center h-full z-10 relative text-blue-400/70">
          <ImageIcon size={48} className="mb-3" />
          <p className="font-semibold tracking-wide">GeoTIFF Loaded</p>
          <p className="text-xs text-slate-400 mt-1">{imageA.name}</p>
        </div>
      ) : previewA ? (
        <img src={previewA} alt="Image A" className="object-contain w-full h-full z-10 relative" />
      ) : null;
    }

    return (
      <>
        <svg className="absolute inset-0 w-full h-full opacity-10 pointer-events-none" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <pattern id="grid" width="32" height="32" patternUnits="userSpaceOnUse">
              <path d="M 32 0 L 0 0 0 32" fill="none" stroke="#94a3b8" strokeWidth="1" />
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#grid)" />
        </svg>
        {content}
      </>
    );
  };

  return (
    <div
      className="flex flex-col h-screen text-slate-100 font-sans overflow-hidden relative"
      style={{
        backgroundImage: `url(${earthBg})`,
        backgroundSize: 'cover',
        backgroundPosition: 'center',
      }}
    >
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          background:
            'linear-gradient(to bottom, rgba(0,0,0,0.80) 0%, rgba(0,0,0,0.40) 45%, rgba(0,0,0,0.75) 100%)',
        }}
      />

      <div className="relative z-10 flex flex-col h-full">

        {/* Top Navbar - Pushed up with reduced padding */}
        <header className="flex items-center justify-between px-6 py-1.5 bg-black/50 backdrop-blur-md border-b border-white/10 shrink-0">
          <div className="flex items-center space-x-3">
            <div className="bg-blue-500/90 p-1.5 rounded-lg text-slate-950 font-bold">
              <Cpu size={20} />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-wide leading-none">SatQuery-AI</h1>
            </div>
          </div>
          <div className="flex items-center space-x-4 text-sm">
            <span className="flex items-center text-blue-300 bg-blue-950/40 backdrop-blur-sm px-3 py-1 rounded-full border border-blue-700/40 text-xs">
              <span className="w-2 h-2 bg-blue-400 rounded-full animate-pulse mr-2"></span>
              Agent Ready
            </span>
          </div>
        </header>

        <div className="flex flex-1 overflow-hidden">

          {/* LEFT: Controls & Map - Set to exactly 50% width */}
          <main className="w-1/2 flex flex-col overflow-hidden p-4 space-y-4">

            <div className="bg-black/50 backdrop-blur-md border border-white/10 rounded-xl p-3 flex items-center gap-4 shrink-0 overflow-x-auto">
              <div className="flex items-center gap-2 text-xs text-slate-300 font-semibold uppercase tracking-wider shrink-0 pr-2 border-r border-white/10">
                <Upload size={14} className="text-blue-400" /> Inputs
              </div>
              
              <input 
                type="file" 
                accept="image/jpeg, image/png, image/webp, image/tiff, .tif" 
                className="hidden" 
                ref={fileInputARef} 
                onChange={(e) => handleImageUpload(e, setImageA, setPreviewA)} 
              />
              <div 
                onClick={() => fileInputARef.current.click()}
                className={`flex items-center gap-2 border border-dashed rounded-lg px-3 py-2 transition cursor-pointer bg-black/30 backdrop-blur-sm shrink-0 ${imageA ? 'border-blue-500 text-blue-300' : 'border-white/20 hover:border-blue-500 text-slate-200'}`}
              >
                <ImageIcon size={14} className={imageA ? "text-blue-400" : "text-slate-300"} />
                <span className="text-xs whitespace-nowrap">{imageA ? imageA.name : 'Image A · Optical/SAR'}</span>
              </div>

              {!showImageB ? (
                <button
                  type="button"
                  onClick={() => setShowImageB(true)}
                  className="flex items-center justify-center w-9 h-9 rounded-lg border border-dashed border-white/20 bg-black/30 hover:border-blue-500 text-slate-300 hover:text-blue-400 transition cursor-pointer shrink-0"
                  title="Add second image for Change Detection or Fusion"
                >
                  <Plus size={16} />
                </button>
              ) : (
                <div className="flex items-center gap-2 shrink-0">
                  <input 
                    type="file" 
                    accept="image/jpeg, image/png, image/webp, image/tiff, .tif" 
                    className="hidden" 
                    ref={fileInputBRef} 
                    onChange={(e) => handleImageUpload(e, setImageB, setPreviewB)} 
                  />
                  <div 
                    onClick={() => fileInputBRef.current.click()}
                    className={`flex items-center gap-2 border border-dashed rounded-lg px-3 py-2 transition cursor-pointer bg-black/30 backdrop-blur-sm ${imageB ? 'border-blue-500 text-blue-300' : 'border-white/20 hover:border-blue-500 text-slate-200'}`}
                  >
                    <ImageIcon size={14} className={imageB ? "text-blue-400" : "text-slate-300"} />
                    <span className="text-xs whitespace-nowrap">{imageB ? imageB.name : 'Image B · Optical/SAR'}</span>
                  </div>
                  <button
                    type="button"
                    onClick={handleRemoveImageB}
                    className="flex items-center justify-center w-9 h-9 rounded-lg border border-white/10 bg-black/40 hover:bg-red-500/20 hover:text-red-400 text-slate-400 transition cursor-pointer"
                    title="Remove Image B"
                  >
                    <X size={14} />
                  </button>
                </div>
              )}
            </div>

            <div className="flex-1 border border-white/10 rounded-xl overflow-hidden relative shadow-inner bg-black/30 backdrop-blur-sm">
              <div className="w-full h-full flex items-center justify-center relative">
                {renderCenterCanvas()}
              </div>

              <div className="absolute top-2 left-2 right-2 z-50 flex justify-between items-center bg-black/60 backdrop-blur-md border border-white/10 px-4 py-1.5 rounded-lg text-xs pointer-events-auto">
                <div className="flex items-center space-x-2 text-slate-200">
                  <Eye size={14} className="text-blue-400" />
                  <span>Active Layer: <strong className="text-blue-400 uppercase">{activeLayer}</strong></span>
                </div>
                <div className="flex space-x-2">
                  <button
                    onClick={() => setActiveLayer('imageA')}
                    disabled={!imageA}
                    className={`px-2.5 py-1 rounded transition cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed ${activeLayer === 'imageA' ? 'bg-blue-600/90 text-slate-950 font-medium' : 'bg-black/40 hover:bg-black/60 text-slate-200'}`}
                  >
                    Optical (A)
                  </button>
                  <button
                    onClick={() => setActiveLayer('imageB')}
                    disabled={!imageB || !showImageB}
                    className={`px-2.5 py-1 rounded transition cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed ${activeLayer === 'imageB' ? 'bg-blue-600/90 text-slate-950 font-medium' : 'bg-black/40 hover:bg-black/60 text-slate-200'}`}
                  >
                    SAR (B)
                  </button>
                  <button
                    onClick={() => setActiveLayer('evidence')}
                    disabled={!executionResult}
                    className={`px-2.5 py-1 rounded transition cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed ${activeLayer === 'evidence' ? 'bg-blue-600/90 text-slate-950 font-medium' : 'bg-black/40 hover:bg-black/60 text-slate-200'}`}
                  >
                    Fused + Mask
                  </button>
                </div>
              </div>
            </div>
          </main>

          {/* RIGHT: AI Assistant + Audit Trace - Set to exactly 50% width */}
          <section className="w-1/2 bg-black/50 backdrop-blur-md border-l border-white/10 flex flex-col h-full p-4 shrink-0">

            <div className="flex flex-col flex-1 min-h-0">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300 flex items-center shrink-0 mb-3">
                <MessageSquare size={14} className="mr-2 text-blue-400" /> AI Assistant & VQA
              </h3>

              <div className="bg-black/40 backdrop-blur-sm rounded-lg p-3 border border-white/10 flex-1 min-h-0 overflow-y-auto space-y-4 text-xs mb-3">
                
                {chatHistory.length === 0 && (
                  <div className="bg-black/30 backdrop-blur-sm p-3 rounded-lg border border-white/10 text-slate-300 flex items-start gap-3">
                    <Bot size={16} className="text-blue-400 mt-0.5 shrink-0" />
                    <p>System online. Please upload your imagery and ask a natural language query.</p>
                  </div>
                )}

                {chatHistory.map((msg, index) => (
                  <div key={index} className={`flex items-start gap-3 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
                    {msg.role === 'user' ? (
                       <User size={16} className="text-slate-400 mt-0.5 shrink-0" />
                    ) : (
                       <Bot size={16} className="text-blue-400 mt-0.5 shrink-0" />
                    )}
                    <div className={`p-3 rounded-lg max-w-[85%] ${msg.role === 'user' ? 'bg-slate-800/60 text-slate-200 border border-slate-700/50' : 'bg-blue-950/30 border border-blue-700/40 text-blue-100'}`}>
                      <p className="whitespace-pre-wrap leading-relaxed">{msg.content}</p>
                    </div>
                  </div>
                ))}

                {isExecuting && (
                  <div className="flex items-center space-x-2 text-blue-400 p-2">
                    <Loader2 size={14} className="animate-spin" />
                    <span className="italic">Processing pipeline...</span>
                  </div>
                )}

                {error && (
                  <div className="bg-red-950/30 backdrop-blur-sm border border-red-800/40 p-2.5 rounded-lg text-red-200">
                    <p className="font-semibold">Error:</p>
                    <p>{error}</p>
                  </div>
                )}
              </div>

              <form onSubmit={handleRunPipeline} className="relative shrink-0 mb-2">
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Ask about changes, land cover..."
                  disabled={isExecuting}
                  className="w-full bg-black/40 backdrop-blur-sm border border-white/20 rounded-lg pl-3 pr-10 py-2.5 text-xs text-slate-100 focus:outline-none focus:border-blue-500 disabled:opacity-50"
                />
                <button
                  type="submit"
                  disabled={isExecuting || !query}
                  className="absolute right-1.5 top-1/2 -translate-y-1/2 flex items-center justify-center w-7 h-7 rounded-md bg-blue-600/90 hover:bg-blue-500 transition cursor-pointer text-slate-950 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <Play size={13} fill="currentColor" />
                </button>
              </form>
            </div>

            <div className="pt-4 border-t border-white/10 shrink-0">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300 flex items-center mb-3">
                <ShieldCheck size={14} className="mr-2 text-blue-400" /> Auditable Trace
              </h3>

              <div className="bg-black/40 backdrop-blur-sm rounded-lg p-3 border border-white/10 text-[11px] space-y-1.5 text-slate-300 font-mono overflow-y-auto max-h-32 mb-3">
                {executionResult?.agent_execution_trace ? (
                  <>
                    <p className="text-blue-400 truncate">ID: {executionResult.agent_execution_trace.pipeline_id}</p>
                    <p className="opacity-80">Nodes: {executionResult.agent_execution_trace.nodes_traversed.join(" -> ")}</p>
                    {Object.entries(executionResult.agent_execution_trace.telemetry).map(([key, val]) => (
                      <p key={key} className="mt-1">{key}: <span className="text-slate-100">{val}</span></p>
                    ))}
                  </>
                ) : (
                  <p className="opacity-50">Awaiting execution telemetry...</p>
                )}
              </div>

              {executionResult?.report_download_url ? (
                <a 
                  href={`${BACKEND_URL}${executionResult.report_download_url}`}
                  download
                  className="w-full bg-blue-600/90 hover:bg-blue-500 backdrop-blur-sm text-slate-950 text-xs font-bold py-2.5 px-3 rounded-lg flex items-center justify-center space-x-2 transition border border-white/20 cursor-pointer"
                >
                  <Download size={14} />
                  <span>Download PDF Report</span>
                </a>
              ) : (
                <button disabled className="w-full bg-black/40 text-slate-500 text-xs font-medium py-2.5 px-3 rounded-lg flex items-center justify-center space-x-2 transition border border-white/20 cursor-not-allowed">
                  <Download size={14} />
                  <span>Report Pending</span>
                </button>
              )}
            </div>

          </section>

        </div>
      </div>
    </div>
  );
}