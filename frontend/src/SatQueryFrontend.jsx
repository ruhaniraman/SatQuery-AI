import React, { useEffect, useRef } from 'react';
import { setOptions, importLibrary } from '@googlemaps/js-api-loader';

const SatQueryFrontend = () => {
  const mapContainerRef = useRef(null);
  const map3DRef = useRef(null);
  const requestRef = useRef();
  const headingValue = useRef(0);
  const aboutRef = useRef(null);

  useEffect(() => {
    let isMounted = true;

    const linkId = 'google-fonts-outfit';
    if (!document.getElementById(linkId)) {
      const link = document.createElement('link');
      link.id = linkId;
      link.rel = 'stylesheet';
      link.href = 'https://fonts.googleapis.com/css2?family=Outfit:wght@800;900&family=Inter:wght@400;500;600&display=swap';
      document.head.appendChild(link);
    }

    const initMap = async () => {
      try {
        const apiKey = import.meta.env.VITE_GOOGLE_MAPS_API_KEY || process.env.REACT_APP_GOOGLE_MAPS_API_KEY;

        setOptions({
          key: apiKey,
          v: "beta",
        });

        const { Map3DElement } = await importLibrary("maps3d");

        if (!isMounted) return;

        const map3D = new Map3DElement({
          mode: "SATELLITE",
          center: { lat: 15.0, lng: 40.0, altitude: 0 },
          range: 7000000,
          tilt: 1000,
          heading: 0,
          defaultUIHidden: true,
        });

        map3DRef.current = map3D;
        
        if (mapContainerRef.current) {
          mapContainerRef.current.innerHTML = "";
          mapContainerRef.current.appendChild(map3D);
        }

        const animate = () => {
          if (map3DRef.current) {
            headingValue.current = (headingValue.current + 0.03) % 360;
            map3DRef.current.heading = headingValue.current;
          }
          requestRef.current = requestAnimationFrame(animate);
        };
        
        requestRef.current = requestAnimationFrame(animate);

      } catch (error) {
        console.error("Error loading Google Maps 3D:", error);
      }
    };

    initMap();

    return () => {
      isMounted = false;
      if (requestRef.current) {
        cancelAnimationFrame(requestRef.current);
      }
    };
  }, []);

  const scrollToAbout = (e) => {
    e.preventDefault();
    if (aboutRef.current) {
      aboutRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  };

  const scrollToTop = (e) => {
    e.preventDefault();
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <div style={{ position: 'relative', width: '100vw', overflowX: 'hidden', backgroundColor: '#030712', color: '#fff' }}>
      
      {/* ================= SECTION 1: HOME (Hero + Globe) ================= */}
      <div style={{ position: 'relative', width: '100vw', height: '100vh', overflow: 'hidden' }}>
        
        {/* Background 3D Globe Container */}
        <div 
          ref={mapContainerRef} 
          style={{ width: '100%', height: '100%', position: 'absolute', top: 0, left: 0, zIndex: 1 }} 
        />

        {/* Cinematic Vignette / Lighting Overlay for Depth */}
        <div style={{
          position: 'absolute',
          inset: 0,
          background: 'radial-gradient(circle at 50% 30%, rgba(59, 130, 246, 0.12) 0%, rgba(3, 7, 18, 0.7) 80%)',
          pointerEvents: 'none',
          zIndex: 2
        }} />

        {/* Top Navigation Bar - Pushed to corners and softened */}
        <nav style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          padding: '2.5rem 4rem',
          boxSizing: 'border-box',
          zIndex: 20,
          fontFamily: "'Inter', sans-serif"
        }}>
          {/* Left Text */}
          <div 
            onClick={scrollToTop}
            style={{
              fontSize: '0.75rem',
              fontWeight: 500,
              letterSpacing: '3px',
              color: '#64748b',
              textTransform: 'uppercase',
              cursor: 'pointer'
            }}
          >
            SIH 2026
          </div>

          {/* Right About Link */}
          <a 
            href="#about" 
            onClick={scrollToAbout}
            style={{
              fontSize: '0.75rem',
              fontWeight: 500,
              letterSpacing: '3px',
              color: '#64748b',
              textDecoration: 'none',
              textTransform: 'uppercase',
              transition: 'color 0.2s ease',
              cursor: 'pointer'
            }}
            onMouseOver={(e) => e.currentTarget.style.color = '#ffffff'}
            onMouseOut={(e) => e.currentTarget.style.color = '#64748b'}
          >
            About
          </a>
        </nav>
        
        {/* Cinematic Poster Typography Layer */}
        <div style={{
          position: 'absolute',
          top: '40%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          zIndex: 10,
          fontFamily: "'Inter', sans-serif",
          color: '#fff',
          textAlign: 'center',
          width: '100%',
          padding: '0 1rem',
          boxSizing: 'border-box',
          pointerEvents: 'none'
        }}>
          <div style={{ 
            fontSize: '0.8rem', 
            textTransform: 'uppercase', 
            letterSpacing: '5px', 
            color: '#60a5fa', 
            marginBottom: '1.2rem', 
            fontWeight: 600,
            textShadow: '0 0 15px rgba(96, 165, 250, 0.6)',
            pointerEvents: 'auto'
          }}>
            ✦ Intelligent Earth Observation
          </div>

          <h1 style={{ 
            margin: 0, 
            fontFamily: "'Outfit', sans-serif",
            fontSize: 'clamp(3.5rem, 10vw, 10rem)', 
            fontWeight: 900, 
            letterSpacing: '-1.5px', 
            lineHeight: '1',
            textTransform: 'uppercase',
            whiteSpace: 'nowrap',
            color: '#ffffff',
            textShadow: '0 2px 15px rgba(0, 0, 0, 0.9), 0 0 10px rgba(255, 255, 255, 0.5)',
            pointerEvents: 'auto'
          }}>
            SATQUERY AI
          </h1>

          <div style={{
            display: 'inline-flex',
            alignItems: 'center',
            justifyContent: 'center',
            marginTop: '1.5rem',
            padding: '12px 32px',
            background: 'rgba(15, 23, 42, 0.85)',
            backdropFilter: 'blur(12px)',
            border: '1px solid rgba(59, 130, 246, 0.5)',
            borderRadius: '40px',
            color: '#ffffff',
            fontSize: '0.9rem',
            letterSpacing: '3px',
            fontWeight: 600,
            boxShadow: '0 10px 30px rgba(0,0,0,0.7), 0 0 25px rgba(59, 130, 246, 0.25)',
            pointerEvents: 'auto',
            cursor: 'pointer',
            transition: 'all 0.2s ease'
          }}
          onMouseOver={(e) => {
            e.currentTarget.style.transform = 'scale(1.04)';
            e.currentTarget.style.borderColor = 'rgba(59, 130, 246, 0.9)';
            e.currentTarget.style.boxShadow = '0 10px 30px rgba(0,0,0,0.7), 0 0 35px rgba(59, 130, 246, 0.5)';
          }}
          onMouseOut={(e) => {
            e.currentTarget.style.transform = 'scale(1)';
            e.currentTarget.style.borderColor = 'rgba(59, 130, 246, 0.5)';
            e.currentTarget.style.boxShadow = '0 10px 30px rgba(0,0,0,0.7), 0 0 25px rgba(59, 130, 246, 0.25)';
          }}
          >
            ACTIVE SYSTEM
          </div>
        </div>

      </div>

      {/* ================= SECTION 2: ABOUT PAGE CONTENT ================= */}
      <div ref={aboutRef} style={{
        position: 'relative',
        zIndex: 10,
        maxWidth: '900px',
        margin: '0 auto',
        padding: '8rem 2rem 6rem 2rem',
        boxSizing: 'border-box',
        fontFamily: "'Inter', sans-serif"
      }}>
        
        <div style={{ textAlign: 'center', marginBottom: '4rem' }}>
          <div style={{ 
            fontSize: '0.8rem', 
            textTransform: 'uppercase', 
            letterSpacing: '5px', 
            color: '#60a5fa', 
            marginBottom: '1rem', 
            fontWeight: 600,
            textShadow: '0 0 15px rgba(96, 165, 250, 0.6)'
          }}>
            ✦ About The Project
          </div>
          <h2 style={{ 
            fontSize: 'clamp(2.2rem, 5vw, 4rem)', 
            fontWeight: 900, 
            letterSpacing: '-1px', 
            textTransform: 'uppercase',
            margin: 0,
            color: '#ffffff'
          }}>
            Architecture & Vision
          </h2>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
          <div style={{
            background: 'rgba(15, 23, 42, 0.75)',
            backdropFilter: 'blur(12px)',
            border: '1px solid rgba(59, 130, 246, 0.3)',
            borderRadius: '16px',
            padding: '2.5rem',
            boxShadow: '0 20px 40px rgba(0,0,0,0.6)'
          }}>
            <h3 style={{ fontSize: '1.1rem', fontWeight: 700, letterSpacing: '2px', color: '#60a5fa', textTransform: 'uppercase', marginTop: 0, marginBottom: '1rem' }}>01 / What is SatQuery AI?</h3>
            <p style={{ fontSize: '1rem', lineHeight: '1.7', color: '#94a3b8', margin: 0 }}>
              SatQuery AI is an intelligent geospatial observation engine built for high-speed analysis of satellite data. It transforms complex orbital telemetry and multi-spectral raster inputs into clear, actionable intelligence accessible through fluid visual interfaces and automated query pipelines.
            </p>
          </div>

          <div style={{
            background: 'rgba(15, 23, 42, 0.75)',
            backdropFilter: 'blur(12px)',
            border: '1px solid rgba(59, 130, 246, 0.3)',
            borderRadius: '16px',
            padding: '2.5rem',
            boxShadow: '0 20px 40px rgba(0,0,0,0.6)'
          }}>
            <h3 style={{ fontSize: '1.1rem', fontWeight: 700, letterSpacing: '2px', color: '#60a5fa', textTransform: 'uppercase', marginTop: 0, marginBottom: '1rem' }}>02 / What Problem Does It Solve?</h3>
            <p style={{ fontSize: '1rem', lineHeight: '1.7', color: '#94a3b8', margin: 0 }}>
              Traditional GIS processing is slow, resource-intensive, and trapped behind expert software, causing critical delays during emergency responses, environmental tracking, and infrastructure monitoring. SatQuery AI eliminates these bottlenecks by democratizing access to real-time Earth observation data.
            </p>
          </div>

          <div style={{
            background: 'rgba(15, 23, 42, 0.75)',
            backdropFilter: 'blur(12px)',
            border: '1px solid rgba(59, 130, 246, 0.3)',
            borderRadius: '16px',
            padding: '2.5rem',
            boxShadow: '0 20px 40px rgba(0,0,0,0.6)'
          }}>
            <h3 style={{ fontSize: '1.1rem', fontWeight: 700, letterSpacing: '2px', color: '#60a5fa', textTransform: 'uppercase', marginTop: 0, marginBottom: '1rem' }}>03 / How Does It Solve It?</h3>
            <p style={{ fontSize: '1rem', lineHeight: '1.7', color: '#94a3b8', margin: 0 }}>
              By pairing high-performance 3D WebGL globe rendering with intelligent background computer vision models, SatQuery AI parses spatial changes on-demand. Users can execute targeted queries, track environmental shifts instantly, and monitor global assets without manual data aggregation.
            </p>
          </div>
        </div>

      </div>
      
    </div>
  );
};

export default SatQueryFrontend;