import React, { useEffect, useRef } from 'react';
import { setOptions, importLibrary } from '@googlemaps/js-api-loader';
import { Link } from 'react-router-dom';
import { useNavigate } from 'react-router-dom';

const SatQueryFrontend = () => {
  const navigate = useNavigate();
  const mapContainerRef = useRef(null);
  const map3DRef = useRef(null);
  const requestRef = useRef();
  const headingValue = useRef(0);

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

  const scrollToTop = (e) => {
    e.preventDefault();
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <div style={{ position: 'relative', width: '100vw', height: '100vh', overflow: 'hidden', backgroundColor: '#030712', color: '#fff' }}>
      
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

      {/* Top Navigation Bar */}
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

        <Link 
          to="/about" 
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
        </Link>
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

        <div 
          onClick={() => navigate('/dashboard')}
          style={{
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
        }}>
          ACTIVE SYSTEM
        </div>
      </div>

    </div>
  );
};

export default SatQueryFrontend;