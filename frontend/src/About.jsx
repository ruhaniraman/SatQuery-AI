import React from 'react';
import { Link } from 'react-router-dom';

const About = () => {
  return (
    <div style={{ 
      position: 'relative', 
      width: '100vw', 
      minHeight: '100vh', 
      backgroundColor: '#010308', 
      color: '#ffffff',
      fontFamily: "'Inter', sans-serif",
      overflowX: 'hidden',
      overflowY: 'auto',
      boxSizing: 'border-box'
    }}>
      
      {/* Animated Space Background Layer */}
      <style>
        {`
          @keyframes spaceDrift {
            0% { background-position: 0% 0%; }
            50% { background-position: 100% 100%; }
            100% { background-position: 0% 0%; }
          }
        `}
      </style>
      <div style={{
        position: 'fixed',
        inset: 0,
        background: 'radial-gradient(circle at 20% 30%, rgba(59, 130, 246, 0.25) 0%, transparent 50%), radial-gradient(circle at 80% 80%, rgba(147, 197, 253, 0.15) 0%, transparent 50%), radial-gradient(circle at 50% 50%, rgba(30, 58, 138, 0.2) 0%, transparent 70%)',
        backgroundSize: '200% 200%',
        animation: 'spaceDrift 15s ease-in-out infinite',
        zIndex: 1,
        pointerEvents: 'none'
      }} />

      {/* Top Navigation Bar */}
      <nav style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100%',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        padding: '2.5rem 4rem',
        boxSizing: 'border-box',
        zIndex: 20,
        background: 'linear-gradient(to bottom, rgba(1,3,8,0.9) 20%, transparent)',
        backdropFilter: 'blur(4px)'
      }}>
        <div style={{
          fontSize: '0.75rem',
          fontWeight: 500,
          letterSpacing: '3px',
          color: '#64748b',
          textTransform: 'uppercase'
        }}>
          SIH 2026
        </div>

        <Link 
          to="/" 
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
          ← Home
        </Link>
      </nav>

      {/* Main Content Container - Expanded for Editorial Layout */}
      <div style={{
        position: 'relative',
        zIndex: 10,
        maxWidth: '1200px',
        margin: '0 auto',
        padding: '12rem 2rem 8rem 2rem',
        boxSizing: 'border-box'
      }}>
        
        {/* Left-Aligned Hero Section */}
        <div style={{ textAlign: 'left', marginBottom: '8rem', maxWidth: '800px' }}>
          <div style={{ 
            fontSize: '0.8rem', 
            textTransform: 'uppercase', 
            letterSpacing: '5px', 
            color: '#60a5fa', 
            marginBottom: '1.5rem', 
            fontWeight: 600,
          }}>
            ✦ Project Overview
          </div>
          <h1 style={{ 
            fontSize: 'clamp(3rem, 7vw, 6rem)', 
            fontWeight: 900, 
            letterSpacing: '-2px', 
            textTransform: 'uppercase',
            lineHeight: '1.1',
            margin: 0,
            color: '#ffffff'
          }}>
            Architecture <br/> & Vision
          </h1>
        </div>

        {/* Staggered Editorial Grid */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '6rem' }}>
          
          {/* Section 01 */}
          <div style={editorialRowStyle}>
            <div style={titleColStyle}>
              <div style={massiveNumberStyle}>01</div>
              <h2 style={chapterTitleStyle}>What is<br/>SatQuery AI?</h2>
            </div>
            <div style={contentColStyle}>
              <div style={cardStyle}
                onMouseOver={(e) => e.currentTarget.style.borderColor = 'rgba(96, 165, 250, 0.4)'}
                onMouseOut={(e) => e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.08)'}
              >
                <p style={cardTextStyle}>
                  SatQuery AI is an intelligent geospatial observation engine built for high-speed analysis of satellite data. It transforms complex orbital telemetry and multi-spectral raster inputs into clear, actionable intelligence accessible through fluid visual interfaces and automated query pipelines.
                </p>
              </div>
            </div>
          </div>

          {/* Section 02 */}
          <div style={{ ...editorialRowStyle, flexDirection: 'row-reverse' }}>
            <div style={titleColStyle}>
              <div style={massiveNumberStyle}>02</div>
              <h2 style={chapterTitleStyle}>The Core<br/>Problem</h2>
            </div>
            <div style={contentColStyle}>
              <div style={cardStyle}
                onMouseOver={(e) => e.currentTarget.style.borderColor = 'rgba(96, 165, 250, 0.4)'}
                onMouseOut={(e) => e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.08)'}
              >
                <p style={cardTextStyle}>
                  Traditional GIS processing is slow, resource-intensive, and trapped behind expert software, causing critical delays during emergency responses, environmental tracking, and infrastructure monitoring. SatQuery AI eliminates these bottlenecks by democratizing access to real-time Earth observation data.
                </p>
              </div>
            </div>
          </div>

          {/* Section 03 */}
          <div style={editorialRowStyle}>
            <div style={titleColStyle}>
              <div style={massiveNumberStyle}>03</div>
              <h2 style={chapterTitleStyle}>How We<br/>Solve It</h2>
            </div>
            <div style={contentColStyle}>
              <div style={cardStyle}
                onMouseOver={(e) => e.currentTarget.style.borderColor = 'rgba(96, 165, 250, 0.4)'}
                onMouseOut={(e) => e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.08)'}
              >
                <p style={cardTextStyle}>
                  By pairing high-performance 3D WebGL globe rendering with intelligent background computer vision models, SatQuery AI parses spatial changes on-demand. Users can execute targeted queries, track environmental shifts instantly, and monitor global assets without manual data aggregation.
                </p>
              </div>
            </div>
          </div>

        </div>

      </div>
    </div>
  );
};

// --- STYLES ---

const editorialRowStyle = {
  display: 'flex',
  flexWrap: 'wrap',
  gap: '4rem',
  alignItems: 'center',
  borderBottom: '1px solid rgba(255, 255, 255, 0.05)',
  paddingBottom: '6rem'
};

const titleColStyle = {
  flex: '1 1 300px',
  display: 'flex',
  flexDirection: 'column',
  gap: '1rem'
};

const massiveNumberStyle = {
  fontSize: 'clamp(5rem, 10vw, 8rem)',
  fontWeight: 900,
  lineHeight: '0.8',
  color: 'rgba(255, 255, 255, 0.05)',
  letterSpacing: '-4px'
};

const chapterTitleStyle = {
  fontSize: 'clamp(2rem, 4vw, 3rem)',
  fontWeight: 700,
  letterSpacing: '-1px',
  textTransform: 'uppercase',
  margin: 0,
  color: '#e2e8f0'
};

const contentColStyle = {
  flex: '1 1 500px'
};

const cardStyle = {
  background: 'rgba(15, 23, 42, 0.4)',
  backdropFilter: 'blur(20px)',
  WebkitBackdropFilter: 'blur(20px)',
  border: '1px solid rgba(255, 255, 255, 0.08)',
  borderRadius: '24px',
  padding: '3rem',
  boxShadow: '0 30px 60px rgba(0, 0, 0, 0.8), inset 0 1px 0 rgba(255, 255, 255, 0.1)',
  transition: 'border-color 0.4s ease',
};

const cardTextStyle = {
  fontSize: '1.1rem',
  lineHeight: '1.8',
  color: '#94a3b8',
  margin: 0,
  fontWeight: 400
};

export default About;