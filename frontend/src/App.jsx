import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';

// Import your two pages (make sure the file paths match where they are saved!)
import SatQueryFrontend from './SatQueryFrontend';
import Dashboard from './Dashboard';
import About from './About'; 


function App() {
  return (
    <Router>
      <Routes>
        {/* When the URL is exactly "/", load the 3D Globe home page */}
        <Route path="/" element={<SatQueryFrontend />} />
        
        {/* When the URL is "/about", load the glass slab About page */}
        <Route path="/about" element={<About />} />

        <Route path="/dashboard" element={<Dashboard />} />
      </Routes>
    </Router>
  );
}

export default App;