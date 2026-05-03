import { BrowserRouter, Routes, Route, Link, useLocation } from 'react-router-dom';
import { useState, useEffect } from 'react';
import { Leaf } from 'lucide-react';
import { translations } from './translations';

import Home from './pages/Home';
import Encyclopedia from './pages/Encyclopedia';
import Transparency from './pages/Transparency';

function Header({ lang, setLang, t }) {
  const location = useLocation();
  
  const navLinks = [
    { path: '/', label: t.navHome },
    { path: '/encyclopedia', label: t.navEncyclopedia },
    { path: '/transparency', label: t.navTransparency },
  ];

  return (
    <header className="bg-white/70 backdrop-blur-xl sticky top-0 z-50 border-b border-green-100 shadow-sm">
      <div className="max-w-6xl mx-auto px-4 py-4 flex flex-col md:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="bg-gradient-to-br from-green-500 to-green-600 p-2 rounded-xl text-white shadow-md shadow-green-200">
            <Leaf size={28} />
          </div>
          <div>
            <h1 className="text-xl font-bold text-gray-900 leading-tight">{t.title}</h1>
            <p className="text-sm text-gray-500 font-medium">{t.subtitle}</p>
          </div>
        </div>
        
        <nav className="flex items-center gap-2 md:gap-6 bg-white/50 p-1 md:p-0 rounded-full md:bg-transparent shadow-sm md:shadow-none border border-gray-100 md:border-none">
          {navLinks.map((link) => (
            <Link 
              key={link.path} 
              to={link.path}
              className={`px-4 py-2 rounded-full text-sm font-semibold transition-all ${
                location.pathname === link.path 
                  ? 'bg-green-100 text-green-700 shadow-inner' 
                  : 'text-gray-600 hover:bg-green-50 hover:text-green-600'
              }`}
            >
              {link.label}
            </Link>
          ))}
        </nav>

        <div className="flex items-center gap-4">
          <div className="hidden lg:flex items-center gap-2 bg-green-50 text-green-700 px-3 py-1.5 rounded-full border border-green-200 shadow-inner text-sm font-semibold">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
            {t.modelReady}
          </div>
          
          <div className="flex bg-gray-100/80 backdrop-blur p-1 rounded-lg border border-gray-200 shadow-inner">
            <button 
              onClick={() => setLang('en')}
              className={`px-3 py-1 text-sm font-bold rounded-md transition-all ${lang === 'en' ? 'bg-white shadow text-green-600' : 'text-gray-500 hover:text-gray-700'}`}
            >
              EN
            </button>
            <button 
              onClick={() => setLang('hi')}
              className={`px-3 py-1 text-sm font-bold rounded-md transition-all ${lang === 'hi' ? 'bg-white shadow text-green-600' : 'text-gray-500 hover:text-gray-700'}`}
            >
              HI
            </button>
          </div>
        </div>
      </div>
    </header>
  );
}

function App() {
  const [lang, setLang] = useState(localStorage.getItem('agrovision_lang') || 'en');
  const t = translations[lang];

  useEffect(() => {
    localStorage.setItem('agrovision_lang', lang);
  }, [lang]);

  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gradient-to-br from-green-50 via-teal-50 to-blue-50 text-gray-800 font-sans selection:bg-green-200 selection:text-green-900">
        <Header lang={lang} setLang={setLang} t={t} />
        <Routes>
          <Route path="/" element={<Home t={t} lang={lang} />} />
          <Route path="/encyclopedia" element={<Encyclopedia t={t} lang={lang} />} />
          <Route path="/transparency" element={<Transparency t={t} />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}

export default App;
