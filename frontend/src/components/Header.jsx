import { Leaf } from 'lucide-react';

export default function Header({ lang, setLang, t }) {
  return (
    <header className="bg-white/80 backdrop-blur-md sticky top-0 z-50 border-b border-green-100 shadow-sm">
      <div className="max-w-6xl mx-auto px-4 py-4 flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="bg-gradient-to-br from-green-500 to-green-600 p-2 rounded-xl text-white shadow-md shadow-green-200">
            <Leaf size={28} />
          </div>
          <div>
            <h1 className="text-xl font-bold text-gray-900 leading-tight">{t.title}</h1>
            <p className="text-sm text-gray-500 font-medium">{t.subtitle}</p>
          </div>
        </div>
        
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 bg-green-50 text-green-700 px-3 py-1.5 rounded-full border border-green-200 shadow-inner text-sm font-semibold">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
            {t.modelReady}
          </div>
          
          <div className="flex bg-gray-100 p-1 rounded-lg">
            <button 
              onClick={() => setLang('en')}
              className={`px-3 py-1 text-sm font-medium rounded-md transition-all ${lang === 'en' ? 'bg-white shadow text-green-600' : 'text-gray-500 hover:text-gray-700'}`}
            >
              EN
            </button>
            <button 
              onClick={() => setLang('hi')}
              className={`px-3 py-1 text-sm font-medium rounded-md transition-all ${lang === 'hi' ? 'bg-white shadow text-green-600' : 'text-gray-500 hover:text-gray-700'}`}
            >
              HI
            </button>
          </div>
        </div>
      </div>
    </header>
  );
}
