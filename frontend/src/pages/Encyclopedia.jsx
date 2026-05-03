import { useState, useEffect } from 'react';
import { Search, BookOpen, AlertCircle } from 'lucide-react';

export default function Encyclopedia({ t, lang }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');

  useEffect(() => {
    setLoading(true);
    fetch(`/disease-info?lang=${lang}`)
      .then(res => res.json())
      .then(d => {
        setData(d);
        setLoading(false);
      })
      .catch(err => {
        console.error(err);
        setError('Failed to load encyclopedia data.');
        setLoading(false);
      });
  }, [lang]);

  if (loading) {
    return (
      <div className="max-w-6xl mx-auto px-4 py-12 flex justify-center items-center min-h-[60vh]">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-green-600"></div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="max-w-6xl mx-auto px-4 py-12 flex justify-center items-center min-h-[60vh]">
        <div className="bg-red-50 text-red-600 p-6 rounded-2xl flex items-center gap-3 shadow-sm border border-red-100">
          <AlertCircle />
          <span className="font-semibold">{error}</span>
        </div>
      </div>
    );
  }

  const { classes, disease_info } = data;

  const filteredClasses = classes.filter(c => 
    c.toLowerCase().includes(searchTerm.toLowerCase()) || 
    (disease_info[c]?.description || '').toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      <div className="bg-white/80 backdrop-blur-xl p-8 rounded-3xl shadow-sm border border-white/60 mb-8">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-6 mb-8">
          <div className="flex items-center gap-4">
            <div className="bg-green-100 p-3 rounded-2xl text-green-600">
              <BookOpen size={32} />
            </div>
            <div>
              <h1 className="text-3xl font-extrabold text-gray-900">{t.encyclopediaTitle}</h1>
              <p className="text-gray-500 font-medium mt-1">Explore {classes.length} distinct crop health categories</p>
            </div>
          </div>
          
          <div className="relative w-full md:w-72">
            <input 
              type="text" 
              placeholder={t.searchPlaceholder || "Search..."}
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-12 pr-4 py-3 rounded-xl border border-gray-200 bg-gray-50 focus:bg-white focus:ring-2 focus:ring-green-500 focus:border-transparent outline-none transition-all shadow-inner"
            />
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-400" size={20} />
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
        {filteredClasses.map(className => {
          const info = disease_info[className];
          if (!info) return null;
          
          const isHealthy = className.startsWith('healthy_');
          const title = className.replace(/_/g, ' ');

          return (
            <div key={className} className="bg-white/90 backdrop-blur-sm p-6 rounded-2xl shadow-sm border border-gray-100 hover:shadow-lg hover:-translate-y-1 transition-all duration-300 flex flex-col group">
              <div className="flex justify-between items-start mb-4">
                <h2 className="text-xl font-bold capitalize text-gray-800 group-hover:text-green-600 transition-colors line-clamp-1">{title}</h2>
                {isHealthy ? (
                  <span className="bg-green-100 text-green-700 px-2 py-1 rounded-md text-xs font-bold uppercase tracking-wider">Healthy</span>
                ) : (
                  <span className="bg-orange-100 text-orange-700 px-2 py-1 rounded-md text-xs font-bold uppercase tracking-wider">Disease</span>
                )}
              </div>
              
              <div className="flex-1 space-y-4">
                <p className="text-sm text-gray-600 line-clamp-3">{info.description}</p>
                <div className="pt-4 border-t border-gray-100">
                  <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-1">{t.treatment}</p>
                  <p className="text-sm text-gray-700 line-clamp-2">{info.treatment}</p>
                </div>
              </div>
            </div>
          );
        })}
      </div>
      
      {filteredClasses.length === 0 && (
        <div className="text-center py-20 text-gray-500 font-medium text-lg">
          No matches found for "{searchTerm}"
        </div>
      )}
    </div>
  );
}
