import { History } from 'lucide-react';

export default function PredictionHistory({ history, t }) {
  return (
    <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-100 flex flex-col h-full">
      <div className="flex items-center gap-2 mb-6 text-gray-800">
        <History className="text-purple-500" />
        <h2 className="text-lg font-bold">{t.predictionHistory}</h2>
      </div>

      <div className="flex-1 overflow-y-auto pr-2">
        {history.length === 0 ? (
          <div className="h-full flex items-center justify-center text-gray-400 italic">
            {t.noHistory}
          </div>
        ) : (
          <div className="space-y-3">
            {history.map((item, idx) => (
              <div key={idx} className="flex items-center gap-4 p-3 rounded-xl border border-gray-100 hover:bg-gray-50 transition-colors">
                <div className="w-16 h-16 rounded-lg overflow-hidden bg-gray-100 shrink-0 border border-gray-200">
                  <img src={item.preview} alt="History thumbnail" className="w-full h-full object-cover" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="font-bold text-gray-900 capitalize truncate">
                    {item.data.predicted_class.replace(/_/g, ' ')}
                  </p>
                  <div className="flex items-center gap-2 mt-1">
                    <div className="h-1.5 w-16 bg-gray-200 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-green-500 rounded-full" 
                        style={{ width: `${item.data.confidence * 100}%` }}
                      ></div>
                    </div>
                    <span className="text-xs font-semibold text-gray-500">
                      {(item.data.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
