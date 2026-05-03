import { Info, Pill, ShieldAlert, Sprout } from 'lucide-react';

export default function PredictionResult({ result, t }) {
  const confidencePercent = (result.confidence * 100).toFixed(1);
  
  let badgeColor = "bg-red-500";
  let badgeText = t.lowConfidence;
  
  if (result.confidence > 0.9) {
    badgeColor = "bg-green-500";
    badgeText = t.highConfidence;
  } else if (result.confidence > 0.7) {
    badgeColor = "bg-yellow-500";
    badgeText = t.mediumConfidence;
  }

  return (
    <div className="bg-white p-6 rounded-2xl shadow-lg border border-green-100 h-full flex flex-col gap-6 animate-in fade-in zoom-in-95 duration-300">
      <div className="flex justify-between items-start flex-wrap gap-4">
        <div>
          <h2 className="text-sm font-bold tracking-wider text-green-600 uppercase mb-1">{t.predictionResult}</h2>
          <p className="text-2xl font-bold text-gray-900 capitalize">{result.predicted_class.replace(/_/g, ' ')}</p>
        </div>
        <div className={`${badgeColor} text-white px-3 py-1 rounded-full text-xs font-bold tracking-wider shadow-sm`}>
          {badgeText}
        </div>
      </div>

      <div>
        <div className="flex justify-between text-sm font-semibold mb-2">
          <span className="text-gray-600">{t.confidence}</span>
          <span className="text-gray-900">{confidencePercent}%</span>
        </div>
        <div className="h-3 w-full bg-gray-100 rounded-full overflow-hidden">
          <div 
            className="h-full bg-gradient-to-r from-green-400 to-green-600 transition-all duration-1000 ease-out rounded-full" 
            style={{ width: `${confidencePercent}%` }}
          ></div>
        </div>
      </div>

      <div className="grid gap-4 flex-1">
        <div className="bg-blue-50/50 p-4 rounded-xl border border-blue-100">
          <h3 className="flex items-center gap-2 font-semibold text-blue-800 mb-2">
            <Info size={18} /> {t.description}
          </h3>
          <p className="text-gray-700 text-sm leading-relaxed">{result.description}</p>
        </div>
        
        <div className="bg-red-50/50 p-4 rounded-xl border border-red-100">
          <h3 className="flex items-center gap-2 font-semibold text-red-800 mb-2">
            <Pill size={18} /> {t.treatment}
          </h3>
          <p className="text-gray-700 text-sm leading-relaxed">{result.treatment}</p>
        </div>

        <div className="bg-orange-50/50 p-4 rounded-xl border border-orange-100">
          <h3 className="flex items-center gap-2 font-semibold text-orange-800 mb-2">
            <ShieldAlert size={18} /> {t.prevention}
          </h3>
          <p className="text-gray-700 text-sm leading-relaxed">{result.prevention}</p>
        </div>

        <div className="bg-emerald-50/50 p-4 rounded-xl border border-emerald-100">
          <h3 className="flex items-center gap-2 font-semibold text-emerald-800 mb-2">
            <Sprout size={18} /> {t.fertilizer}
          </h3>
          <p className="text-gray-700 text-sm leading-relaxed">{result.fertilizer}</p>
        </div>
      </div>
    </div>
  );
}
