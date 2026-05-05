import { Info, Pill, ShieldAlert, Sprout, Sparkles, List, CheckSquare } from 'lucide-react';

/**
 * Render a text block that may contain numbered lists (1. 2. 3.)
 * or bullet lists (- • *) from Gemini's output.
 */
function RichText({ text }) {
  if (!text) return null;

  // Split into lines and detect list patterns
  const lines = text.split('\n').filter(l => l.trim() !== '');

  const isNumbered = (line) => /^\s*\d+[\.\)]\s/.test(line);
  const isBullet   = (line) => /^\s*[-•*]\s/.test(line);

  // If the whole block is a list, render as a list
  const allNumbered = lines.every(isNumbered);
  const allBullets  = lines.every(isBullet);

  if (allNumbered) {
    return (
      <ol className="space-y-2 list-none">
        {lines.map((line, i) => {
          const content = line.replace(/^\s*\d+[\.\)]\s*/, '');
          return (
            <li key={i} className="flex gap-3 text-sm text-gray-700 leading-relaxed">
              <span className="flex-shrink-0 w-6 h-6 bg-current/10 rounded-full flex items-center justify-center text-xs font-bold">
                {i + 1}
              </span>
              <span>{content}</span>
            </li>
          );
        })}
      </ol>
    );
  }

  if (allBullets) {
    return (
      <ul className="space-y-1.5 list-none">
        {lines.map((line, i) => {
          const content = line.replace(/^\s*[-•*]\s*/, '');
          return (
            <li key={i} className="flex gap-2 text-sm text-gray-700 leading-relaxed">
              <span className="flex-shrink-0 mt-1 w-2 h-2 rounded-full bg-current opacity-60" />
              <span>{content}</span>
            </li>
          );
        })}
      </ul>
    );
  }

  // Mixed or plain text – render paragraph(s)
  return (
    <div className="space-y-2">
      {lines.map((line, i) => {
        if (isNumbered(line)) {
          const num     = line.match(/^\s*(\d+)/)[1];
          const content = line.replace(/^\s*\d+[\.\)]\s*/, '');
          return (
            <div key={i} className="flex gap-3 text-sm text-gray-700 leading-relaxed">
              <span className="flex-shrink-0 font-bold opacity-60">{num}.</span>
              <span>{content}</span>
            </div>
          );
        }
        if (isBullet(line)) {
          const content = line.replace(/^\s*[-•*]\s*/, '');
          return (
            <div key={i} className="flex gap-2 text-sm text-gray-700 leading-relaxed">
              <span className="flex-shrink-0 mt-1 w-2 h-2 rounded-full bg-current opacity-60" />
              <span>{content}</span>
            </div>
          );
        }
        return (
          <p key={i} className="text-sm text-gray-700 leading-relaxed">{line}</p>
        );
      })}
    </div>
  );
}

function InfoCard({ bgClass, borderClass, titleClass, icon: Icon, title, content }) {
  return (
    <div className={`${bgClass} p-4 rounded-xl border ${borderClass}`}>
      <h3 className={`flex items-center gap-2 font-semibold ${titleClass} mb-3`}>
        <Icon size={18} /> {title}
      </h3>
      <div className={titleClass}>
        <RichText text={content} />
      </div>
    </div>
  );
}

export default function PredictionResult({ result, t }) {
  const confidencePercent = (result.confidence * 100).toFixed(1);
  const isAI = result.ai_generated === true;

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
    <div className="bg-white p-6 rounded-2xl shadow-lg border border-green-100 h-full flex flex-col gap-5 animate-in fade-in zoom-in-95 duration-300">
      {/* Header */}
      <div className="flex justify-between items-start flex-wrap gap-3">
        <div>
          <h2 className="text-sm font-bold tracking-wider text-green-600 uppercase mb-1">{t.predictionResult}</h2>
          <p className="text-2xl font-bold text-gray-900 capitalize">{result.predicted_class.replace(/_/g, ' ')}</p>
        </div>
        <div className="flex flex-col items-end gap-2">
          <div className={`${badgeColor} text-white px-3 py-1 rounded-full text-xs font-bold tracking-wider shadow-sm`}>
            {badgeText}
          </div>
          {isAI && (
            <div className="flex items-center gap-1.5 bg-gradient-to-r from-violet-500 to-indigo-500 text-white px-3 py-1 rounded-full text-xs font-bold shadow-sm">
              <Sparkles size={11} />
              Gemini AI
            </div>
          )}
        </div>
      </div>

      {/* Confidence bar */}
      <div>
        <div className="flex justify-between text-sm font-semibold mb-2">
          <span className="text-gray-600">{t.confidence}</span>
          <span className="text-gray-900">{confidencePercent}%</span>
        </div>
        <div className="h-3 w-full bg-gray-100 rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-green-400 to-green-600 transition-all duration-1000 ease-out rounded-full"
            style={{ width: `${confidencePercent}%` }}
          />
        </div>
      </div>

      {/* Info cards */}
      <div className="grid gap-4 flex-1">
        <InfoCard
          bgClass="bg-blue-50/50"
          borderClass="border-blue-100"
          titleClass="text-blue-800"
          icon={Info}
          title={t.description}
          content={result.description}
        />
        <InfoCard
          bgClass="bg-red-50/50"
          borderClass="border-red-100"
          titleClass="text-red-800"
          icon={Pill}
          title={t.treatment}
          content={result.treatment}
        />
        <InfoCard
          bgClass="bg-orange-50/50"
          borderClass="border-orange-100"
          titleClass="text-orange-800"
          icon={ShieldAlert}
          title={t.prevention}
          content={result.prevention}
        />
        <InfoCard
          bgClass="bg-emerald-50/50"
          borderClass="border-emerald-100"
          titleClass="text-emerald-800"
          icon={Sprout}
          title={t.fertilizer}
          content={result.fertilizer}
        />
      </div>

      {/* AI attribution footer */}
      {isAI && (
        <p className="text-center text-xs text-gray-400 flex items-center justify-center gap-1.5">
          <Sparkles size={11} className="text-violet-400" />
          Detailed analysis generated by Google Gemini AI
        </p>
      )}
    </div>
  );
}
