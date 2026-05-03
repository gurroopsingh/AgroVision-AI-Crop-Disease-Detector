import { Cpu, Database, Network, Code2, LineChart } from 'lucide-react';
import ModelInfo from '../components/ModelInfo';

export default function Transparency({ t }) {
  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      <div className="bg-white/80 backdrop-blur-xl p-8 rounded-3xl shadow-sm border border-white/60 mb-8">
        <div className="flex items-center gap-4 mb-2">
          <div className="bg-blue-100 p-3 rounded-2xl text-blue-600">
            <Cpu size={32} />
          </div>
          <div>
            <h1 className="text-3xl font-extrabold text-gray-900">{t.transparencyTitle}</h1>
            <p className="text-gray-500 font-medium mt-1">Deep dive into the AI powering AgroVision</p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-1">
          <ModelInfo t={t} />
        </div>

        <div className="lg:col-span-2 space-y-8">
          
          <div className="bg-white p-8 rounded-3xl shadow-sm border border-gray-100">
            <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center gap-3">
              <Network className="text-purple-500" /> {t.architecture || "Architecture"}
            </h2>
            <div className="space-y-4 text-gray-600 leading-relaxed">
              <p dangerouslySetInnerHTML={{ __html: t.transArchP1 }}></p>
              <p dangerouslySetInnerHTML={{ __html: t.transArchP2 }}></p>
            </div>
          </div>

          <div className="bg-white p-8 rounded-3xl shadow-sm border border-gray-100">
            <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center gap-3">
              <Database className="text-orange-500" /> {t.dataset || "Dataset & Preprocessing"}
            </h2>
            <div className="grid sm:grid-cols-2 gap-6 text-sm">
              <div className="bg-gray-50 p-4 rounded-2xl border border-gray-100">
                <h3 className="font-bold text-gray-900 mb-2 flex items-center gap-2">
                  <Code2 size={16} className="text-gray-400" /> {t.transDataPre}
                </h3>
                <ul className="list-disc list-inside space-y-1 text-gray-600">
                  <li>Resize to 224×224 pixels</li>
                  <li>ImageNet mean/std Normalization</li>
                  <li>RGB format conversion</li>
                </ul>
              </div>
              <div className="bg-gray-50 p-4 rounded-2xl border border-gray-100">
                <h3 className="font-bold text-gray-900 mb-2 flex items-center gap-2">
                  <LineChart size={16} className="text-gray-400" /> {t.transDataAug}
                </h3>
                <ul className="list-disc list-inside space-y-1 text-gray-600">
                  <li>Random Horizontal Flips (p=0.5)</li>
                  <li>Random Rotation (±15 degrees)</li>
                  <li>Color Jitter (brightness/contrast)</li>
                </ul>
              </div>
            </div>
            <p className="mt-6 text-gray-600 leading-relaxed" dangerouslySetInnerHTML={{ __html: t.transDataDesc }}></p>
          </div>

        </div>
      </div>
    </div>
  );
}
