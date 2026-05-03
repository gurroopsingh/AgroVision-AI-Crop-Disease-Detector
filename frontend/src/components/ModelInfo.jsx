import { History, Database, Layers, CheckCircle2, Cpu } from 'lucide-react';

export default function ModelInfo({ t }) {
  return (
    <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-100 flex flex-col h-full">
      <div className="flex items-center gap-2 mb-6 text-gray-800">
        <Database className="text-blue-500" />
        <h2 className="text-lg font-bold">{t.modelInformation}</h2>
      </div>
      
      <div className="space-y-4 flex-1">
        <div className="flex items-center justify-between p-3 bg-gray-50 rounded-xl">
          <div className="flex items-center gap-3 text-gray-600 font-medium">
            <Cpu size={18} /> Model
          </div>
          <span className="font-bold text-gray-900">ResNet18 Transfer Learning</span>
        </div>
        
        <div className="flex items-center justify-between p-3 bg-gray-50 rounded-xl">
          <div className="flex items-center gap-3 text-gray-600 font-medium">
            <Layers size={18} /> Classes
          </div>
          <span className="font-bold text-gray-900">86</span>
        </div>

        <div className="flex items-center justify-between p-3 bg-gray-50 rounded-xl">
          <div className="flex items-center gap-3 text-gray-600 font-medium">
            <Database size={18} /> Dataset Size
          </div>
          <span className="font-bold text-gray-900">150,000+ images</span>
        </div>

        <div className="flex items-center justify-between p-3 bg-gray-50 rounded-xl">
          <div className="flex items-center gap-3 text-gray-600 font-medium">
            <CheckCircle2 size={18} /> Framework
          </div>
          <span className="font-bold text-gray-900">PyTorch</span>
        </div>
        
        <div className="flex items-center justify-between p-3 bg-gray-50 rounded-xl">
          <div className="flex items-center gap-3 text-gray-600 font-medium">
            <CheckCircle2 size={18} /> Accuracy
          </div>
          <span className="font-bold text-green-600">86.17%</span>
        </div>
      </div>
    </div>
  );
}
