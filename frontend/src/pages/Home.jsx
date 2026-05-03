import { useState, useRef } from 'react';
import { Camera, Upload, RefreshCw, CheckCircle, Image as ImageIcon } from 'lucide-react';
import PredictionResult from '../components/PredictionResult';
import PredictionHistory from '../components/PredictionHistory';

export default function Home({ t, lang }) {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [error, setError] = useState(null);
  
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const canvasRef = useRef(null);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setIsCameraActive(true);
      setError(null);
    } catch (err) {
      console.error(err);
      setIsCameraActive(false);
      setError("Camera unavailable");
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    setIsCameraActive(false);
  };

  const captureImage = () => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      
      canvas.toBlob((blob) => {
        const file = new File([blob], "capture.jpg", { type: "image/jpeg" });
        setFile(file);
        setPreview(URL.createObjectURL(blob));
        stopCamera();
      }, 'image/jpeg');
    }
  };

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (selected) {
      setFile(selected);
      setPreview(URL.createObjectURL(selected));
      setResult(null);
      setError(null);
      if (isCameraActive) stopCamera();
    }
  };

  const handlePredict = async () => {
    if (!file) return;
    
    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`/predict?lang=${lang}`, {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || t.error);
      
      setResult(data);
      setHistory(prev => {
        const newEntry = { preview, data };
        return [newEntry, ...prev].slice(0, 5);
      });
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <main className="max-w-6xl mx-auto px-4 py-8 grid grid-cols-1 lg:grid-cols-2 gap-8 relative">
        {/* Left Panel: Scanner */}
        <div className="flex flex-col gap-6">
          <div className="bg-white/80 backdrop-blur-xl p-8 rounded-3xl shadow-xl border border-white/40 flex flex-col items-center hover:shadow-2xl transition-all duration-300">
            
            {!preview && !isCameraActive && (
              <div className="w-full h-72 border-2 border-dashed border-green-300 rounded-2xl flex flex-col items-center justify-center bg-green-50/30 hover:bg-green-50 transition-colors relative cursor-pointer group">
                <input type="file" accept="image/*" onChange={handleFileChange} className="absolute inset-0 opacity-0 cursor-pointer w-full h-full z-10" />
                <div className="bg-white p-4 rounded-full shadow-sm mb-4 group-hover:scale-110 transition-transform">
                  <Upload size={32} className="text-green-500" />
                </div>
                <p className="text-gray-600 font-semibold tracking-wide">{t.dragDrop}</p>
              </div>
            )}

            {isCameraActive && (
              <div className="w-full h-72 bg-black rounded-2xl overflow-hidden relative shadow-inner">
                <video ref={videoRef} autoPlay playsInline className="w-full h-full object-cover opacity-90" />
                {/* Scanning overlay effect */}
                <div className="absolute inset-0 border-2 border-green-500/50 rounded-2xl pointer-events-none"></div>
                <button onClick={captureImage} className="absolute bottom-6 left-1/2 -translate-x-1/2 bg-white text-green-700 px-8 py-3 rounded-full font-bold shadow-lg hover:scale-105 transition-transform">
                  {t.captureImage}
                </button>
              </div>
            )}

            {preview && !isCameraActive && (
              <div className="w-full h-72 relative rounded-2xl overflow-hidden shadow-inner group">
                <img src={preview} alt="Preview" className="w-full h-full object-cover" />
                <div className="absolute inset-0 bg-black/10 group-hover:bg-transparent transition-colors"></div>
                <button onClick={() => { setPreview(null); setFile(null); setResult(null); }} className="absolute top-4 right-4 bg-white/90 p-3 rounded-full hover:bg-white text-red-500 shadow-lg hover:scale-110 transition-transform">
                  <RefreshCw size={20} />
                </button>
              </div>
            )}

            <canvas ref={canvasRef} className="hidden" />

            {!isCameraActive && !preview && (
              <div className="mt-6 flex items-center gap-4 w-full">
                <div className="h-px bg-gradient-to-r from-transparent via-gray-300 to-transparent flex-1"></div>
                <span className="text-gray-400 text-sm font-bold tracking-widest uppercase">{t.or}</span>
                <div className="h-px bg-gradient-to-r from-transparent via-gray-300 to-transparent flex-1"></div>
              </div>
            )}

            {!isCameraActive && !preview && (
              <button onClick={startCamera} className="mt-6 w-full flex items-center justify-center gap-3 bg-white border-2 border-green-400 text-green-700 font-bold py-4 rounded-2xl shadow-sm hover:bg-green-50 hover:border-green-500 transition-all">
                <Camera size={24} />
                {t.openCamera}
              </button>
            )}

            {preview && (
              <button 
                onClick={handlePredict} 
                disabled={loading}
                className="mt-8 w-full flex items-center justify-center gap-3 bg-gradient-to-r from-green-500 to-emerald-600 text-white font-bold py-4 rounded-2xl shadow-[0_8px_30px_rgb(16,185,129,0.3)] hover:shadow-[0_8px_30px_rgb(16,185,129,0.5)] hover:-translate-y-1 transition-all disabled:opacity-70 disabled:cursor-not-allowed disabled:transform-none">
                {loading ? <RefreshCw className="animate-spin" size={24} /> : <CheckCircle size={24} />}
                {loading ? t.processing : t.predict}
              </button>
            )}
            
            {error && (
              <div className="mt-6 p-4 bg-red-50/80 backdrop-blur text-red-600 rounded-xl w-full text-center font-bold border border-red-200">
                {error}
              </div>
            )}

          </div>
        </div>

        {/* Right Panel: Result */}
        <div className="flex flex-col gap-6">
          {result ? (
            <PredictionResult result={result} t={t} />
          ) : (
            <div className="bg-white/50 backdrop-blur-md p-8 rounded-3xl border border-white/60 h-full flex items-center justify-center text-gray-400 font-medium border-dashed border-2 hover:bg-white/60 transition-colors">
              <div className="flex flex-col items-center gap-4">
                <div className="bg-gray-100 p-6 rounded-full shadow-inner">
                  <ImageIcon size={48} className="text-gray-300" />
                </div>
                <p className="tracking-wide">{t.predictionResult}</p>
              </div>
            </div>
          )}
        </div>
      </main>

      {/* History Panel */}
      <div className="max-w-6xl mx-auto px-4 pb-12">
        <PredictionHistory history={history} t={t} />
      </div>
    </>
  );
}
