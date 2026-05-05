# AgroVision – AI Crop Disease Detector 🌱

AgroVision is an advanced, AI-powered smart plant health assistant. It combines a state-of-the-art deep learning vision model (ResNet-18) with the powerful reasoning capabilities of **Google Gemini 2.5 Flash** to not only diagnose crop diseases with high accuracy but also generate dynamic, actionable agronomic insights in both English and Hindi.

![AgroVision UI Demo](https://img.shields.io/badge/UI-React_Vite-61DAFB?style=for-the-badge&logo=react)
![Backend](https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge&logo=fastapi)
![AI Model](https://img.shields.io/badge/AI-PyTorch_ResNet18-EE4C2C?style=for-the-badge&logo=pytorch)
![LLM Integration](https://img.shields.io/badge/LLM-Gemini_2.5_Flash-4285F4?style=for-the-badge&logo=google)

---

## ✨ Features

- **📷 Real-Time Disease Detection**: Upload an image of a plant leaf and instantly identify across **86 different crop classes and diseases**.
- **🧠 Dynamic Agronomic Insights (Gemini AI)**: Generates highly detailed, numbered treatment steps with precise chemical dosages, prevention strategies, and fertilizer recommendations.
- **🌐 Bilingual Support**: Seamlessly toggle between **English and Hindi (हिंदी)** for all encyclopedia entries and AI-generated treatment plans.
- **🛡️ Robust Fallback Mechanism**: If a plant is healthy (or if the Gemini API quota is exhausted), the system gracefully falls back to a curated local database without interrupting the user experience.
- **🎨 Modern "Bento Box" UI**: A sleek, responsive, glassmorphism-inspired interface built with React and Tailwind CSS.
- **📖 Disease Encyclopedia**: Browse a comprehensive database of all 86 supported crops and diseases.

## 🏗️ System Architecture

1. **Frontend (Client)**: React.js powered by Vite. Provides a modern scanner interface, prediction history, and an encyclopedia.
2. **Backend (API)**: FastAPI server handling image uploads, preprocessing, and routing.
3. **Vision Model**: PyTorch `ResNet-18` (Transfer Learning) fine-tuned on the PlantVillage dataset.
4. **Generative AI Layer**: Google GenAI SDK (`gemini-2.5-flash`) orchestrated with custom system prompts for structured agronomic outputs.

---

## 🚀 Getting Started

### Prerequisites
- **Python 3.10+**
- **Node.js 18+**
- **Google Gemini API Key** ([Get one here](https://aistudio.google.com/app/apikey))

### 1. Backend Setup (FastAPI + PyTorch)

1. Clone the repository and navigate to the root directory.
2. Activate the virtual environment (or create a new one):
   ```bash
   # Windows
   python -m venv venv310
   .\venv310\Scripts\Activate
   
   # Linux/macOS
   python3 -m venv venv310
   source venv310/bin/activate
   ```
3. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up your environment variables:
   - Copy `.env.example` to a new file named `.env`.
   - Add your Gemini API key: `GEMINI_API_KEY=your_api_key_here`
5. Start the Uvicorn server:
   ```bash
   uvicorn app:app --reload --port 8000
   ```

### 2. Frontend Setup (React + Vite)

1. Open a **new terminal tab** and navigate to the `frontend` directory:
   ```bash
   cd frontend
   ```
2. Install the Node modules:
   ```bash
   npm install
   ```
3. Start the Vite development server:
   ```bash
   npm run dev
   ```
4. Open your browser and navigate to `http://localhost:5173`.

---

## 📁 Project Structure

```text
AgroVision/
├── app.py                   # Main FastAPI server and Gemini integration logic
├── requirements.txt         # Python dependencies
├── .env                     # API keys and environment configurations (ignored in git)
├── models/
│   └── best_model.pth       # Trained PyTorch ResNet-18 weights
├── inference/
│   ├── disease_info.json    # Local JSON fallback database (English)
│   ├── disease_info_hi.json # Local JSON fallback database (Hindi)
│   └── class_labels.json    # Mapping of 86 disease classes
├── frontend/                # React Vite Application
│   ├── src/
│   │   ├── components/      # UI Components (Scanner, Encyclopedia, Bento layout)
│   │   ├── pages/           # Application Routes
│   │   └── translations.js  # i18n logic for Hindi/English
│   ├── tailwind.config.js   # UI Styling configuration
│   └── vite.config.js       # Proxy configurations mapping to FastAPI
└── training/                # Scripts used to originally train the ResNet model
```

## 🛠️ Tech Stack
* **Python Backend:** FastAPI, PyTorch, Pillow, google-genai, python-dotenv
* **Frontend:** React, Vite, Tailwind CSS, Framer Motion, Lucide React
* **AI & Machine Learning:** ResNet-18 (Vision), Gemini 2.5 Flash (LLM)

---
*Developed by Gurroop Singh*
