import { useState, useEffect } from "react";
import api from "../lib/api";

const FEATURES = [
  { name: "age", label: "Age", min: 20, max: 100, step: 1, default: 55 },
  { name: "sex", label: "Sex (1=Male, 0=Female)", min: 0, max: 1, step: 1, default: 1 },
  { name: "cp", label: "Chest Pain Type (1-4)", min: 1, max: 4, step: 1, default: 1 },
  { name: "trestbps", label: "Resting Blood Pressure", min: 80, max: 200, step: 1, default: 130 },
  { name: "chol", label: "Serum Cholesterol", min: 100, max: 600, step: 1, default: 250 },
  { name: "fbs", label: "Fasting Blood Sugar > 120", min: 0, max: 1, step: 1, default: 0 },
  { name: "restecg", label: "Resting ECG (0-2)", min: 0, max: 2, step: 1, default: 0 },
  { name: "thalach", label: "Max Heart Rate", min: 60, max: 220, step: 1, default: 150 },
  { name: "exang", label: "Exercise Angina (1=Yes)", min: 0, max: 1, step: 1, default: 0 },
  { name: "oldpeak", label: "ST Depression", min: 0, max: 7, step: 0.1, default: 1 },
  { name: "slope", label: "Peak ST Slope (1-3)", min: 1, max: 3, step: 1, default: 2 },
  { name: "ca", label: "Major Vessels (0-3)", min: 0, max: 3, step: 1, default: 0 },
  { name: "thal", label: "Thalassemia (3/6/7)", min: 3, max: 7, step: 1, default: 3 },
];

const MODELS = ["Logistic Regression", "Random Forest", "SVM", "TensorFlow MLP", "PyTorch MLP"];

export default function Predict() {
  const [form, setForm] = useState({});
  const [model, setModel] = useState("Random Forest");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const init = {};
    FEATURES.forEach((f) => (init[f.name] = f.default));
    setForm(init);
  }, []);

  const handleChange = (name, value) => setForm((f) => ({ ...f, [name]: value }));

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const r = await api.post("/predict", { model, features: form });
      setResult(r.data);
    } catch (err) {
      setResult({ error: err.response?.data?.error || "Prediction failed" });
    }
    setLoading(false);
  };

  const isHigh = result?.prediction === 1;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-900">Risk Assessment</h1>
        <p className="text-slate-500 mt-1">Enter patient data for AI-powered prediction</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* Form */}
        <div className="lg:col-span-3 bg-white rounded-xl border border-slate-200 p-6">
          <div className="flex gap-2 mb-6">
            {MODELS.map((m) => (
              <button
                key={m}
                onClick={() => setModel(m)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  model === m
                    ? "bg-[#1e3a5f] text-white"
                    : "bg-slate-100 text-slate-600 hover:bg-slate-200"
                }`}
              >
                {m}
              </button>
            ))}
          </div>
          <form onSubmit={handleSubmit} className="grid grid-cols-2 gap-4">
            {FEATURES.map((f) => (
              <div key={f.name}>
                <label className="block text-xs font-medium text-slate-500 uppercase mb-1">{f.label}</label>
                <input
                  type="number"
                  min={f.min} max={f.max} step={f.step}
                  value={form[f.name] ?? ""}
                  onChange={(e) => handleChange(f.name, e.target.value)}
                  className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none text-sm"
                  required
                />
              </div>
            ))}
            <div className="col-span-2 mt-2">
              <button
                type="submit"
                disabled={loading}
                className="w-full py-3 bg-gradient-to-r from-[#1e3a5f] to-[#2563eb] text-white font-semibold rounded-lg hover:opacity-90 transition-opacity disabled:opacity-50"
              >
                {loading ? "Analyzing..." : "Analyze Risk"}
              </button>
            </div>
          </form>
        </div>

        {/* Result */}
        <div className="lg:col-span-2 space-y-4">
          {result && !result.error && (
            <>
              <div className={`rounded-xl p-6 text-center ${isHigh ? "bg-red-50 border border-red-200" : "bg-emerald-50 border border-emerald-200"}`}>
                <div className={`w-16 h-16 rounded-full mx-auto mb-3 flex items-center justify-center ${isHigh ? "bg-red-100" : "bg-emerald-100"}`}>
                  {isHigh ? (
                    <svg className="w-8 h-8 text-red-600" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
                  ) : (
                    <svg className="w-8 h-8 text-emerald-600" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
                  )}
                </div>
                <h2 className={`text-xl font-bold ${isHigh ? "text-red-700" : "text-emerald-700"}`}>
                  {result.label}
                </h2>
                <p className="text-sm text-slate-500 mt-1">
                  Confidence: {(result.confidence * 100).toFixed(1)}% &middot; {result.model}
                </p>
                <div className="mt-4">
                  <div className="text-3xl font-bold text-slate-900">{result.risk_score}%</div>
                  <p className="text-xs text-slate-400">Risk Score</p>
                  <div className="mt-2 w-full h-2 bg-slate-200 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all ${result.risk_score > 60 ? "bg-red-500" : result.risk_score > 35 ? "bg-amber-500" : "bg-emerald-500"}`}
                      style={{ width: `${result.risk_score}%` }}
                    />
                  </div>
                </div>
              </div>

              {/* SHAP Explanation */}
              {result.shap && !result.shap.error && (
                <div className="bg-white rounded-xl border border-slate-200 p-5">
                  <h3 className="font-semibold text-sm text-slate-900 mb-3">Key Factors</h3>
                  <div className="space-y-2">
                    {result.shap.top_features?.map((f, i) => (
                      <div key={i} className="flex justify-between text-sm">
                        <span className="text-slate-600 capitalize">{f.feature}</span>
                        <span className={f.value > 0 ? "text-red-600 font-medium" : "text-emerald-600 font-medium"}>
                          {f.value > 0 ? "+" : ""}{f.value.toFixed(3)}
                        </span>
                      </div>
                    ))}
                  </div>
                  {result.shap.positive_contributors?.length > 0 && (
                    <div className="mt-3 pt-3 border-t border-slate-100">
                      <p className="text-xs font-medium text-red-600 mb-1">Increases Risk</p>
                      <div className="flex flex-wrap gap-1">
                        {result.shap.positive_contributors.map((c, i) => (
                          <span key={i} className="px-2 py-0.5 bg-red-50 text-red-700 text-xs rounded-full capitalize">{c.feature}</span>
                        ))}
                      </div>
                    </div>
                  )}
                  {result.shap.negative_contributors?.length > 0 && (
                    <div className="mt-2">
                      <p className="text-xs font-medium text-emerald-600 mb-1">Decreases Risk</p>
                      <div className="flex flex-wrap gap-1">
                        {result.shap.negative_contributors.map((c, i) => (
                          <span key={i} className="px-2 py-0.5 bg-emerald-50 text-emerald-700 text-xs rounded-full capitalize">{c.feature}</span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </>
          )}

          {result?.error && (
            <div className="bg-red-50 border border-red-200 rounded-xl p-4 text-red-700 text-sm">{result.error}</div>
          )}

          {!result && (
            <div className="bg-white rounded-xl border border-slate-200 p-6 text-center text-slate-400">
              <p>Enter patient data and select a model to see the prediction result here</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
