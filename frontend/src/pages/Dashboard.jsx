import { useState, useEffect } from "react";
import api from "../lib/api";
import { Card, CardContent } from "../components/ui/card";
import { Activity, Heart, Users, TrendingUp } from "lucide-react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, Legend } from "recharts";

const COLORS = ["#059669", "#dc2626"];

function KPI({ icon: Icon, label, value, color }) {
  return (
    <div className="bg-white rounded-xl border border-slate-200 p-5 flex items-center gap-4">
      <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${color}`}>
        <Icon size={22} className="text-white" />
      </div>
      <div>
        <p className="text-sm text-slate-500">{label}</p>
        <p className="text-2xl font-bold text-slate-900">{value ?? "—"}</p>
      </div>
    </div>
  );
}

export default function Dashboard() {
  const [kpi, setKpi] = useState(null);
  const [modelPerf, setModelPerf] = useState([]);

  useEffect(() => {
    api.get("/dashboard/kpi").then((r) => setKpi(r.data)).catch(() => {});
    api.get("/model-metrics").then((r) => {
      const arr = Object.entries(r.data).map(([name, m]) => ({ name, ...m }));
      setModelPerf(arr);
    }).catch(() => {});
  }, []);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-900">Executive Dashboard</h1>
        <p className="text-slate-500 mt-1">Real-time clinical analytics overview</p>
      </div>

      {/* KPIs */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <KPI icon={Users} label="Total Patients" value={kpi?.total_patients} color="bg-[#1e3a5f]" />
        <KPI icon={Heart} label="Heart Disease %" value={kpi ? `${kpi.heart_disease_pct}%` : "—"} color="bg-red-600" />
        <KPI icon={Activity} label="Avg Age" value={kpi?.avg_age} color="bg-blue-600" />
        <KPI icon={TrendingUp} label="Avg Confidence" value={kpi?.avg_confidence} color="bg-emerald-600" />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Model Performance */}
        <div className="bg-white rounded-xl border border-slate-200 p-5">
          <h2 className="font-semibold text-slate-900 mb-4">Model Performance</h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={modelPerf}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="name" tick={{ fontSize: 11 }} />
              <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="accuracy" fill="#1e3a5f" radius={[4, 4, 0, 0]} name="Accuracy" />
              <Bar dataKey="roc_auc" fill="#2563eb" radius={[4, 4, 0, 0]} name="ROC AUC" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Risk Distribution */}
        <div className="bg-white rounded-xl border border-slate-200 p-5">
          <h2 className="font-semibold text-slate-900 mb-4">Risk Distribution</h2>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={[
                  { name: "Low Risk", value: (kpi?.total_patients || 1) - (kpi?.high_risk_patients || 0) },
                  { name: "High Risk", value: kpi?.high_risk_patients || 0 },
                ]}
                cx="50%" cy="50%"
                innerRadius={60} outerRadius={100}
                dataKey="value"
              >
                {COLORS.map((c, i) => <Cell key={i} fill={c} />)}
              </Pie>
              <Legend />
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
