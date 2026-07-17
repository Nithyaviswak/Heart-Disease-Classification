import { useState, useEffect } from "react";
import api from "../lib/api";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend, LineChart, Line,
} from "recharts";

export default function Analytics() {
  const [summary, setSummary] = useState(null);

  useEffect(() => {
    api.get("/analytics/summary").then((r) => setSummary(r.data)).catch(() => {});
  }, []);

  if (!summary) return <div className="text-center py-12 text-slate-400">Loading analytics...</div>;

  const ageData = summary.age_buckets || [];
  const riskData = summary.risk_distribution
    ? [
        { name: "Low Risk", value: summary.risk_distribution.low_risk },
        { name: "High Risk", value: summary.risk_distribution.high_risk },
      ]
    : [];
  const genderData = summary.gender
    ? [
        { name: "Male", Patients: summary.gender.male, Disease: summary.gender.male_disease },
        { name: "Female", Patients: summary.gender.female, Disease: summary.gender.female_disease },
      ]
    : [];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-900">Analytics</h1>
        <p className="text-slate-500 mt-1">Deep dive into clinical data</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Age Distribution */}
        <div className="bg-white rounded-xl border border-slate-200 p-5">
          <h2 className="font-semibold text-slate-900 mb-4">Age Distribution</h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={ageData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="age_group" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="total" fill="#1e3a5f" radius={[4, 4, 0, 0]} name="Total" />
              <Bar dataKey="disease" fill="#dc2626" radius={[4, 4, 0, 0]} name="Disease" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Gender Distribution */}
        <div className="bg-white rounded-xl border border-slate-200 p-5">
          <h2 className="font-semibold text-slate-900 mb-4">Gender Distribution</h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={genderData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="name" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="Patients" fill="#2563eb" radius={[4, 4, 0, 0]} />
              <Bar dataKey="Disease" fill="#dc2626" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Risk Pie */}
        <div className="bg-white rounded-xl border border-slate-200 p-5">
          <h2 className="font-semibold text-slate-900 mb-4">Risk Distribution</h2>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie data={riskData} cx="50%" cy="50%" innerRadius={60} outerRadius={100} dataKey="value">
                {riskData.map((_, i) => (
                  <Cell key={i} fill={i === 0 ? "#059669" : "#dc2626"} />
                ))}
              </Pie>
              <Legend />
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Hospital Summary */}
        <div className="bg-white rounded-xl border border-slate-200 p-5">
          <h2 className="font-semibold text-slate-900 mb-4">Hospital Summary</h2>
          <div className="space-y-4">
            <div className="flex justify-between py-2 border-b border-slate-100">
              <span className="text-slate-600">Total Patients</span>
              <span className="font-bold text-lg">{summary.hospital_summary?.risk?.total_patients || 0}</span>
            </div>
            <div className="flex justify-between py-2 border-b border-slate-100">
              <span className="text-slate-600">Average Age</span>
              <span className="font-bold text-lg">{summary.hospital_summary?.avg_age || 0}</span>
            </div>
            <div className="flex justify-between py-2 border-b border-slate-100">
              <span className="text-slate-600">High Risk %</span>
              <span className="font-bold text-lg text-red-600">{summary.hospital_summary?.risk?.high_risk_pct || 0}%</span>
            </div>
            <div className="flex justify-between py-2">
              <span className="text-slate-600">Male Patients</span>
              <span className="font-bold text-lg">{summary.gender?.male || 0}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
