import { useState, useEffect } from "react";
import api from "../lib/api";
import { Search, Download } from "lucide-react";

export default function History() {
  const [data, setData] = useState([]);
  const [page, setPage] = useState(1);
  const [total, setTotal] = useState(0);
  const [search, setSearch] = useState("");

  useEffect(() => {
    api.get(`/history?page=${page}&per_page=20`).then((r) => {
      setData(r.data.data);
      setTotal(r.data.total);
    });
  }, [page]);

  const filtered = data.filter((r) =>
    r.model_name?.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Prediction History</h1>
          <p className="text-slate-500 mt-1">{total} total predictions</p>
        </div>
        <a
          href="/api/download/csv/predictions"
          className="flex items-center gap-2 px-4 py-2 bg-[#1e3a5f] text-white rounded-lg text-sm font-medium hover:bg-[#2563eb] transition-colors"
        >
          <Download size={16} /> Export CSV
        </a>
      </div>

      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={16} />
        <input
          type="text"
          placeholder="Filter by model..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="w-full pl-10 pr-4 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
        />
      </div>

      <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-slate-50 border-b border-slate-200">
                <th className="text-left px-4 py-3 font-semibold text-slate-600">ID</th>
                <th className="text-left px-4 py-3 font-semibold text-slate-600">Model</th>
                <th className="text-left px-4 py-3 font-semibold text-slate-600">Prediction</th>
                <th className="text-left px-4 py-3 font-semibold text-slate-600">Probability</th>
                <th className="text-left px-4 py-3 font-semibold text-slate-600">Confidence</th>
                <th className="text-left px-4 py-3 font-semibold text-slate-600">Risk Score</th>
                <th className="text-left px-4 py-3 font-semibold text-slate-600">Date</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((r) => (
                <tr key={r.id} className="border-b border-slate-100 hover:bg-slate-50">
                  <td className="px-4 py-3 text-slate-500">#{r.id}</td>
                  <td className="px-4 py-3 font-medium">{r.model_name}</td>
                  <td className="px-4 py-3">
                    <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                      r.prediction === 1 ? "bg-red-100 text-red-700" : "bg-emerald-100 text-emerald-700"
                    }`}>
                      {r.prediction === 1 ? "High Risk" : "Low Risk"}
                    </span>
                  </td>
                  <td className="px-4 py-3">{r.probability?.toFixed(3)}</td>
                  <td className="px-4 py-3">{(r.confidence * 100).toFixed(1)}%</td>
                  <td className="px-4 py-3">{r.risk_score}</td>
                  <td className="px-4 py-3 text-slate-500">{r.created_at}</td>
                </tr>
              ))}
              {filtered.length === 0 && (
                <tr><td colSpan={7} className="px-4 py-8 text-center text-slate-400">No predictions found</td></tr>
              )}
            </tbody>
          </table>
        </div>
        {total > 20 && (
          <div className="flex items-center justify-between px-4 py-3 border-t border-slate-200">
            <button
              disabled={page <= 1}
              onClick={() => setPage((p) => p - 1)}
              className="px-3 py-1.5 text-sm bg-slate-100 rounded-lg disabled:opacity-50 hover:bg-slate-200"
            >
              Previous
            </button>
            <span className="text-sm text-slate-500">Page {page} of {Math.ceil(total / 20)}</span>
            <button
              disabled={page >= Math.ceil(total / 20)}
              onClick={() => setPage((p) => p + 1)}
              className="px-3 py-1.5 text-sm bg-slate-100 rounded-lg disabled:opacity-50 hover:bg-slate-200"
            >
              Next
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
