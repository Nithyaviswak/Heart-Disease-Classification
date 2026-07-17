import { useState } from "react";
import { Search } from "lucide-react";

export default function Patients() {
  // Placeholder for patient management — extensible
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-900">Patient Records</h1>
        <p className="text-slate-500 mt-1">Manage patient data and view history</p>
      </div>

      <div className="bg-white rounded-xl border border-slate-200 p-12 text-center">
        <div className="mx-auto w-16 h-16 bg-slate-100 rounded-full flex items-center justify-center mb-4">
          <Search className="text-slate-400" size={24} />
        </div>
        <h2 className="text-lg font-semibold text-slate-700">Patient Management Module</h2>
        <p className="text-slate-500 mt-2 max-w-md mx-auto">
          This module connects to the prediction database. Patient records are created automatically when predictions are made.
        </p>
        <a
          href="/api/download/excel"
          className="inline-flex items-center gap-2 mt-6 px-4 py-2 bg-[#1e3a5f] text-white rounded-lg text-sm font-medium hover:bg-[#2563eb] transition-colors"
        >
          Download Full Report (Excel)
        </a>
      </div>
    </div>
  );
}
