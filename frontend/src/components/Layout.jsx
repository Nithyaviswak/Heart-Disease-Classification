import { NavLink, Outlet, useNavigate } from "react-router-dom";
import { useAuth } from "../lib/auth";
import {
  LayoutDashboard, Activity, History, BarChart3, Users, LogOut,
} from "lucide-react";

const NAV = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard },
  { to: "/predict", label: "Predict", icon: Activity },
  { to: "/history", label: "History", icon: History },
  { to: "/analytics", label: "Analytics", icon: BarChart3 },
  { to: "/patients", label: "Patients", icon: Users },
];

export default function Layout() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate("/login");
  };

  return (
    <div className="min-h-screen flex">
      {/* Sidebar */}
      <aside className="w-64 bg-[#1e3a5f] text-white flex flex-col shrink-0">
        <div className="p-6 border-b border-white/10">
          <h1 className="text-lg font-bold tracking-tight">Heart Analytics</h1>
          <p className="text-xs text-blue-200 mt-1">Healthcare Platform v2.0</p>
        </div>
        <nav className="flex-1 p-4 space-y-1">
          {NAV.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/"}
              className={({ isActive }) =>
                `flex items-center gap-3 px-4 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                  isActive
                    ? "bg-white/15 text-white"
                    : "text-blue-200 hover:bg-white/10 hover:text-white"
                }`
              }
            >
              <Icon size={18} />
              {label}
            </NavLink>
          ))}
        </nav>
        <div className="p-4 border-t border-white/10">
          <div className="text-sm font-medium truncate">{user?.full_name || user?.username}</div>
          <div className="text-xs text-blue-300 capitalize">{user?.role}</div>
          <button
            onClick={handleLogout}
            className="mt-3 flex items-center gap-2 text-sm text-blue-200 hover:text-white transition-colors"
          >
            <LogOut size={16} /> Sign Out
          </button>
        </div>
      </aside>

      {/* Main */}
      <main className="flex-1 bg-slate-50 overflow-auto">
        <div className="p-6 lg:p-8 max-w-7xl mx-auto">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
