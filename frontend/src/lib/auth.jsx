import { createContext, useContext, useState, useEffect } from "react";
import api from "../lib/api";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const token = localStorage.getItem("token");
    if (token) {
      api
        .get("/auth/me")
        .then((r) => setUser(r.data))
        .catch(() => localStorage.removeItem("token"))
        .finally(() => setLoading(false));
    } else {
      setLoading(false);
    }
  }, []);

  const login = async (username, password) => {
    const r = await api.post("/auth/login", { username, password });
    if (r.data.success) {
      localStorage.setItem("token", r.data.token);
      setUser(r.data.user);
    }
    return r.data;
  };

  const googleLogin = async (credential) => {
    const r = await api.post("/auth/google", { credential });
    if (r.data.success) {
      localStorage.setItem("token", r.data.token);
      setUser(r.data.user);
    }
    return r.data;
  };

  const logout = () => {
    localStorage.removeItem("token");
    setUser(null);
  };

  return (
    <AuthContext.Provider value={{ user, loading, login, googleLogin, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => useContext(AuthContext);
