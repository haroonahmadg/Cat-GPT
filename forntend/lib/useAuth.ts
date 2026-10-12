"use client";

import { useCallback, useEffect, useState } from "react";
import type { AuthUser, LoginResponseDTO, MeResponseDTO } from "@/lib/types";

const TOKEN_STORAGE_KEY = "ai_assistant_session_token";
const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

function readStoredToken(): string | null {
  if (typeof window === "undefined") return null;
  try {
    return window.localStorage.getItem(TOKEN_STORAGE_KEY);
  } catch {
    return null;
  }
}

function writeStoredToken(token: string | null) {
  if (typeof window === "undefined") return;
  try {
    if (token) {
      window.localStorage.setItem(TOKEN_STORAGE_KEY, token);
    } else {
      window.localStorage.removeItem(TOKEN_STORAGE_KEY);
    }
  } catch {
    // Private browsing / storage disabled — login still works for this
    // tab, it just won't survive a reload.
  }
}

export function getStoredToken(): string | null {
  return readStoredToken();
}

async function readJson<T>(res: Response): Promise<T> {
  const text = await res.text();
  if (!text) return {} as T;

  try {
    return JSON.parse(text) as T;
  } catch {
    return {} as T;
  }
}

export function useAuth() {
  const [isAuthenticated, setIsAuthenticated] = useState<boolean | null>(null);
  const [user, setUser] = useState<AuthUser | null>(null);
  const [loginError, setLoginError] = useState<string | null>(null);
  const [loggingIn, setLoggingIn] = useState(false);

  const clearSession = useCallback(() => {
    writeStoredToken(null);
    setUser(null);
    setIsAuthenticated(false);
  }, []);

  const storeSession = useCallback((token: string, authUser: AuthUser) => {
    writeStoredToken(token);
    setUser(authUser);
    setIsAuthenticated(true);
  }, []);

  const ensureSessionUser = useCallback(async (token: string): Promise<boolean> => {
    try {
      const res = await fetch(`${API_BASE_URL}/auth/me`, {
        headers: { Authorization: `Bearer ${token}` },
      });

      if (!res.ok) {
        clearSession();
        return false;
      }

      const data = (await readJson<MeResponseDTO>(res)) as MeResponseDTO;
      if (!data?.email) {
        clearSession();
        return false;
      }

      setUser({
        email: data.email,
        name: data.name ?? null,
        hasPassword: !!data.has_password,
      });
      setIsAuthenticated(true);
      return true;
    } catch {
      clearSession();
      return false;
    }
  }, [clearSession]);

  useEffect(() => {
    const token = readStoredToken();
    if (!token) {
      setIsAuthenticated(false);
      setUser(null);
      return;
    }

    void ensureSessionUser(token);
  }, [ensureSessionUser]);

  const loginWithGoogle = useCallback(async (credential: string) => {
    setLoggingIn(true);
    setLoginError(null);

    try {
      const res = await fetch(`${API_BASE_URL}/auth/google`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ credential }),
      });

      const data = (await readJson<LoginResponseDTO>(res)) as LoginResponseDTO;

      if (!res.ok) {
        setLoginError(data?.detail ?? "Sign-in failed.");
        return;
      }

      if (!data?.token) {
        setLoginError("Sign-in succeeded but no session was returned.");
        return;
      }

      storeSession(data.token ?? "", {
        email: data.email ?? "",
        name: data.name ?? null,
        hasPassword: !!data.has_password,
      });
    } catch {
      setLoginError("Couldn't reach the server. Is the backend running?");
    } finally {
      setLoggingIn(false);
    }
  }, [storeSession]);

  const loginWithEmail = useCallback(async (email: string, password: string) => {
    setLoggingIn(true);
    setLoginError(null);

    try {
      const res = await fetch(`${API_BASE_URL}/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });

      const data = (await readJson<LoginResponseDTO>(res)) as LoginResponseDTO;

      if (!res.ok) {
        setLoginError(data?.detail ?? "Login failed.");
        return;
      }

      if (!data?.token) {
        setLoginError("Login succeeded but no session was returned.");
        return;
      }

      storeSession(data.token ?? "", {
        email: data.email ?? "",
        name: data.name ?? null,
        hasPassword: !!data.has_password,
      });
    } catch {
      setLoginError("Couldn't reach the server. Is the backend running?");
    } finally {
      setLoggingIn(false);
    }
  }, [storeSession]);

  const registerWithEmail = useCallback(async (email: string, password: string, name?: string) => {
    setLoggingIn(true);
    setLoginError(null);

    try {
      const res = await fetch(`${API_BASE_URL}/auth/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password, name: name?.trim() || undefined }),
      });

      const data = (await readJson<LoginResponseDTO>(res)) as LoginResponseDTO;

      if (!res.ok) {
        setLoginError(data?.detail ?? "Sign-up failed.");
        return;
      }

      if (!data?.token) {
        setLoginError("Account created but no session was returned.");
        return;
      }

      storeSession(data.token ?? "", {
        email: data.email ?? "",
        name: data.name ?? null,
        hasPassword: !!data.has_password,
      });
    } catch {
      setLoginError("Couldn't reach the server. Is the backend running?");
    } finally {
      setLoggingIn(false);
    }
  }, [storeSession]);

  const setPassword = useCallback(async (newPassword: string, currentPassword?: string) => {
    const token = readStoredToken();
    if (!token) {
      setLoginError("You must be signed in to set a password.");
      return false;
    }

    try {
      const res = await fetch(`${API_BASE_URL}/auth/set-password`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ new_password: newPassword, current_password: currentPassword ?? null }),
      });

      const data = (await readJson<{ detail?: string; password_set?: boolean }>(res)) as {
        detail?: string;
        password_set?: boolean;
      };

      if (!res.ok) {
        setLoginError(data?.detail ?? "Unable to update password.");
        return false;
      }

      if (user) {
        setUser({ ...user, hasPassword: true });
      }
      setLoginError(null);
      return !!data?.password_set;
    } catch {
      setLoginError("Couldn't reach the server. Is the backend running?");
      return false;
    }
  }, [user]);

  const logout = useCallback(() => {
    clearSession();
  }, [clearSession]);

  const handleUnauthorized = useCallback(() => {
    clearSession();
  }, [clearSession]);

  return {
    isAuthenticated,
    user,
    loginError,
    loggingIn,
    loginWithGoogle,
    loginWithEmail,
    registerWithEmail,
    setPassword,
    logout,
    handleUnauthorized,
  };
}