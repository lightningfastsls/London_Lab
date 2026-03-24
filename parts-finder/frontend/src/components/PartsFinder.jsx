// ══════════════════════════════════════════════════════════════════════
// PartsFinder — Main orchestrator component
//
// Layout (top → bottom):
//   Header → Plate Input → Error Banner → Vehicle Info →
//   Coverage Summary → Category Grid → Footer
//
// Delegates vehicle display to <VehicleInfo> and each product category
// to <CategoryCard>. Shared sub-component (Tag) is defined inline here,
// matching the oil-finder prototype pattern.
// ══════════════════════════════════════════════════════════════════════

import React, { useState } from "react";
import {
  lookupPlate,
  PlateFormatError,
  PlateNotFoundError,
  ServiceError,
} from "../api/partsApi";
import VehicleInfo from "./VehicleInfo";
import CategoryCard from "./CategoryCard";

// ── Category ordering (display order in the grid) ───────────────────

const CATEGORY_KEYS = [
  "oil", "oil_filter", "air_filter", "cabin_filter",
  "brakes", "bulbs", "coolant",
];

// ── SVG Icons ───────────────────────────────────────────────────────

const OilDrop = ({ s = 24, c = "currentColor" }) => (
  <svg width={s} height={s} viewBox="0 0 24 24" fill="none">
    <path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0L12 2.69z" fill={c} opacity=".15" />
    <path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0L12 2.69z" stroke={c} strokeWidth="1.5" strokeLinejoin="round" />
  </svg>
);

const SearchIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="11" cy="11" r="8" />
    <line x1="21" y1="21" x2="16.65" y2="16.65" />
  </svg>
);

const Spinner = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"
    style={{ animation: "spin 1s linear infinite" }}
  >
    <path d="M12 2a10 10 0 0 1 10 10" />
  </svg>
);

// ── Shared sub-components (same as prototype) ───────────────────────

function Tag({ children, bg = "rgba(234,179,8,.08)", color = "#eab308", border = "rgba(234,179,8,.15)" }) {
  return (
    <span style={{
      fontSize: 11, fontWeight: 600, padding: "4px 10px", borderRadius: 6,
      background: bg, border: `1px solid ${border}`, color,
      fontFamily: "'Space Mono',monospace",
    }}>
      {children}
    </span>
  );
}

// ── Data source badge ───────────────────────────────────────────────

const DATA_SOURCE_STYLES = {
  database:    { bg: "rgba(34,197,94,.08)",  color: "#86efac", border: "rgba(34,197,94,.15)",  label: "Database" },
  hybrid:      { bg: "rgba(168,85,247,.08)", color: "#c084fc", border: "rgba(168,85,247,.15)", label: "Hybrid" },
  ai_fallback: { bg: "rgba(168,85,247,.08)", color: "#c084fc", border: "rgba(168,85,247,.15)", label: "AI Estimated" },
};

// ── Error message mapping ───────────────────────────────────────────

function errorMessage(err) {
  if (err instanceof PlateFormatError)   return { title: "Invalid Plate", detail: err.message };
  if (err instanceof PlateNotFoundError) return { title: "Not Found",     detail: err.message };
  if (err instanceof ServiceError)       return { title: "Service Error", detail: err.message };
  return { title: "Error", detail: err.message || "Something went wrong" };
}

// ── Electric vehicle detection ──────────────────────────────────────

function isElectricVehicle(categories) {
  if (!categories) return false;
  const nonCoolant = ["oil", "oil_filter", "air_filter", "cabin_filter", "brakes", "bulbs"];
  const allNonCoolantNull = nonCoolant.every((k) => categories[k] == null);
  return allNonCoolantNull && categories.coolant != null;
}

// ══════════════════════════════════════════════════════════════════════
// Main Component
// ══════════════════════════════════════════════════════════════════════

export default function PartsFinder() {
  const [plate, setPlate] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const canSearch = plate.replace(/[-\s]/g, "").length >= 5 && !loading;

  async function handleSearch(e) {
    e?.preventDefault();
    if (!canSearch) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await lookupPlate(plate.replace(/[-\s]/g, ""));
      setResult(data);
    } catch (err) {
      setError(err);
    } finally {
      setLoading(false);
    }
  }

  // Categories that have data (non-null)
  const activeCategories = result
    ? CATEGORY_KEYS.filter((k) => result.categories[k] != null)
    : [];

  const isEV = result ? isElectricVehicle(result.categories) : false;
  const isAiSource = result?.data_source === "ai_fallback" || result?.data_source === "hybrid";

  return (
    <div style={{
      minHeight: "100vh",
      background: "linear-gradient(145deg,#0a0f1a,#111927,#0d1520)",
      fontFamily: "'DM Sans','Segoe UI',system-ui,sans-serif",
      color: "#e2e8f0",
      position: "relative",
      overflow: "hidden",
    }}>
      {/* Background effects */}
      <div style={{
        position: "fixed", inset: 0, opacity: .03,
        backgroundImage: "radial-gradient(circle at 1px 1px,white 1px,transparent 0)",
        backgroundSize: "40px 40px",
      }} />
      <div style={{
        position: "fixed", top: -200, right: -200, width: 600, height: 600,
        background: "radial-gradient(circle,rgba(234,179,8,.06),transparent 70%)",
      }} />

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');
        @keyframes fadeUp{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}
        @keyframes spin{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}
        *{box-sizing:border-box}
        input{outline:none}
      `}</style>

      <div style={{
        position: "relative", zIndex: 1,
        maxWidth: 760, margin: "0 auto",
        padding: "40px 20px 60px",
      }}>

        {/* ─── Header ─────────────────────────────────────────── */}
        <div style={{ textAlign: "center", marginBottom: 40, animation: "fadeUp .5s ease" }}>
          <div style={{
            display: "inline-flex", alignItems: "center", gap: 10,
            background: "rgba(234,179,8,.08)",
            border: "1px solid rgba(234,179,8,.15)",
            borderRadius: 40, padding: "6px 16px 6px 10px",
            marginBottom: 20, fontSize: 12, fontWeight: 500,
            color: "#eab308", letterSpacing: .5,
            textTransform: "uppercase",
            fontFamily: "'Space Mono',monospace",
          }}>
            <OilDrop s={16} c="#eab308" />
            Cloudy Claude · Parts Finder
          </div>

          <h1 style={{
            fontSize: "clamp(28px,5vw,40px)", fontWeight: 700,
            margin: "0 0 8px", letterSpacing: -.5,
            background: "linear-gradient(135deg,#f8fafc,#94a3b8)",
            WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
          }}>
            Find Your Car Parts
          </h1>
          <p style={{
            fontSize: 15, color: "#64748b", margin: 0,
            maxWidth: 480, marginLeft: "auto", marginRight: "auto",
            lineHeight: 1.5,
          }}>
            Enter your license plate and get oil, filters, brakes, bulbs & coolant specs instantly
          </p>
        </div>

        {/* ─── Plate Input ────────────────────────────────────── */}
        <form
          onSubmit={handleSearch}
          style={{
            background: "rgba(255,255,255,.03)", borderRadius: 16,
            border: "1px solid rgba(255,255,255,.07)", padding: 28,
            animation: "fadeUp .5s ease .15s both",
          }}
        >
          <label style={{
            fontSize: 11, fontWeight: 600, textTransform: "uppercase",
            letterSpacing: 1, color: "#64748b", marginBottom: 10,
            display: "block", fontFamily: "'Space Mono',monospace",
          }}>
            License Plate
          </label>

          <div style={{ display: "flex", gap: 12 }}>
            <input
              type="text"
              inputMode="numeric"
              value={plate}
              onChange={(e) => setPlate(e.target.value)}
              placeholder="1234567"
              maxLength={10}
              style={{
                flex: 1, padding: "14px 18px", borderRadius: 10,
                border: "2px solid rgba(234,179,8,.25)",
                background: "rgba(254,249,195,.08)",
                color: "#fef9c3", fontSize: 22, fontWeight: 700,
                fontFamily: "'Space Mono',monospace",
                letterSpacing: 3, textAlign: "center",
              }}
            />
            <button
              type="submit"
              disabled={!canSearch}
              style={{
                padding: "14px 28px", borderRadius: 10, border: "none",
                cursor: canSearch ? "pointer" : "not-allowed",
                fontSize: 14, fontWeight: 700, fontFamily: "inherit",
                display: "flex", alignItems: "center", gap: 10,
                background: canSearch
                  ? "linear-gradient(135deg,#eab308,#ca8a04)"
                  : "rgba(255,255,255,.06)",
                color: canSearch ? "#0a0f1a" : "#475569",
                transition: "all .3s", whiteSpace: "nowrap",
              }}
            >
              {loading ? <Spinner /> : <SearchIcon />}
              {loading ? "Searching..." : "Search"}
            </button>
          </div>
        </form>

        {/* ─── Error Banner ───────────────────────────────────── */}
        {error && (
          <div style={{
            marginTop: 16, padding: "18px 24px", borderRadius: 14,
            background: "rgba(239,68,68,.06)",
            border: "1px solid rgba(239,68,68,.15)",
            animation: "fadeUp .3s ease",
          }}>
            <p style={{ fontSize: 15, fontWeight: 700, margin: "0 0 4px", color: "#fca5a5" }}>
              {errorMessage(error).title}
            </p>
            <p style={{ fontSize: 13, margin: 0, color: "#f87171", lineHeight: 1.5 }}>
              {errorMessage(error).detail}
            </p>
          </div>
        )}

        {/* ─── Results ────────────────────────────────────────── */}
        {result && (
          <div style={{ marginTop: 24, animation: "fadeUp .4s ease" }}>

            {/* Vehicle Info */}
            <VehicleInfo vehicle={result.vehicle} />

            {/* Electric Vehicle Banner */}
            {isEV && (
              <div style={{
                marginTop: 12, padding: "20px 24px", borderRadius: 14,
                background: "rgba(34,197,94,.06)",
                border: "1px solid rgba(34,197,94,.15)",
                color: "#86efac", fontSize: 14,
              }}>
                <strong>Electric Vehicle</strong> — No engine oil required. Coolant and brake fluid specs shown below.
              </div>
            )}

            {/* Coverage Summary */}
            <div style={{
              display: "flex", alignItems: "center", flexWrap: "wrap",
              gap: 10, marginTop: 16, marginBottom: 16,
            }}>
              <Tag
                bg="rgba(255,255,255,.05)"
                color="#e2e8f0"
                border="rgba(255,255,255,.1)"
              >
                {result.coverage}
              </Tag>

              {result.data_source && DATA_SOURCE_STYLES[result.data_source] && (
                <Tag
                  bg={DATA_SOURCE_STYLES[result.data_source].bg}
                  color={DATA_SOURCE_STYLES[result.data_source].color}
                  border={DATA_SOURCE_STYLES[result.data_source].border}
                >
                  {DATA_SOURCE_STYLES[result.data_source].label}
                </Tag>
              )}

              {result.unmatched_categories?.length > 0 && (
                <span style={{
                  fontSize: 11, color: "#475569",
                  fontFamily: "'Space Mono',monospace",
                }}>
                  Missing: {result.unmatched_categories.join(", ")}
                </span>
              )}
            </div>

            {/* Category Grid */}
            <div style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
              gap: 12,
            }}>
              {activeCategories.map((key) => (
                <CategoryCard
                  key={key}
                  type={key}
                  data={result.categories[key]}
                  isAiFallback={isAiSource}
                />
              ))}
            </div>
          </div>
        )}

        {/* ─── Footer ─────────────────────────────────────────── */}
        <p style={{
          textAlign: "center", marginTop: 48,
          fontSize: 11, color: "#334155",
          fontFamily: "'Space Mono',monospace",
        }}>
          Cloudy Claude · Parts Finder · Specifications are advisory — always verify with manufacturer documentation
        </p>
      </div>
    </div>
  );
}
