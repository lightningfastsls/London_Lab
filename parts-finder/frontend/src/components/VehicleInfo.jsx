// ══════════════════════════════════════════════════════════════════════
// VehicleInfo — Vehicle identity card
// Shows make/model, year, engine code, fuel type, and plate badge.
// Design: reuses Card pattern from oil-finder prototype.
// ══════════════════════════════════════════════════════════════════════

import React from "react";

const Car = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
    <path d="M5 17h14M5 17a2 2 0 01-2-2v-4l2.4-4.8A2 2 0 017.19 5h9.62a2 2 0 011.79 1.2L21 11v4a2 2 0 01-2 2M5 17a2 2 0 002 2h1a2 2 0 002-2M14 17a2 2 0 002 2h1a2 2 0 002-2" />
  </svg>
);

/** Simple heuristic: does the string contain Hebrew characters? */
function hasHebrew(str) {
  return /[\u0590-\u05FF]/.test(str);
}

/** Wrap Hebrew text in RTL span, otherwise return as-is. */
function Bidi({ children }) {
  if (typeof children === "string" && hasHebrew(children)) {
    return <span dir="rtl">{children}</span>;
  }
  return <>{children}</>;
}

/**
 * @param {{ vehicle: { plate: string, make: string, model: string, year: number, engine_code: string, fuel_type: string } }} props
 */
export default function VehicleInfo({ vehicle }) {
  const { plate, make, model, year, engine_code, fuel_type } = vehicle;

  return (
    <div style={{
      background: "rgba(255,255,255,.03)",
      borderRadius: 14,
      border: "1px solid rgba(255,255,255,.07)",
      padding: "20px 24px",
    }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <Car />
          <span style={{
            fontSize: 11, fontWeight: 600, textTransform: "uppercase",
            letterSpacing: 1, color: "#64748b",
            fontFamily: "'Space Mono',monospace",
          }}>
            Vehicle Identified
          </span>
        </div>
        <span style={{
          fontSize: 12, fontWeight: 700, padding: "4px 10px",
          borderRadius: 6, background: "rgba(255,255,255,.06)",
          border: "1px solid rgba(255,255,255,.1)",
          color: "#94a3b8", fontFamily: "'Space Mono',monospace",
          letterSpacing: 1,
        }}>
          {plate}
        </span>
      </div>

      <p style={{ fontSize: 20, fontWeight: 700, margin: "10px 0 0", color: "#f1f5f9" }}>
        <Bidi>{make}</Bidi> <Bidi>{model}</Bidi>
      </p>
      <p style={{ fontSize: 13, color: "#64748b", margin: "4px 0 0" }}>
        {year} · {engine_code} · {fuel_type}
      </p>
    </div>
  );
}
