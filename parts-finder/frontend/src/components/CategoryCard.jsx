// ══════════════════════════════════════════════════════════════════════
// CategoryCard — Smart card that renders any of the 7 product categories.
//
// Accepts { type, data, isAiFallback } and switches on `type` to pick
// the right icon, accent color, and content layout.
//
// Category types:
//   oil | oil_filter | air_filter | cabin_filter | brakes | bulbs | coolant
// ══════════════════════════════════════════════════════════════════════

import React from "react";

// ── Category metadata ───────────────────────────────────────────────

const CATEGORY_META = {
  oil:          { label: "Engine Oil",    color: "#eab308" },
  oil_filter:   { label: "Oil Filter",   color: "#3b82f6" },
  air_filter:   { label: "Air Filter",   color: "#3b82f6" },
  cabin_filter: { label: "Cabin Filter", color: "#3b82f6" },
  brakes:       { label: "Brakes",       color: "#ef4444" },
  bulbs:        { label: "Bulbs",        color: "#f59e0b" },
  coolant:      { label: "Coolant",      color: "#06b6d4" },
};

// ── SVG Icons (inline, matching prototype style) ────────────────────

function OilDropIcon({ size = 24, color = "currentColor" }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none">
      <path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0L12 2.69z" fill={color} opacity=".15" />
      <path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0L12 2.69z" stroke={color} strokeWidth="1.5" strokeLinejoin="round" />
    </svg>
  );
}

function FilterIcon({ size = 24, color = "currentColor" }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M22 3H2l8 9.46V19l4 2v-8.54L22 3z" />
    </svg>
  );
}

function DiscIcon({ size = 24, color = "currentColor" }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="1.5">
      <circle cx="12" cy="12" r="10" />
      <circle cx="12" cy="12" r="3" />
      <line x1="12" y1="2" x2="12" y2="5" />
      <line x1="12" y1="19" x2="12" y2="22" />
      <line x1="2" y1="12" x2="5" y2="12" />
      <line x1="19" y1="12" x2="22" y2="12" />
    </svg>
  );
}

function BulbIcon({ size = 24, color = "currentColor" }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M9 18h6M10 22h4" />
      <path d="M12 2a7 7 0 0 0-4 12.7V17h8v-2.3A7 7 0 0 0 12 2z" />
    </svg>
  );
}

function ThermometerIcon({ size = 24, color = "currentColor" }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M14 14.76V3.5a2.5 2.5 0 0 0-5 0v11.26a4.5 4.5 0 1 0 5 0z" />
    </svg>
  );
}

const ICONS = {
  oil: OilDropIcon,
  oil_filter: FilterIcon,
  air_filter: FilterIcon,
  cabin_filter: FilterIcon,
  brakes: DiscIcon,
  bulbs: BulbIcon,
  coolant: ThermometerIcon,
};

// ── Shared mini-components ──────────────────────────────────────────

function Lbl({ children, color = "#64748b" }) {
  return (
    <span style={{
      fontSize: 11, fontWeight: 600, textTransform: "uppercase",
      letterSpacing: 1, color,
      fontFamily: "'Space Mono',monospace",
    }}>
      {children}
    </span>
  );
}

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

function PartRow({ label, partNumber, aftermarket = [] }) {
  if (!partNumber && aftermarket.length === 0) return null;
  return (
    <div style={{ marginTop: 10 }}>
      <Lbl>{label}</Lbl>
      {partNumber && (
        <p style={{ fontSize: 14, fontWeight: 600, margin: "4px 0 0", color: "#f1f5f9", fontFamily: "'Space Mono',monospace" }}>
          OEM: {partNumber}
        </p>
      )}
      {aftermarket.length > 0 && (
        <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 6 }}>
          {aftermarket.map((am, i) => (
            <Tag key={i}
              bg="rgba(255,255,255,.05)"
              color="#94a3b8"
              border="rgba(255,255,255,.1)"
            >
              {am.brand} {am.part_number}
            </Tag>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Content Renderers (one per category type) ───────────────────────

function OilContent({ data, accent }) {
  return (
    <>
      <p style={{
        fontSize: 28, fontWeight: 700, margin: "8px 0",
        fontFamily: "'Space Mono',monospace",
        color: accent, opacity: .9, letterSpacing: -.5,
      }}>
        {data.viscosity}
      </p>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
        {data.spec && <Tag>{data.spec}</Tag>}
        {data.oem_approval && data.oem_approval !== "None" && (
          <Tag bg="rgba(99,102,241,.1)" color="#a5b4fc" border="rgba(99,102,241,.2)">
            {data.oem_approval}
          </Tag>
        )}
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginTop: 14 }}>
        <div>
          <Lbl>Capacity</Lbl>
          <p style={{ fontSize: 16, fontWeight: 700, margin: "4px 0 0", color: "#f1f5f9" }}>
            ~{data.capacity_l}L
          </p>
        </div>
        <div>
          <Lbl>Interval</Lbl>
          <p style={{ fontSize: 16, fontWeight: 700, margin: "4px 0 0", color: "#f1f5f9" }}>
            {data.change_interval_km != null
              ? Number(data.change_interval_km).toLocaleString()
              : "—"} km
          </p>
        </div>
      </div>
      {data.confidence && (
        <div style={{ marginTop: 10 }}>
          <Tag
            bg={data.confidence === "high" ? "rgba(34,197,94,.1)" : "rgba(234,179,8,.08)"}
            color={data.confidence === "high" ? "#86efac" : "#eab308"}
            border={data.confidence === "high" ? "rgba(34,197,94,.2)" : "rgba(234,179,8,.15)"}
          >
            {data.confidence} confidence
          </Tag>
        </div>
      )}
    </>
  );
}

function FilterContent({ data }) {
  return (
    <>
      <PartRow label="OEM Part" partNumber={data.oem_part_number} />
      {data.aftermarket?.length > 0 && (
        <div style={{ marginTop: 10 }}>
          <Lbl>Aftermarket</Lbl>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 6 }}>
            {data.aftermarket.map((am, i) => (
              <Tag key={i}
                bg="rgba(255,255,255,.05)"
                color="#94a3b8"
                border="rgba(255,255,255,.1)"
              >
                {am.brand} {am.part_number}
              </Tag>
            ))}
          </div>
        </div>
      )}
    </>
  );
}

const BRAKE_POSITIONS = [
  { key: "front_pad",  label: "Front Pads" },
  { key: "rear_pad",   label: "Rear Pads" },
  { key: "front_disc", label: "Front Discs" },
  { key: "rear_disc",  label: "Rear Discs" },
];

function BrakesContent({ data }) {
  return (
    <>
      {BRAKE_POSITIONS.map(({ key, label }) => (
        <PartRow
          key={key}
          label={label}
          partNumber={data[`${key}_oem`]}
          aftermarket={data[`${key}_aftermarket`] || []}
        />
      ))}
      {data.brake_fluid_type && (
        <div style={{ marginTop: 12 }}>
          <Lbl>Brake Fluid</Lbl>
          <p style={{ fontSize: 14, fontWeight: 600, margin: "4px 0 0", color: "#f1f5f9", fontFamily: "'Space Mono',monospace" }}>
            {data.brake_fluid_type}
          </p>
        </div>
      )}
    </>
  );
}

const LAMP_POSITIONS = [
  { key: "low_beam",      label: "Low Beam" },
  { key: "high_beam",     label: "High Beam" },
  { key: "front_turn",    label: "Front Turn" },
  { key: "rear_turn",     label: "Rear Turn" },
  { key: "tail_brake",    label: "Tail / Brake" },
  { key: "reverse",       label: "Reverse" },
  { key: "fog",           label: "Fog" },
  { key: "license_plate", label: "License Plate" },
];

function BulbsContent({ data }) {
  const entries = LAMP_POSITIONS.filter(({ key }) => data[key]);
  return (
    <div style={{
      display: "grid", gridTemplateColumns: "1fr 1fr",
      gap: "8px 16px", marginTop: 6,
    }}>
      {entries.map(({ key, label }) => {
        const val = data[key];
        const isLed = /led/i.test(val);
        return (
          <div key={key}>
            <Lbl>{label}</Lbl>
            <p style={{
              fontSize: 14, fontWeight: 600, margin: "2px 0 0", color: "#f1f5f9",
              fontFamily: "'Space Mono',monospace",
              display: "flex", alignItems: "center", gap: 6,
            }}>
              {val}
              {isLed && (
                <span style={{
                  fontSize: 9, fontWeight: 700, padding: "2px 6px",
                  borderRadius: 4, background: "rgba(34,197,94,.12)",
                  border: "1px solid rgba(34,197,94,.2)", color: "#86efac",
                }}>
                  LED
                </span>
              )}
            </p>
          </div>
        );
      })}
    </div>
  );
}

function CoolantContent({ data }) {
  return (
    <>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginTop: 6 }}>
        {/* Color dot */}
        <span style={{
          width: 14, height: 14, borderRadius: "50%",
          background: coolantColorHex(data.color),
          border: "2px solid rgba(255,255,255,.15)",
          flexShrink: 0,
        }} />
        <p style={{ fontSize: 16, fontWeight: 700, margin: 0, color: "#f1f5f9" }}>
          {data.spec}
        </p>
      </div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 8 }}>
        {data.technology && (
          <Tag bg="rgba(6,182,212,.1)" color="#67e8f9" border="rgba(6,182,212,.2)">
            {data.technology}
          </Tag>
        )}
        {data.color && (
          <Tag bg="rgba(255,255,255,.05)" color="#94a3b8" border="rgba(255,255,255,.1)">
            {data.color}
          </Tag>
        )}
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginTop: 14 }}>
        <div>
          <Lbl>Capacity</Lbl>
          <p style={{ fontSize: 16, fontWeight: 700, margin: "4px 0 0", color: "#f1f5f9" }}>
            ~{data.capacity_l}L
          </p>
        </div>
        {data.aftermarket_match && (
          <div>
            <Lbl>Aftermarket</Lbl>
            <p style={{ fontSize: 14, fontWeight: 600, margin: "4px 0 0", color: "#94a3b8", fontFamily: "'Space Mono',monospace" }}>
              {data.aftermarket_match}
            </p>
          </div>
        )}
      </div>
      {data.mixing_warning && (
        <div style={{
          marginTop: 12, padding: "10px 14px", borderRadius: 8,
          background: "rgba(239,68,68,.06)", border: "1px solid rgba(239,68,68,.15)",
          fontSize: 12, color: "#fca5a5", lineHeight: 1.5,
        }}>
          {data.mixing_warning}
        </div>
      )}
    </>
  );
}

/** Map common coolant color names to hex for the dot indicator. */
function coolantColorHex(color) {
  if (!color) return "#64748b";
  const c = color.toLowerCase();
  if (c.includes("pink") || c.includes("red"))    return "#ef4444";
  if (c.includes("blue"))                          return "#3b82f6";
  if (c.includes("green"))                         return "#22c55e";
  if (c.includes("orange"))                        return "#f97316";
  if (c.includes("yellow"))                        return "#eab308";
  if (c.includes("purple") || c.includes("violet"))return "#a855f7";
  return "#64748b";
}

// ── Content router ──────────────────────────────────────────────────

const CONTENT_RENDERERS = {
  oil: OilContent,
  oil_filter: FilterContent,
  air_filter: FilterContent,
  cabin_filter: FilterContent,
  brakes: BrakesContent,
  bulbs: BulbsContent,
  coolant: CoolantContent,
};

// ── Main Component ──────────────────────────────────────────────────

/**
 * @param {{ type: string, data: Object, isAiFallback?: boolean }} props
 */
export default function CategoryCard({ type, data, isAiFallback = false }) {
  const meta = CATEGORY_META[type] || { label: type, color: "#94a3b8" };
  const accent = meta.color;
  const Icon = ICONS[type] || OilDropIcon;
  const ContentRenderer = CONTENT_RENDERERS[type];

  return (
    <div style={{
      background: `linear-gradient(135deg,${hexToRgba(accent, .06)},${hexToRgba(accent, .02)})`,
      borderRadius: 14,
      border: `1px solid ${hexToRgba(accent, .15)}`,
      padding: 24,
    }}>
      {/* Card header: icon + label + AI badge */}
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <div style={{
          width: 42, height: 42, borderRadius: 10,
          background: hexToRgba(accent, .12),
          display: "flex", alignItems: "center", justifyContent: "center",
          flexShrink: 0,
        }}>
          <Icon size={22} color={accent} />
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <Lbl color={accent}>{meta.label}</Lbl>
        </div>
        {isAiFallback && (
          <span style={{
            fontSize: 10, fontWeight: 600, padding: "3px 8px",
            borderRadius: 4, background: "rgba(168,85,247,.1)",
            border: "1px solid rgba(168,85,247,.2)", color: "#c084fc",
            fontFamily: "'Space Mono',monospace", whiteSpace: "nowrap",
          }}>
            AI-estimated
          </span>
        )}
      </div>

      {/* Category-specific content */}
      <div style={{ marginTop: 12 }}>
        {ContentRenderer
          ? <ContentRenderer data={data} accent={accent} />
          : <p style={{ color: "#64748b", fontSize: 13 }}>No renderer for type "{type}"</p>
        }
      </div>
    </div>
  );
}

// ── Utility ─────────────────────────────────────────────────────────

/** Convert hex color to rgba string. */
function hexToRgba(hex, alpha) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}
