// ══════════════════════════════════════════════════════════════════════
// Parts Finder — API Client
// Single function that calls POST /api/plate-lookup and returns typed
// errors for each failure mode the backend can produce.
// ══════════════════════════════════════════════════════════════════════

const API_BASE_URL =
  typeof window !== "undefined" ? window.location.origin : "http://localhost:8000";

// ── Typed Errors ────────────────────────────────────────────────────

export class PlateFormatError extends Error {
  constructor(detail) {
    super(detail || "Invalid plate format");
    this.name = "PlateFormatError";
  }
}

export class PlateNotFoundError extends Error {
  constructor(detail) {
    super(detail || "Plate not found");
    this.name = "PlateNotFoundError";
  }
}

export class ServiceError extends Error {
  constructor(detail) {
    super(detail || "Government API unavailable");
    this.name = "ServiceError";
  }
}

// ── Main Lookup ─────────────────────────────────────────────────────

/**
 * Look up vehicle parts recommendations by Israeli license plate.
 *
 * @param {string} plate — 7-8 digit Israeli plate number
 * @returns {Promise<Object>} PlateLookupResponse from backend
 * @throws {PlateFormatError} on 400
 * @throws {PlateNotFoundError} on 404
 * @throws {ServiceError} on 503
 * @throws {Error} on any other failure
 */
export async function lookupPlate(plate) {
  let res;
  try {
    res = await fetch(`${API_BASE_URL}/api/plate-lookup`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ plate }),
    });
  } catch (err) {
    throw new ServiceError("Network error — check your connection");
  }

  if (res.ok) {
    return res.json();
  }

  // Parse error body (backend returns { detail: "..." })
  let detail;
  try {
    const body = await res.json();
    detail = body.detail;
  } catch {
    detail = undefined;
  }

  switch (res.status) {
    case 400:
      throw new PlateFormatError(detail);
    case 404:
      throw new PlateNotFoundError(detail);
    case 503:
      throw new ServiceError(detail);
    default:
      throw new Error(detail || `Unexpected error (${res.status})`);
  }
}
