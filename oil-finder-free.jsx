import { useState, useEffect } from "react";

// ══════════════════════════════════════════════════════════════════════
// OIL SPECIFICATION DATABASE — Israeli Market Vehicles
// Source: OEM owner's manuals & service documentation
// This replaces the Claude API call — zero cost per query
// ══════════════════════════════════════════════════════════════════════

const OIL_DB = {
  Toyota: {
    Corolla: {
      "1.2T Petrol": { viscosity: "0W-20", spec: "API SP", oem: "None", capacity: "4.2", interval: "15000" },
      "1.6 Petrol": { viscosity: "0W-20", spec: "API SP, ILSAC GF-6A", oem: "None", capacity: "3.9", interval: "15000" },
      "1.8 Petrol": { viscosity: "0W-20", spec: "API SP, ILSAC GF-6A", oem: "None", capacity: "4.2", interval: "15000" },
      "1.8 Hybrid": { viscosity: "0W-20", spec: "API SP, ILSAC GF-6A", oem: "None", capacity: "4.2", interval: "15000" },
      "2.0 Petrol": { viscosity: "0W-20", spec: "API SP, ILSAC GF-6A", oem: "None", capacity: "4.6", interval: "15000" },
      "2.0 Hybrid": { viscosity: "0W-20", spec: "API SP, ILSAC GF-6A", oem: "None", capacity: "4.6", interval: "15000" },
    },
    Camry: {
      "2.5 Petrol": { viscosity: "0W-20", spec: "API SP, ILSAC GF-6A", oem: "None", capacity: "4.8", interval: "15000" },
      "2.5 Hybrid": { viscosity: "0W-20", spec: "API SP, ILSAC GF-6A", oem: "None", capacity: "4.8", interval: "15000" },
      "3.5 V6": { viscosity: "0W-20", spec: "API SP, ILSAC GF-6A", oem: "None", capacity: "6.1", interval: "15000" },
    },
    Yaris: {
      "1.0 Petrol": { viscosity: "0W-20", spec: "API SP, ILSAC GF-6A", oem: "None", capacity: "3.4", interval: "15000" },
      "1.3 Petrol": { viscosity: "0W-20", spec: "API SP, ILSAC GF-6A", oem: "None", capacity: "3.4", interval: "15000" },
      "1.5 Petrol": { viscosity: "0W-20", spec: "API SP, ILSAC GF-6A", oem: "None", capacity: "3.6", interval: "15000" },
      "1.5 Hybrid": { viscosity: "0W-20", spec: "API SP, ILSAC GF-6A", oem: "None", capacity: "3.6", interval: "15000" },
    },
    RAV4: {
      "2.0 Petrol": { viscosity: "0W-20", spec: "API SP, ILSAC GF-6A", oem: "None", capacity: "4.4", interval: "15000" },
      "2.5 Petrol": { viscosity: "0W-20", spec: "API SP, ILSAC GF-6A", oem: "None", capacity: "4.8", interval: "15000" },
      "2.5 Hybrid": { viscosity: "0W-20", spec: "API SP, ILSAC GF-6A", oem: "None", capacity: "4.8", interval: "15000" },
      "2.5 PHEV": { viscosity: "0W-20", spec: "API SP, ILSAC GF-6A", oem: "None", capacity: "4.8", interval: "15000" },
    },
    "Land Cruiser": {
      "2.8 Diesel": { viscosity: "5W-30", spec: "API CK-4, ACEA C3", oem: "None", capacity: "7.7", interval: "10000" },
      "3.5 V6 Petrol": { viscosity: "0W-20", spec: "API SP", oem: "None", capacity: "6.4", interval: "15000" },
      "4.0 V6 Petrol": { viscosity: "5W-30", spec: "API SN", oem: "None", capacity: "5.7", interval: "15000" },
    },
    "C-HR": {
      "1.2T Petrol": { viscosity: "0W-20", spec: "API SP", oem: "None", capacity: "4.2", interval: "15000" },
      "1.8 Hybrid": { viscosity: "0W-20", spec: "API SP, ILSAC GF-6A", oem: "None", capacity: "4.2", interval: "15000" },
      "2.0 Hybrid": { viscosity: "0W-20", spec: "API SP, ILSAC GF-6A", oem: "None", capacity: "4.6", interval: "15000" },
    },
    Hilux: {
      "2.4 Diesel": { viscosity: "5W-30", spec: "API CK-4, ACEA C3", oem: "None", capacity: "6.5", interval: "10000" },
      "2.8 Diesel": { viscosity: "5W-30", spec: "API CK-4, ACEA C3", oem: "None", capacity: "7.7", interval: "10000" },
      "2.7 Petrol": { viscosity: "5W-30", spec: "API SP", oem: "None", capacity: "5.0", interval: "15000" },
      "4.0 V6 Petrol": { viscosity: "5W-30", spec: "API SN", oem: "None", capacity: "5.7", interval: "15000" },
    },
  },
  Hyundai: {
    Tucson: {
      "1.6T Petrol": { viscosity: "5W-30", spec: "API SP, ACEA A5/B5", oem: "Hyundai/Kia Oil Standard", capacity: "4.5", interval: "15000" },
      "1.6T Hybrid": { viscosity: "5W-30", spec: "API SP, ACEA A5/B5", oem: "Hyundai/Kia Oil Standard", capacity: "4.5", interval: "15000" },
      "2.0 Petrol": { viscosity: "5W-30", spec: "API SP, ACEA A5/B5", oem: "Hyundai/Kia Oil Standard", capacity: "4.2", interval: "15000" },
      "2.0 Diesel": { viscosity: "5W-30", spec: "ACEA C3", oem: "Hyundai/Kia DPF Oil", capacity: "5.3", interval: "15000" },
    },
    i20: {
      "1.0T Petrol": { viscosity: "5W-30", spec: "API SP, ACEA A5/B5", oem: "Hyundai/Kia Oil Standard", capacity: "3.5", interval: "15000" },
      "1.2 Petrol": { viscosity: "5W-30", spec: "API SP, ACEA A5/B5", oem: "Hyundai/Kia Oil Standard", capacity: "3.5", interval: "15000" },
      "1.4 Petrol": { viscosity: "5W-30", spec: "API SP, ACEA A5/B5", oem: "Hyundai/Kia Oil Standard", capacity: "3.6", interval: "15000" },
    },
    i30: {
      "1.0T Petrol": { viscosity: "5W-30", spec: "API SP, ACEA A5/B5", oem: "Hyundai/Kia Oil Standard", capacity: "3.5", interval: "15000" },
      "1.4T Petrol": { viscosity: "5W-30", spec: "API SP, ACEA A5/B5", oem: "Hyundai/Kia Oil Standard", capacity: "4.0", interval: "15000" },
      "1.5T Petrol": { viscosity: "5W-30", spec: "API SP, ACEA A5/B5", oem: "Hyundai/Kia Oil Standard", capacity: "4.0", interval: "15000" },
      "1.6 Diesel": { viscosity: "5W-30", spec: "ACEA C3", oem: "Hyundai/Kia DPF Oil", capacity: "4.8", interval: "15000" },
      "2.0T N": { viscosity: "5W-30", spec: "API SP, ACEA A5/B5", oem: "Hyundai/Kia Oil Standard", capacity: "4.5", interval: "10000" },
    },
    Kona: {
      "1.0T Petrol": { viscosity: "5W-30", spec: "API SP, ACEA A5/B5", oem: "Hyundai/Kia Oil Standard", capacity: "3.5", interval: "15000" },
      "1.6T Petrol": { viscosity: "5W-30", spec: "API SP, ACEA A5/B5", oem: "Hyundai/Kia Oil Standard", capacity: "4.5", interval: "15000" },
      "1.6 Hybrid": { viscosity: "5W-30", spec: "API SP, ACEA A5/B5", oem: "Hyundai/Kia Oil Standard", capacity: "4.0", interval: "15000" },
      "Electric": { viscosity: "N/A", spec: "N/A", oem: "N/A", capacity: "N/A", interval: "N/A", note: "Electric vehicle — no engine oil required. Requires coolant and brake fluid maintenance." },
    },
    "Santa Fe": {
      "1.6T Hybrid": { viscosity: "5W-30", spec: "API SP, ACEA A5/B5", oem: "Hyundai/Kia Oil Standard", capacity: "4.5", interval: "15000" },
      "2.0T Petrol": { viscosity: "5W-30", spec: "API SP, ACEA A5/B5", oem: "Hyundai/Kia Oil Standard", capacity: "4.8", interval: "15000" },
      "2.2 Diesel": { viscosity: "5W-30", spec: "ACEA C3", oem: "Hyundai/Kia DPF Oil", capacity: "6.3", interval: "15000" },
      "2.5 Petrol": { viscosity: "5W-30", spec: "API SP, ACEA A5/B5", oem: "Hyundai/Kia Oil Standard", capacity: "5.0", interval: "15000" },
    },
    Ioniq: {
      "1.6 Hybrid": { viscosity: "5W-30", spec: "API SP, ACEA A5/B5", oem: "Hyundai/Kia Oil Standard", capacity: "4.0", interval: "15000" },
      "1.6 PHEV": { viscosity: "5W-30", spec: "API SP, ACEA A5/B5", oem: "Hyundai/Kia Oil Standard", capacity: "4.0", interval: "15000" },
      "Electric": { viscosity: "N/A", spec: "N/A", oem: "N/A", capacity: "N/A", interval: "N/A", note: "Electric vehicle — no engine oil required." },
    },
  },
  Kia: {
    Sportage: {
      "1.6T Petrol": { viscosity: "5W-30", spec: "API SP, ACEA A5/B5", oem: "Hyundai/Kia Oil Standard", capacity: "4.5", interval: "15000" },
      "1.6T Hybrid": { viscosity: "5W-30", spec: "API SP, ACEA A5/B5", oem: "Hyundai/Kia Oil Standard", capacity: "4.5", interval: "15000" },
      "2.0 Petrol": { viscosity: "5W-30", spec: "API SP, ACEA A5/B5", oem: "Hyundai/Kia Oil Standard", capacity: "4.2", interval: "15000" },
      "2.0 Diesel": { viscosity: "5W-30", spec: "ACEA C3", oem: "Hyundai/Kia DPF Oil", capacity: "5.3", interval: "15000" },
    },
    Picanto: {
      "1.0 Petrol": { viscosity: "5W-30", spec: "API SP, ACEA A5/B5", oem: "Hyundai/Kia Oil Standard", capacity: "3.0", interval: "15000" },
      "1.2 Petrol": { viscosity: "5W-30", spec: "API SP, ACEA A5/B5", oem: "Hyundai/Kia Oil Standard", capacity: "3.3", interval: "15000" },
    },
    Ceed: {
      "1.0T Petrol": { viscosity: "5W-30", spec: "API SP, ACEA A5/B5", oem: "Hyundai/Kia Oil Standard", capacity: "3.5", interval: "15000" },
      "1.4T Petrol": { viscosity: "5W-30", spec: "API SP, ACEA A5/B5", oem: "Hyundai/Kia Oil Standard", capacity: "4.0", interval: "15000" },
      "1.5T Petrol": { viscosity: "5W-30", spec: "API SP, ACEA A5/B5", oem: "Hyundai/Kia Oil Standard", capacity: "4.0", interval: "15000" },
      "1.6 Diesel": { viscosity: "5W-30", spec: "ACEA C3", oem: "Hyundai/Kia DPF Oil", capacity: "4.8", interval: "15000" },
    },
    Niro: {
      "1.6 Hybrid": { viscosity: "5W-30", spec: "API SP, ACEA A5/B5", oem: "Hyundai/Kia Oil Standard", capacity: "4.0", interval: "15000" },
      "1.6 PHEV": { viscosity: "5W-30", spec: "API SP, ACEA A5/B5", oem: "Hyundai/Kia Oil Standard", capacity: "4.0", interval: "15000" },
      "Electric": { viscosity: "N/A", spec: "N/A", oem: "N/A", capacity: "N/A", interval: "N/A", note: "Electric vehicle — no engine oil required." },
    },
    Sorento: {
      "1.6T Hybrid": { viscosity: "5W-30", spec: "API SP, ACEA A5/B5", oem: "Hyundai/Kia Oil Standard", capacity: "4.5", interval: "15000" },
      "2.0T Petrol": { viscosity: "5W-30", spec: "API SP, ACEA A5/B5", oem: "Hyundai/Kia Oil Standard", capacity: "4.8", interval: "15000" },
      "2.2 Diesel": { viscosity: "5W-30", spec: "ACEA C3", oem: "Hyundai/Kia DPF Oil", capacity: "6.3", interval: "15000" },
      "2.5 Petrol": { viscosity: "5W-30", spec: "API SP, ACEA A5/B5", oem: "Hyundai/Kia Oil Standard", capacity: "5.0", interval: "15000" },
    },
  },
  Volkswagen: {
    Golf: {
      "1.0T Petrol": { viscosity: "5W-30", spec: "ACEA C3", oem: "VW 504.00/507.00", capacity: "4.0", interval: "15000" },
      "1.4T Petrol": { viscosity: "5W-30", spec: "ACEA C3", oem: "VW 504.00/507.00", capacity: "4.0", interval: "15000" },
      "1.5T Petrol": { viscosity: "5W-30", spec: "ACEA C3", oem: "VW 504.00/507.00", capacity: "4.0", interval: "15000" },
      "2.0T GTI": { viscosity: "5W-30", spec: "ACEA C3", oem: "VW 504.00/507.00", capacity: "4.6", interval: "15000" },
      "2.0T R": { viscosity: "5W-40", spec: "ACEA C3", oem: "VW 504.00/507.00", capacity: "4.6", interval: "15000" },
      "1.6 Diesel": { viscosity: "5W-30", spec: "ACEA C3", oem: "VW 507.00", capacity: "4.3", interval: "15000" },
      "2.0 Diesel": { viscosity: "5W-30", spec: "ACEA C3", oem: "VW 507.00", capacity: "4.7", interval: "15000" },
    },
    Polo: {
      "1.0 Petrol": { viscosity: "5W-30", spec: "ACEA C3", oem: "VW 504.00/507.00", capacity: "3.6", interval: "15000" },
      "1.0T Petrol": { viscosity: "5W-30", spec: "ACEA C3", oem: "VW 504.00/507.00", capacity: "3.6", interval: "15000" },
      "1.4 Diesel": { viscosity: "5W-30", spec: "ACEA C3", oem: "VW 507.00", capacity: "3.8", interval: "15000" },
      "2.0T GTI": { viscosity: "5W-30", spec: "ACEA C3", oem: "VW 504.00/507.00", capacity: "4.6", interval: "10000" },
    },
    Tiguan: {
      "1.4T Petrol": { viscosity: "5W-30", spec: "ACEA C3", oem: "VW 504.00/507.00", capacity: "4.0", interval: "15000" },
      "1.5T Petrol": { viscosity: "5W-30", spec: "ACEA C3", oem: "VW 504.00/507.00", capacity: "4.0", interval: "15000" },
      "2.0T Petrol": { viscosity: "5W-30", spec: "ACEA C3", oem: "VW 504.00/507.00", capacity: "4.6", interval: "15000" },
      "2.0 Diesel": { viscosity: "5W-30", spec: "ACEA C3", oem: "VW 507.00", capacity: "5.0", interval: "15000" },
    },
    "T-Cross": {
      "1.0T Petrol": { viscosity: "5W-30", spec: "ACEA C3", oem: "VW 504.00/507.00", capacity: "3.6", interval: "15000" },
      "1.5T Petrol": { viscosity: "5W-30", spec: "ACEA C3", oem: "VW 504.00/507.00", capacity: "4.0", interval: "15000" },
    },
    Passat: {
      "1.4T Petrol": { viscosity: "5W-30", spec: "ACEA C3", oem: "VW 504.00/507.00", capacity: "4.0", interval: "15000" },
      "1.5T Petrol": { viscosity: "5W-30", spec: "ACEA C3", oem: "VW 504.00/507.00", capacity: "4.0", interval: "15000" },
      "2.0T Petrol": { viscosity: "5W-30", spec: "ACEA C3", oem: "VW 504.00/507.00", capacity: "4.6", interval: "15000" },
      "1.6 Diesel": { viscosity: "5W-30", spec: "ACEA C3", oem: "VW 507.00", capacity: "4.3", interval: "15000" },
      "2.0 Diesel": { viscosity: "5W-30", spec: "ACEA C3", oem: "VW 507.00", capacity: "5.0", interval: "15000" },
    },
    Touareg: {
      "3.0 V6 Petrol": { viscosity: "5W-30", spec: "ACEA C3", oem: "VW 504.00/507.00", capacity: "7.5", interval: "15000" },
      "3.0 V6 Diesel": { viscosity: "5W-30", spec: "ACEA C3", oem: "VW 507.00", capacity: "8.0", interval: "15000" },
      "3.0 V6 eHybrid": { viscosity: "5W-30", spec: "ACEA C3", oem: "VW 504.00/507.00", capacity: "7.5", interval: "15000" },
    },
  },
  "Škoda": {
    Octavia: {
      "1.0T Petrol": { viscosity: "5W-30", spec: "ACEA C3", oem: "VW 504.00/507.00", capacity: "4.0", interval: "15000" },
      "1.4T Petrol": { viscosity: "5W-30", spec: "ACEA C3", oem: "VW 504.00/507.00", capacity: "4.0", interval: "15000" },
      "1.5T Petrol": { viscosity: "5W-30", spec: "ACEA C3", oem: "VW 504.00/507.00", capacity: "4.0", interval: "15000" },
      "2.0T Petrol": { viscosity: "5W-30", spec: "ACEA C3", oem: "VW 504.00/507.00", capacity: "4.6", interval: "15000" },
      "1.6 Diesel": { viscosity: "5W-30", spec: "ACEA C3", oem: "VW 507.00", capacity: "4.3", interval: "15000" },
      "2.0 Diesel": { viscosity: "5W-30", spec: "ACEA C3", oem: "VW 507.00", capacity: "4.7", interval: "15000" },
    },
    Fabia: {
      "1.0 Petrol": { viscosity: "5W-30", spec: "ACEA C3", oem: "VW 504.00/507.00", capacity: "3.6", interval: "15000" },
      "1.0T Petrol": { viscosity: "5W-30", spec: "ACEA C3", oem: "VW 504.00/507.00", capacity: "3.6", interval: "15000" },
      "1.4 Diesel": { viscosity: "5W-30", spec: "ACEA C3", oem: "VW 507.00", capacity: "3.8", interval: "15000" },
    },
    Kodiaq: {
      "1.4T Petrol": { viscosity: "5W-30", spec: "ACEA C3", oem: "VW 504.00/507.00", capacity: "4.0", interval: "15000" },
      "1.5T Petrol": { viscosity: "5W-30", spec: "ACEA C3", oem: "VW 504.00/507.00", capacity: "4.0", interval: "15000" },
      "2.0T Petrol": { viscosity: "5W-30", spec: "ACEA C3", oem: "VW 504.00/507.00", capacity: "4.6", interval: "15000" },
      "2.0 Diesel": { viscosity: "5W-30", spec: "ACEA C3", oem: "VW 507.00", capacity: "5.0", interval: "15000" },
    },
    Karoq: {
      "1.0T Petrol": { viscosity: "5W-30", spec: "ACEA C3", oem: "VW 504.00/507.00", capacity: "3.6", interval: "15000" },
      "1.5T Petrol": { viscosity: "5W-30", spec: "ACEA C3", oem: "VW 504.00/507.00", capacity: "4.0", interval: "15000" },
      "2.0 Diesel": { viscosity: "5W-30", spec: "ACEA C3", oem: "VW 507.00", capacity: "5.0", interval: "15000" },
    },
    Superb: {
      "1.4T Petrol": { viscosity: "5W-30", spec: "ACEA C3", oem: "VW 504.00/507.00", capacity: "4.0", interval: "15000" },
      "1.5T Petrol": { viscosity: "5W-30", spec: "ACEA C3", oem: "VW 504.00/507.00", capacity: "4.0", interval: "15000" },
      "2.0T Petrol": { viscosity: "5W-30", spec: "ACEA C3", oem: "VW 504.00/507.00", capacity: "4.6", interval: "15000" },
      "1.6 Diesel": { viscosity: "5W-30", spec: "ACEA C3", oem: "VW 507.00", capacity: "4.3", interval: "15000" },
      "2.0 Diesel": { viscosity: "5W-30", spec: "ACEA C3", oem: "VW 507.00", capacity: "5.0", interval: "15000" },
    },
  },
  BMW: {
    "3 Series": {
      "318i 1.5T": { viscosity: "0W-30", spec: "ACEA C3", oem: "BMW LL-17 FE+", capacity: "4.0", interval: "15000" },
      "320i 2.0T": { viscosity: "0W-30", spec: "ACEA C3", oem: "BMW LL-17 FE+", capacity: "4.6", interval: "15000" },
      "330i 2.0T": { viscosity: "0W-30", spec: "ACEA C3", oem: "BMW LL-17 FE+", capacity: "4.6", interval: "15000" },
      "330e PHEV": { viscosity: "0W-30", spec: "ACEA C3", oem: "BMW LL-17 FE+", capacity: "4.6", interval: "15000" },
      "M340i 3.0T": { viscosity: "0W-30", spec: "ACEA C3", oem: "BMW LL-04", capacity: "6.5", interval: "15000" },
      "318d 2.0D": { viscosity: "5W-30", spec: "ACEA C3", oem: "BMW LL-04", capacity: "5.2", interval: "15000" },
      "320d 2.0D": { viscosity: "5W-30", spec: "ACEA C3", oem: "BMW LL-04", capacity: "5.2", interval: "15000" },
      "330d 3.0D": { viscosity: "5W-30", spec: "ACEA C3", oem: "BMW LL-04", capacity: "6.5", interval: "15000" },
    },
    "5 Series": {
      "520i 2.0T": { viscosity: "0W-30", spec: "ACEA C3", oem: "BMW LL-17 FE+", capacity: "4.6", interval: "15000" },
      "530i 2.0T": { viscosity: "0W-30", spec: "ACEA C3", oem: "BMW LL-17 FE+", capacity: "4.6", interval: "15000" },
      "530e PHEV": { viscosity: "0W-30", spec: "ACEA C3", oem: "BMW LL-17 FE+", capacity: "4.6", interval: "15000" },
      "540i 3.0T": { viscosity: "0W-30", spec: "ACEA C3", oem: "BMW LL-04", capacity: "6.5", interval: "15000" },
      "520d 2.0D": { viscosity: "5W-30", spec: "ACEA C3", oem: "BMW LL-04", capacity: "5.2", interval: "15000" },
      "530d 3.0D": { viscosity: "5W-30", spec: "ACEA C3", oem: "BMW LL-04", capacity: "6.5", interval: "15000" },
    },
    X1: {
      "sDrive18i 1.5T": { viscosity: "0W-30", spec: "ACEA C3", oem: "BMW LL-17 FE+", capacity: "4.0", interval: "15000" },
      "sDrive20i 2.0T": { viscosity: "0W-30", spec: "ACEA C3", oem: "BMW LL-17 FE+", capacity: "4.6", interval: "15000" },
      "xDrive25e PHEV": { viscosity: "0W-30", spec: "ACEA C3", oem: "BMW LL-17 FE+", capacity: "4.6", interval: "15000" },
      "sDrive18d 2.0D": { viscosity: "5W-30", spec: "ACEA C3", oem: "BMW LL-04", capacity: "5.2", interval: "15000" },
      "xDrive20d 2.0D": { viscosity: "5W-30", spec: "ACEA C3", oem: "BMW LL-04", capacity: "5.2", interval: "15000" },
    },
    X3: {
      "xDrive20i 2.0T": { viscosity: "0W-30", spec: "ACEA C3", oem: "BMW LL-17 FE+", capacity: "4.6", interval: "15000" },
      "xDrive30i 2.0T": { viscosity: "0W-30", spec: "ACEA C3", oem: "BMW LL-17 FE+", capacity: "4.6", interval: "15000" },
      "xDrive30e PHEV": { viscosity: "0W-30", spec: "ACEA C3", oem: "BMW LL-17 FE+", capacity: "4.6", interval: "15000" },
      "M40i 3.0T": { viscosity: "0W-30", spec: "ACEA C3", oem: "BMW LL-04", capacity: "6.5", interval: "15000" },
      "xDrive20d 2.0D": { viscosity: "5W-30", spec: "ACEA C3", oem: "BMW LL-04", capacity: "5.2", interval: "15000" },
      "xDrive30d 3.0D": { viscosity: "5W-30", spec: "ACEA C3", oem: "BMW LL-04", capacity: "6.5", interval: "15000" },
    },
    X5: {
      "xDrive40i 3.0T": { viscosity: "0W-30", spec: "ACEA C3", oem: "BMW LL-04", capacity: "6.5", interval: "15000" },
      "xDrive45e PHEV": { viscosity: "0W-30", spec: "ACEA C3", oem: "BMW LL-04", capacity: "6.5", interval: "15000" },
      "M50i 4.4T V8": { viscosity: "0W-30", spec: "ACEA C3", oem: "BMW LL-04", capacity: "8.0", interval: "15000" },
      "xDrive25d 2.0D": { viscosity: "5W-30", spec: "ACEA C3", oem: "BMW LL-04", capacity: "5.2", interval: "15000" },
      "xDrive30d 3.0D": { viscosity: "5W-30", spec: "ACEA C3", oem: "BMW LL-04", capacity: "6.5", interval: "15000" },
      "xDrive40d 3.0D": { viscosity: "5W-30", spec: "ACEA C3", oem: "BMW LL-04", capacity: "6.5", interval: "15000" },
    },
  },
  "Mercedes-Benz": {
    "A-Class": {
      "A180 1.3T": { viscosity: "5W-30", spec: "ACEA C3", oem: "MB 229.52", capacity: "4.5", interval: "15000" },
      "A200 1.3T": { viscosity: "5W-30", spec: "ACEA C3", oem: "MB 229.52", capacity: "4.5", interval: "15000" },
      "A250 2.0T": { viscosity: "5W-30", spec: "ACEA C3", oem: "MB 229.52", capacity: "5.0", interval: "15000" },
      "A35 AMG 2.0T": { viscosity: "5W-30", spec: "ACEA C3", oem: "MB 229.52", capacity: "5.0", interval: "10000" },
      "A180d 1.5D": { viscosity: "5W-30", spec: "ACEA C3", oem: "MB 229.52", capacity: "5.0", interval: "15000" },
    },
    "C-Class": {
      "C180 1.5T": { viscosity: "5W-30", spec: "ACEA C3", oem: "MB 229.52", capacity: "5.0", interval: "15000" },
      "C200 1.5T/2.0T": { viscosity: "5W-30", spec: "ACEA C3", oem: "MB 229.52", capacity: "5.0", interval: "15000" },
      "C300 2.0T": { viscosity: "5W-30", spec: "ACEA C3", oem: "MB 229.52", capacity: "5.5", interval: "15000" },
      "C43 AMG 3.0T V6": { viscosity: "5W-30", spec: "ACEA C3", oem: "MB 229.52", capacity: "6.5", interval: "10000" },
      "C220d 2.0D": { viscosity: "5W-30", spec: "ACEA C3", oem: "MB 229.52", capacity: "5.5", interval: "15000" },
      "C300d 2.0D": { viscosity: "5W-30", spec: "ACEA C3", oem: "MB 229.52", capacity: "5.5", interval: "15000" },
    },
    "E-Class": {
      "E200 2.0T": { viscosity: "5W-30", spec: "ACEA C3", oem: "MB 229.52", capacity: "5.5", interval: "15000" },
      "E300 2.0T": { viscosity: "5W-30", spec: "ACEA C3", oem: "MB 229.52", capacity: "5.5", interval: "15000" },
      "E450 3.0T": { viscosity: "5W-30", spec: "ACEA C3", oem: "MB 229.52", capacity: "7.0", interval: "15000" },
      "E53 AMG 3.0T": { viscosity: "5W-30", spec: "ACEA C3", oem: "MB 229.52", capacity: "7.0", interval: "10000" },
      "E220d 2.0D": { viscosity: "5W-30", spec: "ACEA C3", oem: "MB 229.52", capacity: "5.5", interval: "15000" },
      "E300d 2.0D": { viscosity: "5W-30", spec: "ACEA C3", oem: "MB 229.52", capacity: "5.5", interval: "15000" },
      "E400d 3.0D": { viscosity: "5W-30", spec: "ACEA C3", oem: "MB 229.52", capacity: "7.0", interval: "15000" },
    },
    GLC: {
      "GLC200 2.0T": { viscosity: "5W-30", spec: "ACEA C3", oem: "MB 229.52", capacity: "5.5", interval: "15000" },
      "GLC300 2.0T": { viscosity: "5W-30", spec: "ACEA C3", oem: "MB 229.52", capacity: "5.5", interval: "15000" },
      "GLC300e PHEV": { viscosity: "5W-30", spec: "ACEA C3", oem: "MB 229.52", capacity: "5.5", interval: "15000" },
      "GLC43 AMG 3.0T": { viscosity: "5W-30", spec: "ACEA C3", oem: "MB 229.52", capacity: "6.5", interval: "10000" },
      "GLC220d 2.0D": { viscosity: "5W-30", spec: "ACEA C3", oem: "MB 229.52", capacity: "5.5", interval: "15000" },
      "GLC300d 2.0D": { viscosity: "5W-30", spec: "ACEA C3", oem: "MB 229.52", capacity: "5.5", interval: "15000" },
    },
    GLE: {
      "GLE350 2.0T": { viscosity: "5W-30", spec: "ACEA C3", oem: "MB 229.52", capacity: "5.5", interval: "15000" },
      "GLE450 3.0T": { viscosity: "5W-30", spec: "ACEA C3", oem: "MB 229.52", capacity: "7.0", interval: "15000" },
      "GLE350de PHEV": { viscosity: "5W-30", spec: "ACEA C3", oem: "MB 229.52", capacity: "7.0", interval: "15000" },
      "GLE53 AMG 3.0T": { viscosity: "5W-30", spec: "ACEA C3", oem: "MB 229.52", capacity: "7.0", interval: "10000" },
      "GLE300d 2.0D": { viscosity: "5W-30", spec: "ACEA C3", oem: "MB 229.52", capacity: "5.5", interval: "15000" },
      "GLE350d 3.0D": { viscosity: "5W-30", spec: "ACEA C3", oem: "MB 229.52", capacity: "7.0", interval: "15000" },
      "GLE400d 3.0D": { viscosity: "5W-30", spec: "ACEA C3", oem: "MB 229.52", capacity: "7.0", interval: "15000" },
    },
  },
  Mazda: {
    "3": {
      "1.5 Petrol": { viscosity: "0W-20", spec: "API SP, ILSAC GF-6A", oem: "Mazda Original Oil Ultra", capacity: "4.2", interval: "15000" },
      "2.0 Petrol": { viscosity: "0W-20", spec: "API SP, ILSAC GF-6A", oem: "Mazda Original Oil Ultra", capacity: "4.2", interval: "15000" },
      "2.0 e-Skyactiv X": { viscosity: "0W-20", spec: "API SP", oem: "Mazda Original Oil Ultra", capacity: "4.5", interval: "15000" },
      "2.5 Petrol": { viscosity: "0W-20", spec: "API SP, ILSAC GF-6A", oem: "Mazda Original Oil Ultra", capacity: "4.5", interval: "15000" },
    },
    "CX-5": {
      "2.0 Petrol": { viscosity: "0W-20", spec: "API SP, ILSAC GF-6A", oem: "Mazda Original Oil Ultra", capacity: "4.2", interval: "15000" },
      "2.5 Petrol": { viscosity: "0W-20", spec: "API SP", oem: "Mazda Original Oil Ultra", capacity: "4.5", interval: "15000" },
      "2.2 Diesel": { viscosity: "5W-30", spec: "ACEA C3", oem: "Mazda Original Oil DPF", capacity: "5.1", interval: "12500" },
    },
    "CX-30": {
      "2.0 Petrol": { viscosity: "0W-20", spec: "API SP, ILSAC GF-6A", oem: "Mazda Original Oil Ultra", capacity: "4.2", interval: "15000" },
      "2.0 e-Skyactiv X": { viscosity: "0W-20", spec: "API SP", oem: "Mazda Original Oil Ultra", capacity: "4.5", interval: "15000" },
      "1.8 Diesel": { viscosity: "5W-30", spec: "ACEA C3", oem: "Mazda Original Oil DPF", capacity: "4.7", interval: "12500" },
    },
    "CX-60": {
      "2.5 PHEV": { viscosity: "0W-20", spec: "API SP", oem: "Mazda Original Oil Ultra", capacity: "4.5", interval: "15000" },
      "3.3 Diesel": { viscosity: "5W-30", spec: "ACEA C3", oem: "Mazda Original Oil DPF", capacity: "6.5", interval: "12500" },
    },
  },
  Suzuki: {
    Swift: {
      "1.2 Petrol": { viscosity: "0W-20", spec: "API SP", oem: "None", capacity: "3.2", interval: "15000" },
      "1.2 Mild Hybrid": { viscosity: "0W-20", spec: "API SP", oem: "None", capacity: "3.2", interval: "15000" },
      "1.4T Sport": { viscosity: "5W-30", spec: "API SP", oem: "None", capacity: "3.5", interval: "15000" },
    },
    Vitara: {
      "1.4T Petrol": { viscosity: "5W-30", spec: "API SP", oem: "None", capacity: "3.6", interval: "15000" },
      "1.4T Hybrid": { viscosity: "5W-30", spec: "API SP", oem: "None", capacity: "3.6", interval: "15000" },
      "1.6 Petrol": { viscosity: "0W-20", spec: "API SP", oem: "None", capacity: "3.9", interval: "15000" },
    },
    Jimny: {
      "1.5 Petrol": { viscosity: "5W-30", spec: "API SP", oem: "None", capacity: "3.5", interval: "15000" },
    },
    Baleno: {
      "1.0T Petrol": { viscosity: "5W-30", spec: "API SP", oem: "None", capacity: "3.2", interval: "15000" },
      "1.2 Petrol": { viscosity: "0W-20", spec: "API SP", oem: "None", capacity: "3.2", interval: "15000" },
    },
  },
};

// Derive structure
const makes = Object.keys(OIL_DB).sort();
const getModels = (make) => Object.keys(OIL_DB[make] || {}).sort();
const getEngines = (make, model) => Object.keys(OIL_DB[make]?.[model] || {});

// Placeholder catalog
const CATALOG = {
  "0W-20": "TBD-0W20", "0W-30": "TBD-0W30", "5W-20": "TBD-5W20",
  "5W-30": "TBD-5W30", "5W-40": "TBD-5W40", "10W-40": "TBD-10W40",
};

// ── Icons ─────────────────────────────────────────────────────────────
const OilDrop = ({ s = 24, c = "currentColor" }) => (
  <svg width={s} height={s} viewBox="0 0 24 24" fill="none">
    <path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0L12 2.69z" fill={c} opacity=".15" />
    <path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0L12 2.69z" stroke={c} strokeWidth="1.5" strokeLinejoin="round" />
  </svg>
);
const Chev = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="6 9 12 15 18 9" /></svg>
);
const Car = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
    <path d="M5 17h14M5 17a2 2 0 01-2-2v-4l2.4-4.8A2 2 0 017.19 5h9.62a2 2 0 011.79 1.2L21 11v4a2 2 0 01-2 2M5 17a2 2 0 002 2h1a2 2 0 002-2M14 17a2 2 0 002 2h1a2 2 0 002-2" />
  </svg>
);
const Zap = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#22c55e" strokeWidth="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" /></svg>
);

// ── Main App ──────────────────────────────────────────────────────────
export default function App() {
  const [make, setMake] = useState("");
  const [model, setModel] = useState("");
  const [engine, setEngine] = useState("");
  const [result, setResult] = useState(null);

  const models = make ? getModels(make) : [];
  const engines = make && model ? getEngines(make, model) : [];

  useEffect(() => { setModel(""); setEngine(""); setResult(null); }, [make]);
  useEffect(() => { setEngine(""); setResult(null); }, [model]);
  useEffect(() => { setResult(null); }, [engine]);

  const search = () => {
    const data = OIL_DB[make]?.[model]?.[engine];
    if (data) setResult({ make, model, engine, ...data });
  };

  const cat = result ? CATALOG[result.viscosity] : null;

  return (
    <div style={{ minHeight: "100vh", background: "linear-gradient(145deg,#0a0f1a,#111927,#0d1520)", fontFamily: "'DM Sans','Segoe UI',system-ui,sans-serif", color: "#e2e8f0", position: "relative", overflow: "hidden" }}>
      <div style={{ position: "fixed", inset: 0, opacity: .03, backgroundImage: "radial-gradient(circle at 1px 1px,white 1px,transparent 0)", backgroundSize: "40px 40px" }} />
      <div style={{ position: "fixed", top: -200, right: -200, width: 600, height: 600, background: "radial-gradient(circle,rgba(234,179,8,.06),transparent 70%)" }} />
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');
        @keyframes fadeUp{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}
        *{box-sizing:border-box} select,input{outline:none}
      `}</style>

      <div style={{ position: "relative", zIndex: 1, maxWidth: 680, margin: "0 auto", padding: "40px 20px 60px" }}>
        {/* Header */}
        <div style={{ textAlign: "center", marginBottom: 40, animation: "fadeUp .5s ease" }}>
          <div style={{ display: "inline-flex", alignItems: "center", gap: 10, background: "rgba(234,179,8,.08)", border: "1px solid rgba(234,179,8,.15)", borderRadius: 40, padding: "6px 16px 6px 10px", marginBottom: 20, fontSize: 12, fontWeight: 500, color: "#eab308", letterSpacing: .5, textTransform: "uppercase", fontFamily: "'Space Mono',monospace" }}>
            <OilDrop s={16} c="#eab308" /> Cloudy Claude · Oil Finder
          </div>
          <h1 style={{ fontSize: "clamp(28px,5vw,40px)", fontWeight: 700, margin: "0 0 8px", letterSpacing: -.5, background: "linear-gradient(135deg,#f8fafc,#94a3b8)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
            Find the Right Oil
          </h1>
          <p style={{ fontSize: 15, color: "#64748b", margin: 0, maxWidth: 440, marginLeft: "auto", marginRight: "auto", lineHeight: 1.5 }}>
            Select your vehicle and get the exact engine oil specification instantly
          </p>
          <div style={{ display: "inline-flex", alignItems: "center", gap: 6, marginTop: 12, padding: "4px 12px", borderRadius: 20, background: "rgba(34,197,94,.08)", border: "1px solid rgba(34,197,94,.15)", fontSize: 11, color: "#22c55e", fontWeight: 600 }}>
            <Zap /> Free · No API calls · Instant results
          </div>
        </div>

        {/* Form */}
        <div style={{ background: "rgba(255,255,255,.03)", borderRadius: 16, border: "1px solid rgba(255,255,255,.07)", padding: 28, animation: "fadeUp .5s ease .15s both" }}>
          <div style={{ display: "grid", gap: 16 }}>
            <Sel label="Make" value={make} onChange={setMake} options={makes} placeholder="Select make..." />
            <Sel label="Model" value={model} onChange={setModel} options={models} placeholder="Select model..." disabled={!make} dim={!make} />
            <Sel label="Engine" value={engine} onChange={setEngine} options={engines} placeholder="Select engine..." disabled={!model} dim={!model} />
          </div>
          <button onClick={search} disabled={!engine}
            style={{ width: "100%", marginTop: 24, padding: "15px 24px", borderRadius: 10, border: "none", cursor: engine ? "pointer" : "not-allowed", fontSize: 14, fontWeight: 700, fontFamily: "inherit", display: "flex", alignItems: "center", justifyContent: "center", gap: 10, background: engine ? "linear-gradient(135deg,#eab308,#ca8a04)" : "rgba(255,255,255,.06)", color: engine ? "#0a0f1a" : "#475569", transition: "all .3s" }}>
            <OilDrop s={18} c={engine ? "#0a0f1a" : "#475569"} /> Find Oil Specification
          </button>
        </div>

        {/* Results */}
        {result && (
          <div style={{ marginTop: 24, animation: "fadeUp .4s ease" }}>
            {/* Vehicle */}
            <Card>
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <Car />
                <Lbl>Vehicle Identified</Lbl>
              </div>
              <p style={{ fontSize: 20, fontWeight: 700, margin: "8px 0 0", color: "#f1f5f9" }}>{result.make} {result.model}</p>
              <p style={{ fontSize: 13, color: "#64748b", margin: "4px 0 0" }}>{result.engine}</p>
            </Card>

            {result.viscosity === "N/A" ? (
              <div style={{ marginTop: 12, padding: "20px 24px", borderRadius: 14, background: "rgba(34,197,94,.06)", border: "1px solid rgba(34,197,94,.15)", color: "#86efac", fontSize: 14 }}>
                <strong>Electric Vehicle</strong> — {result.note || "No engine oil required. Requires coolant and brake fluid maintenance."}
              </div>
            ) : (
              <>
                {/* Oil Spec */}
                <div style={{ marginTop: 12, background: "linear-gradient(135deg,rgba(234,179,8,.06),rgba(234,179,8,.02))", borderRadius: 14, border: "1px solid rgba(234,179,8,.15)", padding: 24 }}>
                  <div style={{ display: "flex", alignItems: "flex-start", gap: 16 }}>
                    <div style={{ width: 52, height: 52, borderRadius: 12, background: "rgba(234,179,8,.12)", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                      <OilDrop s={28} c="#eab308" />
                    </div>
                    <div style={{ flex: 1 }}>
                      <Lbl color="#eab308">Recommended Oil</Lbl>
                      <p style={{ fontSize: 28, fontWeight: 700, margin: "6px 0", fontFamily: "'Space Mono',monospace", color: "#fef9c3", letterSpacing: -.5 }}>{result.viscosity}</p>
                      <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 4 }}>
                        {result.spec !== "None" && <Tag>{result.spec}</Tag>}
                        {result.oem !== "None" && <Tag bg="rgba(99,102,241,.1)" color="#a5b4fc" border="rgba(99,102,241,.2)">{result.oem}</Tag>}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Details */}
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginTop: 12 }}>
                  <Card><Lbl>Oil Capacity</Lbl><p style={{ fontSize: 20, fontWeight: 700, margin: "6px 0 2px", color: "#f1f5f9" }}>~{result.capacity}L</p><span style={{ fontSize: 11, color: "#475569" }}>with filter</span></Card>
                  <Card><Lbl>Change Interval</Lbl><p style={{ fontSize: 20, fontWeight: 700, margin: "6px 0 2px", color: "#f1f5f9" }}>{Number(result.interval).toLocaleString()} km</p><span style={{ fontSize: 11, color: "#475569" }}>or 12 months</span></Card>
                </div>

                {/* Catalog */}
                <Card style={{ marginTop: 12 }}>
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                    <div>
                      <Lbl>Our Catalog Number</Lbl>
                      <p style={{ fontSize: 18, fontWeight: 700, margin: "6px 0 0", fontFamily: "'Space Mono',monospace", color: cat?.startsWith("TBD") ? "#475569" : "#f1f5f9" }}>{cat || "—"}</p>
                    </div>
                    <div style={{ padding: "6px 12px", borderRadius: 6, background: "rgba(234,179,8,.08)", border: "1px solid rgba(234,179,8,.15)", fontSize: 11, color: "#eab308", fontWeight: 600 }}>
                      Catalog # pending
                    </div>
                  </div>
                </Card>
              </>
            )}
          </div>
        )}

        <p style={{ textAlign: "center", marginTop: 48, fontSize: 11, color: "#334155", fontFamily: "'Space Mono',monospace" }}>
          Cloudy Claude · Static Database · Specifications are advisory — always verify with manufacturer documentation
        </p>
      </div>
    </div>
  );
}

// ── Sub-components ────────────────────────────────────────────────────
function Sel({ label, value, onChange, options, placeholder, disabled, dim }) {
  return (
    <div style={{ opacity: dim ? .35 : 1, transition: "opacity .2s" }}>
      <label style={{ fontSize: 11, fontWeight: 600, textTransform: "uppercase", letterSpacing: 1, color: "#64748b", marginBottom: 8, display: "block", fontFamily: "'Space Mono',monospace" }}>{label}</label>
      <div style={{ position: "relative" }}>
        <select value={value} onChange={e => onChange(e.target.value)} disabled={disabled}
          style={{ width: "100%", padding: "13px 40px 13px 16px", borderRadius: 10, border: "1px solid rgba(255,255,255,.1)", background: "rgba(0,0,0,.3)", color: "#f8fafc", fontSize: 14, fontFamily: "inherit", appearance: "none", cursor: disabled ? "not-allowed" : "pointer" }}>
          <option value="">{placeholder}</option>
          {options.map(o => <option key={o} value={o}>{o}</option>)}
        </select>
        <div style={{ position: "absolute", right: 14, top: "50%", transform: "translateY(-50%)", color: "#475569", pointerEvents: "none" }}><Chev /></div>
      </div>
    </div>
  );
}

function Card({ children, style = {} }) {
  return <div style={{ background: "rgba(255,255,255,.03)", borderRadius: 14, border: "1px solid rgba(255,255,255,.07)", padding: "20px 24px", ...style }}>{children}</div>;
}

function Lbl({ children, color = "#64748b" }) {
  return <span style={{ fontSize: 11, fontWeight: 600, textTransform: "uppercase", letterSpacing: 1, color, fontFamily: "'Space Mono',monospace" }}>{children}</span>;
}

function Tag({ children, bg = "rgba(234,179,8,.08)", color = "#eab308", border = "rgba(234,179,8,.15)" }) {
  return <span style={{ fontSize: 11, fontWeight: 600, padding: "4px 10px", borderRadius: 6, background: bg, border: `1px solid ${border}`, color, fontFamily: "'Space Mono',monospace" }}>{children}</span>;
}
