"""
Text input parser for v3 calculator format.
Port of validate_v3_input.m by Greg Balco (BGC).

Input format described at:
  https://hess.ess.washington.edu/math/docs/v3/v3_input_explained.html
"""

import re
import datetime
import numpy as np
from .constants import make_consts


_NUCLIDE_TOKENS = {"Be-10", "Al-26", "Ne-21", "He-3", "C-14", "Cl-36"}


def parse_v3_input(text_block: str, consts=None) -> dict:
    """
    Parse and validate a v3-format input text block.

    Parameters
    ----------
    text_block : str
        Semicolon-delimited block of sample, nuclide, and optional age lines.
    consts : Constants, optional

    Returns
    -------
    dict with keys ``s`` (samples), ``n`` (nuclide measurements),
    and optionally ``c`` (calibration ages).
    Raises ValueError on malformed input.
    """
    if consts is None:
        consts = make_consts()

    # Replace tabs with spaces and split on semicolons
    text_block = text_block.replace("\t", " ").strip()
    raw_lines = [l.strip() for l in text_block.split(";")]
    lines = [l for l in raw_lines if l]

    if len(lines) < 2:
        raise ValueError("Need at least one sample line and one nuclide line.")

    # Classify lines
    s_lines, n_lines, t_lines = [], [], []
    for i, line in enumerate(lines):
        tokens = line.split()
        if any(tok in _NUCLIDE_TOKENS for tok in tokens):
            n_lines.append(i)
        elif "true_t" in tokens:
            t_lines.append(i)
        else:
            s_lines.append(i)

    if t_lines and len(t_lines) != len(s_lines):
        raise ValueError("Number of independent-age lines must match number of sample lines.")

    # ---- Preallocate sample structure ----
    ns = len(s_lines)
    nn = len(n_lines)

    s = {
        "sample_name": [],
        "aa":          [],
        "lat":         np.zeros(ns),
        "long":        np.zeros(ns),
        "elv":         np.zeros(ns),
        "pressure":    np.zeros(ns),
        "thick":       np.zeros(ns),
        "rho":         np.zeros(ns),
        "othercorr":   np.zeros(ns),
        "E":           np.zeros(ns),
        "yr":          np.zeros(ns, dtype=int),
    }

    n = {
        "index":       np.zeros(nn, dtype=int),
        "nuclide":     [],
        "sample_name": [],
        "N":           np.zeros(nn),
        "delN":        np.zeros(nn),
    }

    # ---- Parse sample lines ----
    today_yr = datetime.date.today().year
    for a, li in enumerate(s_lines):
        parts = lines[li].split()
        if len(parts) != 10:
            raise ValueError(
                f"Sample line {li}: expected 10 fields, got {len(parts)}."
            )
        name, lat_s, lon_s, elv_s, aa_s, thick_s, rho_s, shield_s, eros_s, yr_s = parts

        _check_name(name, li)
        s["sample_name"].append(name)

        lat_v = _to_float(lat_s, "latitude", li)
        if not -90 <= lat_v <= 90:
            raise ValueError(f"Latitude out of bounds on line {li}.")
        s["lat"][a] = lat_v

        lon_v = _to_float(lon_s, "longitude", li)
        if not -180 <= lon_v <= 180:
            raise ValueError(f"Longitude out of bounds on line {li}.")
        s["long"][a] = lon_v

        if aa_s not in ("std", "ant", "pre"):
            raise ValueError(f"Unknown elevation/pressure flag '{aa_s}' on line {li}.")
        s["aa"].append(aa_s)

        elv_v = _to_float(elv_s, "elevation/pressure", li)
        if aa_s in ("std", "ant"):
            if elv_v < -500:
                raise ValueError(f"Elevation too low on line {li}.")
            s["elv"][a] = elv_v
            s["pressure"][a] = -99
        else:
            if not 0 <= elv_v <= 1080:
                raise ValueError(f"Pressure out of bounds on line {li}.")
            s["pressure"][a] = elv_v
            s["elv"][a] = -99

        thick_v = _to_float(thick_s, "thickness", li)
        if thick_v < 0:
            raise ValueError(f"Thickness < 0 on line {li}.")
        s["thick"][a] = thick_v

        rho_v = _to_float(rho_s, "density", li)
        if rho_v < 0:
            raise ValueError(f"Density < 0 on line {li}.")
        s["rho"][a] = rho_v

        shield_v = _to_float(shield_s, "shielding", li)
        if not 0 <= shield_v <= 1:
            raise ValueError(f"Shielding out of range on line {li}.")
        s["othercorr"][a] = shield_v

        eros_v = _to_float(eros_s, "erosion rate", li)
        if eros_v < 0:
            raise ValueError(f"Erosion rate < 0 on line {li}.")
        s["E"][a] = eros_v

        yr_v = int(round(_to_float(yr_s, "collection year", li)))
        if yr_v == 0:
            yr_v = consts.default_yr
        elif yr_v > today_yr:
            raise ValueError(f"Collection year is in the future on line {li}.")
        elif yr_v < 1700:
            raise ValueError(f"Collection year out of bounds on line {li}.")
        s["yr"][a] = yr_v

    # ---- Parse nuclide lines ----
    for a, li in enumerate(n_lines):
        parts = lines[li].split()
        name = parts[0]
        _check_name(name, li)
        n["sample_name"].append(name)

        idx = _match_sample(name, s["sample_name"], li)
        n["index"][a] = idx

        nuclide_id = parts[1]

        if nuclide_id == "Be-10":
            if len(parts) != 6:
                raise ValueError(f"Be-10 line {li}: expected 6 fields.")
            mineral = parts[2]
            if mineral not in consts.ok_mins_10:
                raise ValueError(f"Unknown mineral for Be-10 on line {li}.")
            N_raw    = _to_float(parts[3], "Be-10 concentration", li)
            delN_raw = _to_float(parts[4], "Be-10 uncertainty", li)
            std_name = parts[5]
            cf = _get_cf(std_name, consts.be_stds_names, consts.be_stds_cfs, "Be-10", li)
            n["N"][a]    = N_raw * cf
            n["delN"][a] = delN_raw * cf
            n["nuclide"].append(f"N10{mineral}")

        elif nuclide_id == "Al-26":
            if len(parts) != 6:
                raise ValueError(f"Al-26 line {li}: expected 6 fields.")
            mineral = parts[2]
            if mineral not in consts.ok_mins_26:
                raise ValueError(f"Unknown mineral for Al-26 on line {li}.")
            N_raw    = _to_float(parts[3], "Al-26 concentration", li)
            delN_raw = _to_float(parts[4], "Al-26 uncertainty", li)
            std_name = parts[5]
            cf = _get_cf(std_name, consts.al_stds_names, consts.al_stds_cfs, "Al-26", li)
            n["N"][a]    = N_raw * cf
            n["delN"][a] = delN_raw * cf
            n["nuclide"].append(f"N26{mineral}")

        elif nuclide_id == "He-3":
            if len(parts) != 7:
                raise ValueError(f"He-3 line {li}: expected 7 fields.")
            mineral  = parts[2]
            if mineral not in consts.ok_mins_3:
                raise ValueError(f"Unknown mineral for He-3 on line {li}.")
            N_raw    = _to_float(parts[3], "He-3 concentration", li)
            delN_raw = _to_float(parts[4], "He-3 uncertainty", li)
            std_name = parts[5]
            std_val  = _to_float(parts[6], "He-3 standard concentration", li)
            if std_name == "NONE":
                n["N"][a]    = N_raw
                n["delN"][a] = delN_raw
            else:
                std_ref = _get_cf(std_name, consts.he_stds_names, consts.he_stds_ref, "He-3", li)
                n["N"][a]    = N_raw    * (std_ref / std_val)
                n["delN"][a] = delN_raw * (std_ref / std_val)
            n["nuclide"].append(f"N3{mineral}")

        elif nuclide_id == "Ne-21":
            if len(parts) != 7:
                raise ValueError(f"Ne-21 line {li}: expected 7 fields.")
            mineral  = parts[2]
            if mineral not in consts.ok_mins_21:
                raise ValueError(f"Unknown mineral for Ne-21 on line {li}.")
            N_raw    = _to_float(parts[3], "Ne-21 concentration", li)
            delN_raw = _to_float(parts[4], "Ne-21 uncertainty", li)
            std_name = parts[5]
            std_val  = _to_float(parts[6], "Ne-21 standard concentration", li)
            if std_name == "NONE":
                n["N"][a]    = N_raw
                n["delN"][a] = delN_raw
            else:
                std_ref = _get_cf(std_name, consts.ne_stds_names, consts.ne_stds_ref, "Ne-21", li)
                n["N"][a]    = N_raw    * (std_ref / std_val)
                n["delN"][a] = delN_raw * (std_ref / std_val)
            n["nuclide"].append(f"N21{mineral}")

        elif nuclide_id == "C-14":
            if len(parts) != 5:
                raise ValueError(f"C-14 line {li}: expected 5 fields.")
            mineral  = parts[2]
            if mineral not in consts.ok_mins_14:
                raise ValueError(f"Unknown mineral for C-14 on line {li}.")
            N_raw    = _to_float(parts[3], "C-14 concentration", li)
            delN_raw = _to_float(parts[4], "C-14 uncertainty", li)
            n["N"][a]    = N_raw
            n["delN"][a] = delN_raw
            n["nuclide"].append(f"N14{mineral}")

        elif nuclide_id == "Cl-36":
            raise ValueError(f"Cl-36 not yet supported (line {li}).")
        else:
            raise ValueError(f"Unrecognized nuclide '{nuclide_id}' on line {li}.")

    # ---- Validate nuclide-sample linkage ----
    for a in range(ns):
        if not any(n["index"] == a):
            raise ValueError(f"Sample '{s['sample_name'][a]}' has no nuclide measurements.")

    out = {"s": s, "n": n}

    # ---- Parse calibration age lines ----
    if t_lines:
        c = {
            "sample_name": [],
            "site_name":   [],
            "truet":       np.zeros(ns),
            "dtruet":      np.zeros(ns),
            "mint":        np.zeros(ns),
            "dmint":       np.zeros(ns),
            "maxt":        np.full(ns, np.inf),
            "dmaxt":       np.zeros(ns),
            "index":       np.zeros(ns, dtype=int),
        }
        for a, li in enumerate(t_lines):
            parts = lines[li].split()
            name = parts[0]
            _check_name(name, li)
            c["sample_name"].append(name)
            idx = _match_sample(name, s["sample_name"], li)
            c["index"][a] = idx

            _check_name(parts[2], li)
            c["site_name"].append(parts[2])

            age_parts = parts[3:]
            ages = []
            for p in age_parts:
                if p == "Inf":
                    ages.append(np.inf)
                else:
                    ages.append(float(p))

            if len(ages) == 2:
                c["truet"][idx]  = ages[0]
                c["dtruet"][idx] = ages[1]
            elif len(ages) == 4:
                c["mint"][idx]  = ages[0]
                c["dmint"][idx] = ages[1]
                c["maxt"][idx]  = ages[2]
                c["dmaxt"][idx] = ages[3]
            else:
                raise ValueError(f"Independent age line {li}: need 2 or 4 age fields.")

        out["c"] = c

    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_name(name, line_no):
    if len(name) > 32:
        raise ValueError(f"Name too long on line {line_no}.")
    if re.search(r"[^\w-]", name):
        raise ValueError(f"Illegal characters in name on line {line_no}.")


def _to_float(s, field, line_no):
    try:
        return float(s)
    except ValueError:
        raise ValueError(f"Cannot parse {field} '{s}' on line {line_no}.")


def _match_sample(name, sample_names, line_no):
    matches = [i for i, n in enumerate(sample_names) if n == name]
    if not matches:
        raise ValueError(f"Cannot match name '{name}' to any sample (line {line_no}).")
    if len(matches) > 1:
        raise ValueError(f"Name '{name}' matches multiple samples (line {line_no}).")
    return matches[0]


def _get_cf(std_name, std_names, std_cfs, nuclide, line_no):
    try:
        idx = list(std_names).index(std_name)
    except ValueError:
        raise ValueError(f"Unknown {nuclide} standard '{std_name}' on line {line_no}.")
    return std_cfs[idx]


# ---------------------------------------------------------------------------
# CSV → v3 conversion
# ---------------------------------------------------------------------------

def csv_to_v3(path: str) -> str:
    """
    Convert a v3-field CSV to a v3 text block accepted by parse_v3_input().

    Required columns
    ----------------
    name, lat, lon, elevation

    Optional location columns (defaults shown)
    ------------------------------------------
    elv_flag     std | ant | pre    (default: std)
    thickness    cm                 (default: 2.0)
    density      g/cm³              (default: 2.65)
    shielding    0–1                (default: 1.0)
    erosion      cm/yr              (default: 0.0)
    year         collection year   (default: 0 → uses consts.default_yr)

    Optional nuclide columns (omit for production-rate-only workflow)
    -----------------------------------------------------------------
    nuclide      Be-10 | Al-26 | He-3 | Ne-21 | C-14
    mineral      quartz | pyroxene | olivine
    concentration  atoms/g
    uncertainty    atoms/g
    standard       standard name (e.g. 07KNSTD, KNSTD, CRONUS-P, NONE)
    standard_value measured standard concentration (He-3/Ne-21 only)

    Multiple rows with the same sample name are collapsed into one v3 sample
    block with multiple nuclide lines (used for dual-nuclide samples).
    """
    import csv
    from collections import OrderedDict

    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        raise ValueError(f"No data rows in {path}")

    has_nuclide = "nuclide" in rows[0] and any(r.get("nuclide", "").strip() for r in rows)
    if not has_nuclide:
        raise ValueError(
            "csv_to_v3() requires nuclide columns (nuclide, mineral, concentration, "
            "uncertainty, standard). For location-only CSVs use from_csv() directly."
        )

    # Group rows by sample name, preserving insertion order
    samples: "OrderedDict[str, dict]" = OrderedDict()
    for r in rows:
        name = r["name"].strip()
        if name not in samples:
            samples[name] = {"info": r, "nuclide_rows": []}
        if r.get("nuclide", "").strip():
            samples[name]["nuclide_rows"].append(r)

    lines = []
    for name, data in samples.items():
        r = data["info"]

        def _get(key, default):
            v = r.get(key, default)
            return str(v).strip() if v is not None and str(v).strip() else str(default)

        lat      = _get("lat", "")
        lon      = _get("lon", "")
        elv      = _get("elevation", "")
        elv_flag = _get("elv_flag", "std")
        thick    = _get("thickness", "2.0")
        density  = _get("density", "2.65")
        shield   = _get("shielding", "1.0")
        erosion  = _get("erosion", "0.0")
        year     = _get("year", "0")

        lines.append(
            f"{name} {lat} {lon} {elv} {elv_flag} "
            f"{thick} {density} {shield} {erosion} {year};"
        )

        for nr in data["nuclide_rows"]:
            nuclide = nr.get("nuclide", "").strip()
            mineral = nr.get("mineral", "").strip()
            conc    = nr.get("concentration", "").strip()
            uncert  = nr.get("uncertainty", "").strip()
            std     = nr.get("standard", "NONE").strip() or "NONE"

            if nuclide in ("Be-10", "Al-26"):
                lines.append(f"{name} {nuclide} {mineral} {conc} {uncert} {std};")
            elif nuclide in ("He-3", "Ne-21"):
                std_val = nr.get("standard_value", "").strip()
                if not std_val:
                    raise ValueError(
                        f"He-3/Ne-21 row for sample '{name}' requires a "
                        f"'standard_value' column (measured standard concentration)."
                    )
                lines.append(f"{name} {nuclide} {mineral} {conc} {uncert} {std} {std_val};")
            elif nuclide == "C-14":
                lines.append(f"{name} {nuclide} {mineral} {conc} {uncert};")
            else:
                raise ValueError(
                    f"Unrecognized nuclide '{nuclide}' for sample '{name}'. "
                    f"Expected one of: Be-10, Al-26, He-3, Ne-21, C-14."
                )

    return "\n".join(lines)
