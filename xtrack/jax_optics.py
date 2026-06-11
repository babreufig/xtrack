"""JAX optics backend for ``line.match(..., use_jax=True)``.

This is a new JAX backend that should support more precise physics than the
previous linear optics approximation approach. It differentiates the *exact*
element maps (ports of ``xtrack/beam_elements/elements_src``) rather than a linear
transfer-matrix approximation: the per-element transfer matrix R is obtained as
``jax.jacfwd`` of the exact map around the closed orbit, and the Twiss
parameters are propagated through the section with the standard formulas.

Only the Jacobian d(target)/d(knob) is produced here; the residual itself is
still evaluated by the normal xtrack twiss.  Supported target quantities are
``betx, alfx, mux, bety, alfy, muy, dx, dpx, dy, dpy`` (and relative phase
advance), matched against quadrupole-knob varies.

Performance design:
  * factorization - the closed orbit through a matched section is ~0 and
    essentially knob-independent, so R is built once at the frozen
    (``stop_gradient``) reference orbit with a single forward ``jacfwd`` vmap'd
    across elements, and the Twiss propagation is a no-AD ``lax.scan``;
  * compression - identity rows are dropped and runs of consecutive drifts are
    merged (the drift matrix is additive in length), never across a target.
"""

from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

jax.config.update("jax_enable_x64", True)

# Element-type codes for the jitted scan dispatch.
ET_DRIFT, ET_QUAD, ET_BEND, ET_SEXT, ET_KICK, ET_CORR, ET_IDENT = range(7)


# ---------------------------------------------------------------------------
# Single-element exact maps (ports of xtrack/beam_elements/elements_src).
# State layout is the xsuite 6-vector [x, px, y, py, zeta, delta].
# ---------------------------------------------------------------------------
def rvv_from_delta(delta, beta0):
    """rvv = beta / beta0 as a function of delta (see particles.py)."""
    one_plus_delta = 1.0 + delta
    denom = jnp.sqrt(
        beta0 * beta0 * one_plus_delta * one_plus_delta + 1.0 - beta0 * beta0
    )
    return one_plus_delta / denom


def _cs(K, length):
    """Focusing block, valid for any sign of K, autodiff-stable through K = 0.

    K > 0 : C = cos(sqrt(K) L),    S = sin(sqrt(K) L) / sqrt(K)
    K < 0 : C = cosh(sqrt(-K) L),  S = sinh(sqrt(-K) L) / sqrt(-K)
    K = 0 : C = 1,                 S = L
    """
    absK = jnp.abs(K)
    is_zero = absK <= 0.0
    # Double-where so neither the value nor the gradient sees sqrt(0).
    safe_absK = jnp.where(is_zero, 1.0, absK)
    r = jnp.where(is_zero, 0.0, jnp.sqrt(safe_absK))
    rl = r * length
    safe_r = jnp.where(is_zero, 1.0, r)
    C = jnp.where(K > 0, jnp.cos(rl), jnp.cosh(rl))
    S = jnp.where(
        is_zero,
        length,
        jnp.where(K > 0, jnp.sin(rl) / safe_r, jnp.sinh(rl) / safe_r),
    )
    return C, S


def drift_exact_jax(state, length, beta0):
    """Exact drift - port of Drift_single_particle_exact (track_drift.h)."""
    x, px, y, py, zeta, delta = (
        state[0],
        state[1],
        state[2],
        state[3],
        state[4],
        state[5],
    )
    one_plus_delta = 1.0 + delta
    one_over_pz = 1.0 / jnp.sqrt(one_plus_delta * one_plus_delta - px * px - py * py)
    rv0v = 1.0 / rvv_from_delta(delta, beta0)
    dzeta = 1.0 - rv0v * one_plus_delta * one_over_pz
    return jnp.array(
        [
            x + px * one_over_pz * length,
            px,
            y + py * one_over_pz * length,
            py,
            zeta + dzeta * length,
            delta,
        ]
    )


def quad_combined_map_jax(state, length, k0_, k1_, h, beta0):
    """Expanded combined dipole-quad map.

    Port of track_expanded_combined_dipole_quad_single_particle
    (track_magnet_drift.h, chi = 1). For a pure quad pass k0_ = h = 0.
    """
    x, px, y, py, zeta, delta = (
        state[0],
        state[1],
        state[2],
        state[3],
        state[4],
        state[5],
    )
    rvv = rvv_from_delta(delta, beta0)
    dp1 = 1.0 + delta

    k0 = k0_ / dp1
    k1 = k1_ / dp1
    Kx = k0 * h + k1
    Ky = -k1
    Cx, Sx = _cs(Kx, length)
    Cy, Sy = _cs(Ky, length)

    xp = px / dp1
    yp = py / dp1
    A = -Kx * x - k0 + h
    B = xp
    Cq = -Ky * y
    Dq = yp

    x_ = x * Cx + xp * Sx
    y_ = y * Cy + yp * Sy
    px_ = (A * Sx + B * Cx) * dp1
    py_ = (Cq * Sy + Dq * Cy) * dp1

    tol = 1e-9
    Kx_nz = jnp.abs(Kx) > tol
    Ky_nz = jnp.abs(Ky) > tol
    Kx_safe = jnp.where(Kx_nz, Kx, 1.0)
    Ky_safe = jnp.where(Ky_nz, Ky, 1.0)

    x_ = jnp.where(
        Kx_nz,
        x_ + (k0 - h) * (Cx - 1.0) / Kx_safe,
        x_ - (k0 - h) * 0.5 * length**2,
    )

    length_ = length
    term_x_nz = -(
        h * ((Cx - 1.0) * xp + Sx * A + length * (k0 - h))
    ) / Kx_safe + 0.5 * (
        -(A**2 * Cx * Sx) / (2.0 * Kx_safe)
        + (B**2 * Cx * Sx) / 2.0
        + (A**2 * length) / (2.0 * Kx_safe)
        + (B**2 * length) / 2.0
        - (A * B * Cx**2) / Kx_safe
        + (A * B) / Kx_safe
    )
    term_x_z = (
        h * length * (3.0 * length * xp + 6.0 * x - (k0 - h) * length**2) / 6.0
        + 0.5 * B**2 * length
    )
    length_ = length_ + jnp.where(Kx_nz, term_x_nz, term_x_z)

    term_y_nz = 0.5 * (
        -(Cq**2 * Cy * Sy) / (2.0 * Ky_safe)
        + (Dq**2 * Cy * Sy) / 2.0
        + (Cq**2 * length) / (2.0 * Ky_safe)
        + (Dq**2 * length) / 2.0
        - (Cq * Dq * Cy**2) / Ky_safe
        + (Cq * Dq) / Ky_safe
    )
    term_y_z = 0.5 * Dq**2 * length
    length_ = length_ + jnp.where(Ky_nz, term_y_nz, term_y_z)

    dzeta = length - length_ / rvv
    return jnp.array([x_, px_, y_, py_, zeta + dzeta, delta])


def curved_bend_map_jax(state, length, k0, h, beta0):
    """Exact curved sector-bend map (constant k0, h; assumes k0, h != 0).

    Taken from track_curved_exact_bend_single_particle (track_magnet_drift.h).
    """
    x, px, y, py, zeta, delta = (
        state[0],
        state[1],
        state[2],
        state[3],
        state[4],
        state[5],
    )
    rvv = rvv_from_delta(delta, beta0)
    dp1 = 1.0 + delta
    s = length
    k0_chi = k0

    A = 1.0 / jnp.sqrt(dp1**2 - py**2)
    pz = jnp.sqrt(dp1**2 - px**2 - py**2)

    Cc = pz - k0_chi * (1.0 / h + x)
    new_px = px * jnp.cos(s * h) + Cc * jnp.sin(s * h)
    new_pz = jnp.sqrt(dp1**2 - new_px**2 - py**2)
    d_new_px_ds = Cc * h * jnp.cos(h * s) - h * px * jnp.sin(h * s)

    new_x = (new_pz * h - d_new_px_ds - k0_chi) / (h * k0_chi)
    D = jnp.arcsin(A * px) - jnp.arcsin(A * new_px)
    new_y = y + (py * s) / (k0_chi / h) + (py / k0_chi) * D
    delta_ell = (dp1 * s * h) / k0_chi + (dp1 / k0_chi) * D

    return jnp.array(
        [new_x, new_px, new_y, py, zeta + (length - delta_ell / rvv), delta]
    )

def thin_quad_skew_kick_jax(state, knl1, ksl1):
    """Linear (n=1) part of a thin multipole kick: normal knl1 + skew ksl1.

    The only part of a thin multipole that affects the linear optics at zero
    orbit (dipole/sextupole/... give no first-order focusing there).
    """
    x, px, y, py, zeta, delta = (
        state[0],
        state[1],
        state[2],
        state[3],
        state[4],
        state[5],
    )
    dpx = knl1 * x - ksl1 * y
    dpy = knl1 * y + ksl1 * x
    return jnp.array([x, px - dpx, y, py + dpy, zeta, delta])


def sext_kick_jax(state, g):
    """Thin sextupole kick of integrated strength g = k2 * L."""
    x, px, y, py, zeta, delta = state
    return jnp.array(
        [x, px - 0.5 * g * (x * x - y * y), y, py + g * x * y, zeta, delta]
    )


def dipole_kick_jax(state, kh, kv):
    """Thin corrector dipole kick (horizontal kh = knl[0], vertical kv = ksl[0])."""
    x, px, y, py, zeta, delta = state
    return jnp.array([x, px - kh, y, py + kv, zeta, delta])


def quad_body_jax(state, length, k1, beta0):
    """Quadrupole body, 'mat-kick-mat' (two half combined-maps; xsuite default)."""
    half = 0.5 * length
    state = quad_combined_map_jax(state, half, 0.0, k1, 0.0, beta0)
    state = quad_combined_map_jax(state, half, 0.0, k1, 0.0, beta0)
    return state


def bend_body_jax(state, length, k0, h, beta0):
    """Pure-dipole body for model 'bend-kick-bend' (two half curved maps)."""
    half = 0.5 * length
    state = curved_bend_map_jax(state, half, k0, h, beta0)
    state = curved_bend_map_jax(state, half, k0, h, beta0)
    return state


def dipole_edge_linear_jax(state, r21, r43):
    """Linear dipole-edge kick: px += r21 x, py += r43 y.

    Port of DipoleEdgeLinear_single_particle (track_dipole_edge_linear.h).
    ``r21, r43`` are precomputed by ``bend_edge_coeffs`` and are constant during
    a quad match (they depend only on fixed bend strength/geometry), so jacfwd
    folds this linear kick straight into the transfer matrix R.
    """
    x, px, y, py, zeta, delta = (
        state[0],
        state[1],
        state[2],
        state[3],
        state[4],
        state[5],
    )
    return jnp.array([x, px + r21 * x, y, py + r43 * y, zeta, delta])


def bend_with_edges_jax(state, length, k0, h, beta0, r21_in, r43_in, r21_out, r43_out):
    """Curved-dipole body with entry/exit edge effects.

    Inactive/zero faces pass r21 = r43 = 0, so this is the code for
    both main ``Bend`` faces (e1 = e2 = 0) and ``RBend`` faces (effective face
    angle = h*L/2).
    """
    state = dipole_edge_linear_jax(state, r21_in, r43_in)
    state = bend_body_jax(state, length, k0, h, beta0)
    state = dipole_edge_linear_jax(state, r21_out, r43_out)
    return state


def sext_body_jax(state, length, g, beta0):
    """Sextupole body: drift(L/2) . thin kick(g = k2*L) . drift(L/2)."""
    state = drift_exact_jax(state, 0.5 * length, beta0)
    state = sext_kick_jax(state, g)
    return drift_exact_jax(state, 0.5 * length, beta0)


def corr_body_jax(state, length, kh, kv, beta0):
    """Corrector body: drift(L/2) . dipole kick(kh, kv) . drift(L/2)."""
    state = drift_exact_jax(state, 0.5 * length, beta0)
    state = dipole_kick_jax(state, kh, kv)
    return drift_exact_jax(state, 0.5 * length, beta0)


def bend_edge_coeffs(e):
    """Linear dipole-edge coefficients for a Bend/RBend's two faces.

    Python-side port of ``compute_dipole_edge_linear_coefficients`` plus the
    rbend curved-body face-angle augmentation (effective face angle = attribute
    + ``(h*L -/+ rbend_angle_diff)/2``). Returns
    ``(r21_in, r43_in, r21_out, r43_out)``; an inactive face contributes
    ``(0, 0)``.  Only the linear edge model is supported.
    """
    cls = type(e).__name__
    if cls not in ("Bend", "RBend"):
        return 0.0, 0.0, 0.0, 0.0
    k0 = float(e.k0)
    angle = float(e.h) * float(e.length)
    is_rbend = cls == "RBend"
    if is_rbend and str(getattr(e, "rbend_model", "adaptive")) == "straight":
        raise NotImplementedError(
            "use_jax: rbend straight-body edges are not supported "
            f"(element {getattr(e, 'name', '?')!r})"
        )
    angle_diff = float(getattr(e, "rbend_angle_diff", 0.0)) if is_rbend else 0.0
    aug_in = (angle - angle_diff) / 2.0 if is_rbend else 0.0
    aug_out = (angle + angle_diff) / 2.0 if is_rbend else 0.0

    def _face(active, model, e_ang, e_fd, fint, hgap, aug):
        if not int(active):
            return 0.0, 0.0
        if str(model) != "linear":
            raise NotImplementedError(
                f"use_jax supports only the linear dipole-edge model, "
                f"got {model!r} on element {getattr(e, 'name', '?')!r}"
            )
        e1 = float(e_ang) + aug
        r21 = k0 * np.tan(e1)
        e1v = e1 + float(e_fd)
        corr = 2.0 * k0 * float(hgap) * float(fint)
        temp = corr / np.cos(e1v) * (1.0 + np.sin(e1v) ** 2)
        r43 = -k0 * np.tan(e1v - temp)
        return r21, r43

    r21i, r43i = _face(
        e.edge_entry_active,
        e.edge_entry_model,
        e.edge_entry_angle,
        e.edge_entry_angle_fdown,
        e.edge_entry_fint,
        e.edge_entry_hgap,
        aug_in,
    )
    r21o, r43o = _face(
        e.edge_exit_active,
        e.edge_exit_model,
        e.edge_exit_angle,
        e.edge_exit_angle_fdown,
        e.edge_exit_fint,
        e.edge_exit_hgap,
        aug_out,
    )
    return r21i, r43i, r21o, r43o


# ---------------------------------------------------------------------------
# Linear optics (Twiss) propagation.
# ---------------------------------------------------------------------------

TW_LABELS = ["betx", "alfx", "mux", "bety", "alfy", "muy", "dx", "dpx", "dy", "dpy"]
TW_INDEX = {name: i for i, name in enumerate(TW_LABELS)}


def propagate_twiss(R, params):
    """Propagate Twiss params through one transfer matrix R (TW_LABELS order)."""
    betx0, alfx0, mux0, bety0, alfy0, muy0, dx0, dpx0, dy0, dpy0 = [
        params[i] for i in range(10)
    ]

    r00, r01, r10, r11 = R[0, 0], R[0, 1], R[1, 0], R[1, 1]
    tmp_x = r00 * betx0 - r01 * alfx0
    betx = (tmp_x**2 + r01**2) / betx0
    alfx = -((tmp_x * (r10 * betx0 - r11 * alfx0) + r01 * r11) / betx0)
    mux = mux0 + jnp.arctan2(r01, tmp_x) / (2 * jnp.pi)

    r22, r23, r32, r33 = R[2, 2], R[2, 3], R[3, 2], R[3, 3]
    tmp_y = r22 * bety0 - r23 * alfy0
    bety = (tmp_y**2 + r23**2) / bety0
    alfy = -((tmp_y * (r32 * bety0 - r33 * alfy0) + r23 * r33) / bety0)
    muy = muy0 + jnp.arctan2(r23, tmp_y) / (2 * jnp.pi)

    dx = r00 * dx0 + r01 * dpx0 + R[0, 5]
    dpx = r10 * dx0 + r11 * dpx0 + R[1, 5]
    dy = r22 * dy0 + r23 * dpy0 + R[2, 5]
    dpy = r32 * dy0 + r33 * dpy0 + R[3, 5]

    return jnp.array([betx, alfx, mux, bety, alfy, muy, dx, dpx, dy, dpy])


# ---------------------------------------------------------------------------
# Section encoding (struct-of-arrays) + compression.
# ---------------------------------------------------------------------------
# Field schema shared by all three backends: (name, is_int, default).  ``Enc``
# lists these fields in the same order and ``emap_jax`` unpacks them
# positionally, so the three must stay in sync.
_ENC_FIELDS = (
    ("etype", True, ET_IDENT),
    ("L", False, 0.0),
    ("k0", False, 0.0),
    ("h", False, 0.0),
    ("k1fix", False, 0.0),  # quad k1 when kqidx < 0
    ("k2fix", False, 0.0),  # sext k2 when ksidx < 0
    ("kqidx", True, -1),  # index into live kq vector, or -1
    ("ksidx", True, -1),  # index into live ks vector, or -1
    ("hidx", True, -1),  # index into live corrector vector (knl[0]), or -1
    ("vidx", True, -1),  # index into live corrector vector (ksl[0]), or -1
    ("knl1", False, 0.0),  # thin normal quad component knl[1]
    ("ksl1", False, 0.0),  # thin skew   quad component ksl[1]
    ("r21i", False, 0.0),  # linear dipole-edge coeffs (0 => no edge)
    ("r43i", False, 0.0),
    ("r21o", False, 0.0),
    ("r43o", False, 0.0),
    ("Dx", False, 0.0),  # frozen dispersion (global off-momentum seed)
    ("Dpx", False, 0.0),
    ("Dy", False, 0.0),
    ("Dpy", False, 0.0),
)


class Enc(NamedTuple):
    """Struct-of-arrays encoding of a lattice section for the JAX scan.

    Each field is a 1-D array of length N (one entry per encoded row).  All
    three backends (optics / global / orbit) share this single layout; a backend
    leaves the fields it does not use at their defaults (``-1`` index / ``0``).
    Built via :func:`pack_rows`, consumed by :func:`emap_jax`; field order must
    match ``_ENC_FIELDS``.
    """

    etype: jnp.ndarray
    L: jnp.ndarray
    k0: jnp.ndarray
    h: jnp.ndarray
    k1fix: jnp.ndarray
    k2fix: jnp.ndarray
    kqidx: jnp.ndarray
    ksidx: jnp.ndarray
    hidx: jnp.ndarray
    vidx: jnp.ndarray
    knl1: jnp.ndarray
    ksl1: jnp.ndarray
    r21i: jnp.ndarray
    r43i: jnp.ndarray
    r21o: jnp.ndarray
    r43o: jnp.ndarray
    Dx: jnp.ndarray
    Dpx: jnp.ndarray
    Dy: jnp.ndarray
    Dpy: jnp.ndarray


def pack_rows(rows):
    """Pack a list of per-element dicts into an :class:`Enc`; missing keys take
    the field default.  The single place that does the SoA conversion."""
    cols = {}
    for name, is_int, default in _ENC_FIELDS:
        vals = [r.get(name, default) for r in rows]
        cols[name] = (
            jnp.array(vals, dtype=jnp.int32) if is_int else jnp.array(vals, dtype=float)
        )
    return Enc(**cols)


def _row_at(enc, i):
    """Extract original row ``i`` of an ``Enc`` as a dict (used by compression)."""
    return {
        name: (int(getattr(enc, name)[i]) if is_int else float(getattr(enc, name)[i]))
        for name, is_int, _ in _ENC_FIELDS
    }


def emap_jax(state, row, kq, ks, corr, beta0):
    """Single element map: dispatch one ``Enc`` row on its ``etype``.

    ``kq`` / ``ks`` / ``corr`` are the live differentiable strength vectors
    (varied quads / sexts / corrector kicks); a row reads its strength from the
    relevant vector when its index is >= 0, else from the baked-in fixed value.
    Each vector must be non-empty (callers append a trailing 0 so the
    ``maximum(idx, 0)`` gather stays in-bounds when ``idx == -1``).
    """
    (
        etype, L, k0, h, k1fix, k2fix, kqidx, ksidx, hidx, vidx,
        knl1, ksl1, r21i, r43i, r21o, r43o, Dx, Dpx, Dy, Dpy,
    ) = row
    k1 = jnp.where(kqidx >= 0, kq[jnp.maximum(kqidx, 0)], k1fix)
    k2 = jnp.where(ksidx >= 0, ks[jnp.maximum(ksidx, 0)], k2fix)
    g = k2 * L
    kh = jnp.where(hidx >= 0, corr[jnp.maximum(hidx, 0)], 0.0)
    kv = jnp.where(vidx >= 0, corr[jnp.maximum(vidx, 0)], 0.0)
    return lax.switch(
        etype,
        [
            lambda s: drift_exact_jax(s, L, beta0),  # ET_DRIFT
            lambda s: quad_body_jax(s, L, k1, beta0),  # ET_QUAD
            lambda s: bend_with_edges_jax(  # ET_BEND
                s, L, k0, h, beta0, r21i, r43i, r21o, r43o
            ),
            lambda s: sext_body_jax(s, L, g, beta0),  # ET_SEXT
            lambda s: thin_quad_skew_kick_jax(s, knl1, ksl1),  # ET_KICK
            lambda s: corr_body_jax(s, L, kh, kv, beta0),  # ET_CORR
            lambda s: s,  # ET_IDENT
        ],
        state,
    )


def encode_section(line, ordered_names, kq_index, want_edges=True):
    """Encode every section element into an :class:`Enc` (optics layout).

    Linear-focusing elements (Quadrupole, Bend, RBend) get their exact map;
    thick non-focusing elements are drifts; thin elements apply their linear
    multipole kick.  Quads named in ``kq_index`` read k1 from the live kq vector.
    Dipole edges are encoded when ``want_edges`` (the optics default).
    """
    ed = line.element_dict
    rows = []
    for nm in ordered_names:
        e = ed.get(nm)
        cls = type(e).__name__ if e is not None else None
        length = float(getattr(e, "length", 0.0) or 0.0)
        isthick = bool(getattr(e, "isthick", False))
        if e is None:
            rows.append({"etype": ET_IDENT})
        elif nm in kq_index:
            rows.append({"etype": ET_QUAD, "L": length, "kqidx": kq_index[nm]})
        elif cls == "Quadrupole" and float(e.k1) != 0.0:
            rows.append({"etype": ET_QUAD, "L": length, "k1fix": float(e.k1)})
        elif cls in ("Bend", "RBend"):
            k0v, hv, k1v = float(e.k0), float(e.h), float(e.k1)
            if not (k1v == 0.0 and hv != 0.0):
                raise NotImplementedError(
                    f"combined-function bend {nm} not supported by use_jax"
                )
            r = bend_edge_coeffs(e) if want_edges else (0.0, 0.0, 0.0, 0.0)
            rows.append(
                {
                    "etype": ET_BEND, "L": length, "k0": k0v, "h": hv,
                    "r21i": r[0], "r43i": r[1], "r21o": r[2], "r43o": r[3],
                }
            )
        elif isthick and length > 0.0:
            rows.append({"etype": ET_DRIFT, "L": length})
        else:
            knl = np.asarray(getattr(e, "knl", [0.0]), dtype=float)
            ksl = np.asarray(getattr(e, "ksl", [0.0]), dtype=float)
            kn1 = float(knl[1]) if len(knl) > 1 else 0.0
            ks1 = float(ksl[1]) if len(ksl) > 1 else 0.0
            if kn1 or ks1:
                rows.append({"etype": ET_KICK, "knl1": kn1, "ksl1": ks1})
            else:
                rows.append({"etype": ET_IDENT})
    return pack_rows(rows)


def compress_encoding(enc, boundary_rows):
    """Drop identity rows and merge runs of consecutive drifts.

    `Boundary rows are target positions that must not be merged away, so that
    scan output index i still corresponds to the optics after a specific element.
    Returns (compressed Enc, orig_to_comp) where ``orig_to_comp[i]`` is the
    compressed row whose output holds the optics *after* original element i.
    """
    et = np.asarray(enc.etype)
    L = np.asarray(enc.L)
    boundary = set(int(b) for b in boundary_rows)

    rows = []
    orig_to_comp = np.empty(len(et), dtype=np.int64)
    acc = 0.0

    def flush():
        nonlocal acc
        if acc > 0.0:
            rows.append({"etype": ET_DRIFT, "L": acc})
            acc = 0.0

    for i in range(len(et)):
        if et[i] == ET_IDENT:
            pass
        elif et[i] == ET_DRIFT:
            acc += float(L[i])
        else:
            flush()
            rows.append(_row_at(enc, i))
        if i in boundary:
            flush()
        orig_to_comp[i] = len(rows) - 1
    flush()

    return pack_rows(rows), orig_to_comp


def build_section_twiss(enc, beta0, p0, s0):
    """Factorized Twiss target builder over a (compressed) encoded section.

    Returns ``targets(kq, rows, qidx)``: propagate the orbit (frozen for the
    Jacobian), build every element's R at that orbit with one vmap'd forward
    jacfwd, propagate Twiss with a scan, and gather the requested components.
    """

    # optics differentiates only w.r.t. the quad vector kq; ks / corr are unused
    # (every optics row carries ksidx == hidx == vidx == -1).
    _ks0 = jnp.zeros(1)
    _corr0 = jnp.zeros(1)

    def emap_single(s, e, kq):
        return emap_jax(s, e, kq, _ks0, _corr0, beta0)

    def all_params(kq):
        # Step 1: propagate the orbit (frozen - no gradient tracking)
        # Works thus only if orbit changes have negligible effect on
        # the optics Jacobian.
        def obody(s, e):
            return emap_single(s, e, kq), s

        _, orbits_in = lax.scan(obody, s0, enc)
        orbits_in = lax.stop_gradient(orbits_in)

        # Obtain R matrices at frozen orbits
        Rs = jax.vmap(lambda e, o: jax.jacfwd(lambda s: emap_single(s, e, kq))(o))(
            enc, orbits_in
        )

        # Twiss propagation with scan over the section.
        def tbody(params, R):
            pn = propagate_twiss(R, params)
            return pn, pn

        _, hist = lax.scan(tbody, p0, Rs)
        return hist

    @jax.jit
    def targets(kq, rows, qidx):
        return all_params(kq)[rows, qidx]

    return targets


def knob_strength_jacobian(line, vary_names, attr_key, idx=None):
    """Exact d(element.<attr_key>)/d(knob) from the xdeps expression graph.

    This asks the dependency manager (``ref_manager.find_deps``) for exactly
    the element attributes that depend on each knob, builds the knob's symbolic
    expression function (``mk_fun``) and differentiates it with SymPy.
    The optics knobs are linear in the strengths, so the derivative is constant.
    Returns ``{knob_name: {element_name: d(attr)/d(knob)}}``.  Falls back to a central
    difference (over just the dependent elements) if the symbolic path fails.

    ``attr_key`` may be a scalar attribute (``'k1'``, ``'k2'``) or an array
    attribute (``'knl'``, ``'ksl'``); for the latter pass ``idx`` to select the
    element read/differentiated (e.g. ``attr_key='knl', idx=0`` for a corrector
    dipole kick).  The generated update code assigns ``elem.knl[idx] = expr``,
    so the dummy containers expose array attributes that record those item
    assignments and the readback differentiates ``elem.<attr_key>[idx]``.
    """
    import sympy

    class _ArrayProxy(dict):
        # records ``proxy[i] = expr`` from the generated code; unset items read
        # back as 0.0 (an index this knob does not drive).
        def __missing__(self, k):
            return 0.0

    class _Dummy:
        # auto-vivify any attribute as an array proxy so ``dummy.knl[i] = expr``
        # works; scalar ``dummy.k1 = expr`` goes through normal setattr instead.
        def __getattr__(self, name):
            p = _ArrayProxy()
            object.__setattr__(self, name, p)
            return p

    def _read(container, attr):
        v = getattr(container, attr)
        return v[idx] if idx is not None else v

    rm = line.ref_manager
    ed = line.element_dict
    out = {}
    for knob in vary_names:
        deps = [
            d
            for d in rm.find_deps([line.vars[knob]])
            if d.__class__.__name__ == "AttrRef"
        ]
        names = [d._owner._key for d in deps if d._key == attr_key]
        if not names:
            out[knob] = {}
            continue
        try:
            a = sympy.var("a")
            dummy = {d._owner._key: _Dummy() for d in deps}
            code = rm.mk_fun("myfun", a=line.vars[knob])
            g = {"vars": rm.containers["vars"]._owner.copy(), "element_refs": dummy}
            loc = {}
            exec(code, g, loc)
            loc["myfun"](a)
            out[knob] = {
                en: float(sympy.diff(_read(g["element_refs"][en], attr_key), a))
                for en in names
            }
        except Exception:
            v0 = float(line[knob])
            d = {}
            for en in names:
                line[knob] = v0 + 1e-4
                kp = float(_read(ed[en], attr_key))
                line[knob] = v0 - 1e-4
                km = float(_read(ed[en], attr_key))
                d[en] = (kp - km) / 2e-4
            line[knob] = v0
            out[knob] = d
    return out


def scalar_chain(line, vary_names, attr, element_names):
    """(varied_names, dp) for a scalar element attribute (k1, k2), exact via
    the xdeps graph; ``dp[i, j] = d(varied[i].attr)/d(vary[j])``."""
    d = knob_strength_jacobian(line, vary_names, attr)
    varied = [n for n in element_names if any(n in d[v] for v in vary_names)]
    dp = np.zeros((len(varied), len(vary_names)))
    for j, v in enumerate(vary_names):
        for i, n in enumerate(varied):
            dp[i, j] = d[v].get(n, 0.0)
    return varied, dp


def kick_chain(line, vary_names, ordered):
    """Corrector dipole kicks (knl[0]/ksl[0]) driven by the knobs, exact via
    the xdeps graph (``knob_strength_jacobian`` with ``idx=0``).  Returns
    (params, hidx, vidx, dp) where dp[i, j] = d(param_i)/d(vary_j)."""
    dh = knob_strength_jacobian(line, vary_names, "knl", idx=0)
    dv = knob_strength_jacobian(line, vary_names, "ksl", idx=0)

    def driven(d):
        s = {n for v in vary_names for n, val in d[v].items() if val != 0.0}
        return [n for n in ordered if n in s]  # deterministic order

    hnames, vnames = driven(dh), driven(dv)
    params = [(n, "knl") for n in hnames] + [(n, "ksl") for n in vnames]
    ppos = {pc: i for i, pc in enumerate(params)}
    hidx = {n: ppos[(n, "knl")] for n in hnames}
    vidx = {n: ppos[(n, "ksl")] for n in vnames}

    dp = np.zeros((len(params), len(vary_names)))
    for j, v in enumerate(vary_names):
        for n in hnames:
            dp[ppos[(n, "knl")], j] = dh[v].get(n, 0.0)
        for n in vnames:
            dp[ppos[(n, "ksl")], j] = dv[v].get(n, 0.0)
    return params, hidx, vidx, dp