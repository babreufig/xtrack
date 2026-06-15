"""Functional integrators + element maps for the JAX backends (the
physics layer used by ``use_jax_integrators``).

Each thick magnet is tracked the way xsuite does - ``fringe . integrator(drift,
kick) . fringe`` - with the drift-slot map, the drift/kick strength split, the
integrator and ``num_kicks`` resolved per element by :func:`resolve_magnet` /
:func:`configure_tracking_model` (ports of the ``track_magnet_*`` C kernels).
Default integrator: Yoshida-4 (== xsuite/MAD-NG); Yoshida-2 / exact selectable.
"""

import math
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

# Leaf maps composed here (no import cycle: jax_optics does not import this).
from .jax_optics import (
    rvv_from_delta,
    curved_bend_map_jax,
    drift_exact_jax,
    quad_combined_map_jax,
    dipole_edge_linear_jax,
    bend_edge_coeffs,
    propagate_twiss,
)

jax.config.update("jax_enable_x64", True)


# Yoshida-4 coefficients
_D_YOSHIDA = (
    3.922568052387799819591407413100e-01,
    5.100434119184584780271052295575e-01,
    -4.710533854097565531482416645304e-01,
    6.875316825251809316199569366290e-02,
)
_K_YOSHIDA = (
    7.845136104775599639182814826199e-01,
    2.355732133593569921359289764951e-01,
    -1.177679984178870098432412305556e00,
    1.315186320683906284756403692882e00,
)
_N_KICKS_YOSHIDA = 7

# Default bend body integrator
BEND_SCHEME = "yoshida4"


# ---------------------------------------------------------------------------
# Physics maps (ports of track_magnet_drift.h / track_magnet_kick.h).
# State is the xsuite 6-vector [x, px, y, py, zeta, delta].
# ---------------------------------------------------------------------------
def drift_expanded_jax(state, length, beta0):
    """Expanded (paraxial) drift - port of track_expanded_drift_single_particle."""
    x, px, y, py, zeta, delta = (
        state[0],
        state[1],
        state[2],
        state[3],
        state[4],
        state[5],
    )
    rpp = 1.0 / (1.0 + delta)
    rv0v = 1.0 / rvv_from_delta(delta, beta0)
    xp = px * rpp
    yp = py * rpp
    dzeta = 1.0 - rv0v * (1.0 + (xp * xp + yp * yp) / 2.0)
    return jnp.array(
        [x + xp * length, px, y + yp * length, py, zeta + length * dzeta, delta]
    )


def polar_drift_jax(state, length, h, beta0):
    """Polar (curved) drift - port of track_polar_drift_single_particle.

    The "rot" of rot-kick-rot; field-free, so it carries no focusing.
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
    one_plus_delta = 1.0 + delta
    pz = jnp.sqrt(one_plus_delta * one_plus_delta - px * px - py * py)
    rho = 1.0 / h
    ca = jnp.cos(h * length)
    sa = jnp.sin(h * length)
    sa2 = jnp.sin(0.5 * h * length)
    _pz = 1.0 / pz
    pxt = px * _pz
    _ptt = 1.0 / (ca - sa * pxt)
    pst = (x + rho) * sa * _pz * _ptt
    new_x = (x + rho * (2.0 * sa2 * sa2 + sa * pxt)) * _ptt
    new_px = ca * px + sa * pz
    new_y = y + pst * py
    delta_ell = one_plus_delta * (x + rho) * sa / ca / pz / (1.0 - px * sa / ca / pz)
    return jnp.array(
        [new_x, new_px, new_y, py, zeta + (length - delta_ell / rvv), delta]
    )


def rot_kick_rot_bend_jax(state, length, k0, h, beta0):
    """Curved-bend drift slot for ``drift_model == 7`` (track_magnet_drift.h
    case 7): a 4th-order Yoshida of polar drifts + dipole px-kicks (k0 & h only;
    k1 and the weak-focusing corrections live in the outer kick)."""
    d0, kc0 = 0.6756035959798289, 1.3512071919596578
    d1, kc1 = -0.17560359597982889, -1.7024143839193155

    def _px(s, coef):
        return s.at[1].add(-coef * k0 * length)

    state = polar_drift_jax(state, d0 * length, h, beta0)
    state = _px(state, kc0)
    state = polar_drift_jax(state, d1 * length, h, beta0)
    state = _px(state, kc1)
    state = polar_drift_jax(state, d1 * length, h, beta0)
    state = _px(state, kc0)
    state = polar_drift_jax(state, d0 * length, h, beta0)
    return state


def straight_bend_jax(state, length, k0, beta0):
    """Exact straight bend (h=0) - port of track_straight_exact_bend_single_particle."""
    x, px, y, py, zeta, delta = (
        state[0],
        state[1],
        state[2],
        state[3],
        state[4],
        state[5],
    )
    rvv = rvv_from_delta(delta, beta0)
    one_plus_delta = 1.0 + delta
    A = 1.0 / jnp.sqrt(one_plus_delta * one_plus_delta - py * py)
    pz = jnp.sqrt(one_plus_delta * one_plus_delta - px * px - py * py)
    new_px = px - k0 * length
    new_x = (
        x
        + (jnp.sqrt(one_plus_delta * one_plus_delta - new_px * new_px - py * py) - pz)
        / k0
    )
    D = jnp.arcsin(A * px) - jnp.arcsin(A * new_px)
    new_y = y + (py / k0) * D
    delta_ell = (one_plus_delta / k0) * D
    return jnp.array(
        [new_x, new_px, new_y, py, zeta + (length - delta_ell / rvv), delta]
    )


def bend_field_kick_jax(state, klen, k0, h):
    """Dipole + weak-focusing kick: ``dpx = -k0*klen*(1 + h*x)`` (the ``h*x``
    term is the focusing the focus-free polar drift does not carry)."""
    x, px, y, py, zeta, delta = (
        state[0],
        state[1],
        state[2],
        state[3],
        state[4],
        state[5],
    )
    return jnp.array([x, px - k0 * klen * (1.0 + h * x), y, py, zeta, delta])


# ---------------------------------------------------------------------------
# Faithful field kick (port of track_magnet_kick_single_particle).
# ---------------------------------------------------------------------------
def _multipole_dpx_dpy(x, y, knl, ksl):
    """Accumulated (dpx, dpy) of a thin multipole, *without* the final dpx sign
    flip (the caller does ``px += -dpx`` / ``py += +dpy``). Fixed-shape arrays."""
    order = knl.shape[0] - 1
    inv_fact = 1.0
    for n in range(2, order + 1):
        inv_fact = inv_fact / n
    dpx = knl[order] * inv_fact
    dpy = ksl[order] * inv_fact
    idx = order
    while idx > 0:
        zre = dpx * x - dpy * y
        zim = dpx * y + dpy * x
        inv_fact = inv_fact * idx
        idx -= 1
        dpx = knl[idx] * inv_fact + zre
        dpy = ksl[idx] * inv_fact + zim
    return dpx, dpy


def magnet_kick_jax(
    state,
    kl,
    length,
    k0k,
    k1k,
    k2,
    k3,
    knl,
    ksl,
    h,
    k0_h_corr,
    k1_h_corr,
    rot_frame,
    beta0,
):
    """The body kick slot (port of track_magnet_kick.h): main multipole
    ``[k0k,k1k,k2,k3]`` + user ``knl/ksl``, the rot-frame dipole feed, and the
    ``k0_h``/``k1_h`` curvature corrections. ``kl`` = integrated kick length."""
    x, px, y, py, zeta, delta = (
        state[0],
        state[1],
        state[2],
        state[3],
        state[4],
        state[5],
    )
    kw = kl / length

    knl_main = jnp.array([k0k, k1k, k2, k3]) * length
    ksl_main = jnp.zeros(4)
    dpx_m, dpy_m = _multipole_dpx_dpy(x, y, knl_main, ksl_main)
    dpx_u, dpy_u = _multipole_dpx_dpy(x, y, knl, ksl)
    # kick_simple negates dpx only: px += -dpx*kw, py += +dpy*kw
    dpx = -(dpx_m + dpx_u) * kw
    dpy = (dpy_m + dpy_u) * kw

    dzeta = 0.0
    if rot_frame:
        hl = h * length * kw
        dpx = dpx + hl * (1.0 + delta)
        rv0v = 1.0 / rvv_from_delta(delta, beta0)
        dzeta = dzeta - rv0v * hl * x

    htot = h
    knl0_u = knl[0] if knl.shape[0] > 0 else 0.0
    knl1_u = knl[1] if knl.shape[0] > 1 else 0.0
    # k0_h (weak focusing): H = 1/2 h k0 x^2  -> dpx = -h k0 x
    dpx = dpx - (k0_h_corr * length + knl0_u) * kw * htot * x
    # k1_h: H = 1/3 h k1 x^3 - 1/2 h k1 x y^2
    k1l = (k1_h_corr * length + knl1_u) * kw * htot
    dpx = dpx + k1l * (-x * x + 0.5 * y * y)
    dpy = dpy + k1l * x * y

    return jnp.array([x, px + dpx, y, py + dpy, zeta + dzeta, delta])


# ---------------------------------------------------------------------------
# Functional integrators.
# Each takes drift_fn(state, dlength) and kick_fn(state, klength) and composes
# them over ``length`` with ``num_kicks`` kicks.  num_kicks is a *static* Python
# int (resolved per element at encode time) so the composition unrolls cleanly.
# ---------------------------------------------------------------------------
def integrate_yoshida4(state, drift_fn, kick_fn, length, num_kicks):
    """4th-order Yoshida (7-kick, MAD-NG coefficients)."""
    n_slices = math.ceil(num_kicks / _N_KICKS_YOSHIDA)  # ceil (round-up division)
    sl = length / n_slices
    kw = 1.0 / n_slices
    seq = [
        ("d", 0),
        ("k", 0),
        ("d", 1),
        ("k", 1),
        ("d", 2),
        ("k", 2),
        ("d", 3),
        ("k", 3),
        ("d", 3),
        ("k", 2),
        ("d", 2),
        ("k", 1),
        ("d", 1),
        ("k", 0),
        ("d", 0),
    ]
    for _ in range(n_slices):
        for kind, i in seq:
            if kind == "d":
                state = drift_fn(state, _D_YOSHIDA[i] * sl)
            else:
                state = kick_fn(state, _K_YOSHIDA[i] * kw * length)
    return state


def integrate_yoshida2(state, drift_fn, kick_fn, length, num_kicks):
    """2nd-order leapfrog (drift L/2, kick L, drift L/2) per kick."""
    sl = length / num_kicks
    for _ in range(num_kicks):
        state = drift_fn(state, 0.5 * sl)
        state = kick_fn(state, sl)
        state = drift_fn(state, 0.5 * sl)
    return state


def integrate_uniform(state, drift_fn, kick_fn, length, num_kicks):
    """Uniform splitting (xsuite INTEGRATOR==3) - same as leapfrog per slice."""
    return integrate_yoshida2(state, drift_fn, kick_fn, length, num_kicks)


def integrate_teapot(state, drift_fn, kick_fn, length, num_kicks):
    """TEAPOT splitting (xsuite INTEGRATOR==1)."""
    kw = 1.0 / num_kicks
    if num_kicks > 1:
        edge = 1.0 / (2 * (1 + num_kicks))
        inside = num_kicks / (num_kicks * num_kicks - 1.0)
    else:
        edge = 0.5
        inside = 0.0
    state = drift_fn(state, edge * length)
    for _ in range(num_kicks - 1):
        state = kick_fn(state, kw * length)
        state = drift_fn(state, inside * length)
    state = kick_fn(state, kw * length)
    state = drift_fn(state, edge * length)
    return state


_INTEGRATORS = {
    "yoshida4": integrate_yoshida4,
    "yoshida2": integrate_yoshida2,
    "uniform": integrate_uniform,
    "teapot": integrate_teapot,
}


def integrate(state, drift_fn, kick_fn, length, num_kicks, scheme):
    return _INTEGRATORS[scheme](state, drift_fn, kick_fn, length, num_kicks)


# ---------------------------------------------------------------------------
# Body builders (the drift slot swapped for the configured map).
# ---------------------------------------------------------------------------
def track_magnet_body_jax(
    state,
    length,
    drift_model,
    integrator,
    num_kicks,
    k0,
    k1,
    k2,
    k3,
    h,
    knl,
    ksl,
    beta0,
):
    """Integrate a thick-magnet body: the ``drift_model`` drift-slot map + the
    field kick (:func:`magnet_kick_jax`), composed by ``integrator`` over
    ``num_kicks``. ``drift_model``/``integrator``/``num_kicks`` are static
    (resolved by :func:`resolve_magnet`); the drift/kick strength split mirrors
    :func:`configure_tracking_model` (see jax_summary.md §2b)."""
    dm = drift_model
    if dm == 0:
        drift = lambda s, dl: drift_expanded_jax(s, dl, beta0)
        k0k, k1k, k0hc, k1hc, rot = k0, k1, k0, k1, 1
    elif dm == 1:
        drift = lambda s, dl: drift_exact_jax(s, dl, beta0)
        k0k, k1k, k0hc, k1hc, rot = k0, k1, k0, k1, 1
    elif dm == 2:
        drift = lambda s, dl: polar_drift_jax(s, dl, h, beta0)
        k0k, k1k, k0hc, k1hc, rot = k0, k1, k0, k1, 0
    elif dm == 7:  # rot-kick-rot (nested Yoshida bend)
        drift = lambda s, dl: rot_kick_rot_bend_jax(s, dl, k0, h, beta0)
        k0k, k1k, k0hc, k1hc, rot = 0.0, k1, k0, k1, 0
    elif dm == 3:
        drift = lambda s, dl: quad_combined_map_jax(s, dl, k0, k1, h, beta0)
        k0k, k1k, k0hc, k1hc, rot = 0.0, 0.0, 0.0, k1, 0
    elif dm == 4:
        drift = lambda s, dl: curved_bend_map_jax(s, dl, k0, h, beta0)
        k0k, k1k, k0hc, k1hc, rot = 0.0, k1, 0.0, k1, 0
    elif dm == 5:
        drift = lambda s, dl: straight_bend_jax(s, dl, k0, beta0)
        k0k, k1k, k0hc, k1hc, rot = 0.0, k1, 0.0, 0.0, 0
    else:
        drift = lambda s, dl: drift_exact_jax(s, dl, beta0)
        k0k, k1k, k0hc, k1hc, rot = k0, k1, k0, k1, 1

    def kick(s, kl):
        return magnet_kick_jax(
            s, kl, length, k0k, k1k, k2, k3, knl, ksl, h, k0hc, k1hc, rot, beta0
        )

    if num_kicks <= 0:  # no active kick -> thick drift only
        return drift(state, length)
    return integrate(state, drift, kick, length, num_kicks, integrator)


def track_element_jax(
    state,
    length,
    drift_model,
    integrator,
    num_kicks,
    k0,
    k1,
    k2,
    k3,
    h,
    knl,
    ksl,
    r21i,
    r43i,
    r21o,
    r43o,
    beta0,
):
    """``fringe -> body -> fringe`` for a thick magnet: the linear entry/exit
    dipole edges bracket the integrated body (:func:`track_magnet_body_jax`).
    Inactive faces pass r21 = r43 = 0 (a no-op)."""
    state = dipole_edge_linear_jax(state, r21i, r43i)
    state = track_magnet_body_jax(
        state,
        length,
        drift_model,
        integrator,
        num_kicks,
        k0,
        k1,
        k2,
        k3,
        h,
        knl,
        ksl,
        beta0,
    )
    state = dipole_edge_linear_jax(state, r21o, r43o)
    return state


def bend_body_jax(state, length, k0, h, beta0, num_kicks=1, scheme=None):
    """Curved pure-dipole body (used by the global/orbit backends).

    Curved bend (h != 0) -> drift_model 7 (nested rot-kick-rot),
    straight (h ~ 0) -> dm 2,
    ``scheme='exact'`` -> the single exact curved map.
    ``scheme`` selects the outer integrator (default ``BEND_SCHEME``)."""
    if scheme is None:
        scheme = BEND_SCHEME
    if scheme == "exact":
        return curved_bend_map_jax(state, length, k0, h, beta0)
    # dm 7 (curved, the faithful nested rot-kick-rot) unless h is a concrete 0
    # (a straight bend -> dm 2).  A *traced* h is always a real curved-bend body.
    try:
        straight = abs(float(h)) < 1e-12
    except (TypeError, jax.errors.ConcretizationTypeError):
        straight = False
    dm = 2 if straight else 7
    return track_magnet_body_jax(
        state,
        length,
        dm,
        scheme,
        num_kicks,
        k0,
        0.0,
        0.0,
        0.0,
        h,
        jnp.zeros(1),
        jnp.zeros(1),
        beta0,
    )


# ---------------------------------------------------------------------------
# Model resolution - what xsuite selects at runtime (ports of
# default_magnet_config.h, the adaptive resolution + auto-num_kicks in
# track_magnet.template.h, and track_magnet_configure.h).
# ---------------------------------------------------------------------------
# adaptive (model/integrator 0) -> these per element class
_DEFAULT_MODEL = {
    "Bend": 3,
    "RBend": 3,
    "Quadrupole": 4,
    "Sextupole": 6,
    "Octupole": 6,
    "Multipole": 6,
}
_DEFAULT_INTEGRATOR = {
    "Bend": 2,
    "RBend": 2,
    "Quadrupole": 3,
    "Sextupole": 3,
    "Octupole": 3,
    "Multipole": 3,
}
_INTEGRATOR_NAME = {1: "teapot", 2: "yoshida4", 3: "uniform"}
_MODEL_TO_INDEX = {
    "adaptive": 0,
    "bend-kick-bend": 2,
    "rot-kick-rot": 3,
    "mat-kick-mat": 4,
    "drift-kick-drift-exact": 5,
    "drift-kick-drift-expanded": 6,
    "rot-kick-rot-low-order": 7,
}
_INTEGRATOR_TO_INDEX = {"adaptive": 0, "teapot": 1, "yoshida4": 2, "uniform": 3}


def _as_model_index(cls, model):
    if isinstance(model, str):
        model = _MODEL_TO_INDEX.get(model, 0)
    if model in (0, 1):
        model = _DEFAULT_MODEL.get(cls, 4)
    return model


def _as_integrator_index(cls, integrator):
    if isinstance(integrator, str):
        integrator = _INTEGRATOR_TO_INDEX.get(integrator, 0)
    if integrator == 0:
        integrator = _DEFAULT_INTEGRATOR.get(cls, 3)
    return integrator


def resolve_num_kicks(cls, num_kicks, length, h, has_kicks):
    """Port of the auto-``num_multipole_kicks`` rule (track_magnet.template.h).

    ``num_kicks==0`` means auto: 0 if no active kick; 1 for a straight magnet;
    ``|L|/(2*pi/|h|)/0.5e-3`` (>=1) for a curved one (0.5 mrad per kick)."""
    if num_kicks and num_kicks > 0:
        return int(num_kicks)
    if not has_kicks:
        return 0
    if abs(h) < 1e-8:
        return 1
    b_circum = 2 * math.pi / abs(h)
    nk = int(abs(length) / b_circum / 0.5e-3)
    return max(nk, 1)


def configure_tracking_model(cls, model, k0, k1, h):
    """Port of track_magnet_configure.h: pick the drift model and split the
    strengths between the drift slot and the kick slot.

    Returns a dict with ``drift_model`` and the drift/kick strengths
    (``k0_drift,k1_drift,h_drift``, ``k0_kick,k1_kick,h_kick``,
    ``k0_h_correction,k1_h_correction``, ``kick_rot_frame``)."""
    model = _as_model_index(cls, model)
    h_is_zero = abs(h) < 1e-8
    if model == 2:  # bend-kick-bend
        drift_model = 5 if h_is_zero else 4
    elif model == 3:  # rot-kick-rot
        drift_model = 1 if h_is_zero else 7
    elif model == 4:  # mat-kick-mat
        drift_model = 3
    elif model == 5:  # drift-kick-drift-exact
        drift_model = 1
    elif model == 6:  # drift-kick-drift-expanded
        drift_model = 0
    elif model == 7:  # rot-kick-rot-low-order
        drift_model = 1 if h_is_zero else 2
    else:
        drift_model = 0
    out = dict(
        drift_model=drift_model,
        k0_drift=0.0,
        k1_drift=0.0,
        h_drift=0.0,
        k0_kick=0.0,
        k1_kick=0.0,
        h_kick=0.0,
        k0_h_correction=0.0,
        k1_h_correction=0.0,
        kick_rot_frame=0,
    )
    if drift_model in (0, 1):  # plain drift
        out.update(
            k0_kick=k0,
            k1_kick=k1,
            h_kick=h,
            k0_h_correction=k0,
            k1_h_correction=k1,
            kick_rot_frame=1,
        )
    elif drift_model == 2:  # polar drift
        out.update(
            h_drift=h,
            k0_kick=k0,
            k1_kick=k1,
            h_kick=h,
            k0_h_correction=k0,
            k1_h_correction=k1,
        )
    elif drift_model == 3:  # expanded dipole-quad
        out.update(k0_drift=k0, k1_drift=k1, h_drift=h, h_kick=h, k1_h_correction=k1)
    elif drift_model == 4:  # bend with h
        out.update(k0_drift=k0, h_drift=h, k1_kick=k1, h_kick=h, k1_h_correction=k1)
    elif drift_model == 5:  # bend without h
        out.update(k0_drift=k0, k1_kick=k1)
    elif drift_model == 7:  # rot-kick-rot (nested)
        out.update(
            k0_drift=k0,
            h_drift=h,
            k1_kick=k1,
            h_kick=h,
            k0_h_correction=k0,
            k1_h_correction=k1,
        )
    return out


def resolve_magnet(e):
    """Interpret a thick magnet the way xsuite tracking does: return the
    resolved (model, integrator name, drift_model, num_kicks) + the field/kick
    split.  Used by the encoders to dispatch to the right JAX map."""
    cls = type(e).__name__
    k0 = float(getattr(e, "k0", 0.0) or 0.0)
    k1 = float(getattr(e, "k1", 0.0) or 0.0)
    k2 = float(getattr(e, "k2", 0.0) or 0.0)
    k3 = float(getattr(e, "k3", 0.0) or 0.0)
    h = float(getattr(e, "h", 0.0) or 0.0)
    length = float(getattr(e, "length", 0.0) or 0.0)
    model_idx = _as_model_index(cls, getattr(e, "model", 0))
    integ_idx = _as_integrator_index(cls, getattr(e, "integrator", 0))
    has_kicks = any(abs(v) > 0 for v in (k0, k1, k2, k3, h))
    knl = np.asarray(getattr(e, "knl", [0.0]), dtype=float)
    ksl = np.asarray(getattr(e, "ksl", [0.0]), dtype=float)
    has_kicks = has_kicks or bool(np.any(knl != 0.0)) or bool(np.any(ksl != 0.0))
    num_kicks = resolve_num_kicks(
        cls, int(getattr(e, "num_multipole_kicks", 0) or 0), length, h, has_kicks
    )
    cfg = configure_tracking_model(cls, getattr(e, "model", 0), k0, k1, h)
    cfg.update(
        cls=cls,
        model=model_idx,
        integrator=_INTEGRATOR_NAME[integ_idx],
        num_kicks=num_kicks,
        length=length,
        k0=k0,
        k1=k1,
        k2=k2,
        k3=k3,
        h=h,
    )
    return cfg


# ---------------------------------------------------------------------------
# Section encoding (struct-of-arrays): every thick magnet is dispatched by its
# resolved (drift_model, integrator, num_kicks) *signature*. Parallel to (not
# shared with) the etype-based encoding in jax_optics.
# ---------------------------------------------------------------------------
def thin_multipole_kick_jax(state, knl, ksl):
    """Thin multipolar kick from integrated normal/skew strengths knl/ksl."""
    x, px, y, py, zeta, delta = (
        state[0],
        state[1],
        state[2],
        state[3],
        state[4],
        state[5],
    )
    order = knl.shape[0] - 1
    inv_fact = 1.0
    for n in range(2, order + 1):
        inv_fact = inv_fact / n
    dpx = knl[order] * inv_fact
    dpy = ksl[order] * inv_fact
    idx = order
    while idx > 0:
        zre = dpx * x - dpy * y
        zim = dpx * y + dpy * x
        inv_fact = inv_fact * idx
        idx -= 1
        dpx = knl[idx] * inv_fact + zre
        dpy = ksl[idx] * inv_fact + zim
    return jnp.array([x, px - dpx, y, py + dpy, zeta, delta])


# Dispatch codes: fixed leaves (identity / merged-drift / thin multipole), then
# one code per unique magnet signature present, starting at DISP_MAGNET0.
DISP_IDENT, DISP_DRIFT, DISP_THIN, DISP_MAGNET0 = 0, 1, 2, 3

_MAGNET_CLASSES = ("Bend", "RBend", "Quadrupole", "Sextupole", "Octupole", "Multipole")


def _magnet_active(e):
    """True if a thick magnet carries any field (so it needs a real body map)."""
    for a in ("k0", "k1", "k2", "k3", "h"):
        if float(getattr(e, a, 0.0) or 0.0) != 0.0:
            return True
    knl = np.asarray(getattr(e, "knl", [0.0]), dtype=float)
    ksl = np.asarray(getattr(e, "ksl", [0.0]), dtype=float)
    return bool(np.any(knl != 0.0) or np.any(ksl != 0.0))


class Enc(NamedTuple):
    disp: jnp.ndarray  # dispatch code (DISP_* / DISP_MAGNET0 + sig index)
    L: jnp.ndarray
    k0: jnp.ndarray
    k1fix: jnp.ndarray  # baked k1 (overridden by the kq vector when varied)
    k2: jnp.ndarray
    k3: jnp.ndarray
    h: jnp.ndarray
    kqidx: jnp.ndarray  # index into the kq vector, or -1 if not varied
    knl: jnp.ndarray  # (N, W) integrated multipole strengths
    ksl: jnp.ndarray
    r21i: jnp.ndarray  # linear dipole-edge coeffs (Bend/RBend faces, else 0)
    r43i: jnp.ndarray
    r21o: jnp.ndarray
    r43o: jnp.ndarray


def encode_section(line, ordered_names, kq_index):
    """Encode every section element into an ``Enc`` (struct-of-arrays).

    Active thick magnets carry their ``resolve_magnet`` signature + strengths +
    padded multipoles + edge coeffs; inactive thick -> drift; thin -> full
    multipole kick; markers -> identity. Magnets in ``kq_index`` read k1 from the
    live kq vector. Returns ``(enc, sigs)`` (unique signatures in code order)."""
    ed = line.element_dict

    # Find the widest knl/ksl across the section so every row's multipole
    # arrays can be zero-padded to a common width W (struct-of-arrays needs
    # rectangular columns). Floor of 4 keeps room for the usual k0..k3 orders.
    W = 4
    for nm in ordered_names:
        e = ed.get(nm)
        if e is None:
            continue
        W = max(
            W,
            len(np.asarray(getattr(e, "knl", [0.0]))),
            len(np.asarray(getattr(e, "ksl", [0.0]))),
        )

    sig_index, sigs = {}, []

    # Intern a magnet signature into a dispatch code: the first time a given
    # (drift_model, integrator, num_kicks, is_bend) tuple is seen it gets the
    # next code DISP_MAGNET0 + k and is appended to ``sigs``; repeats reuse it.
    # This lets every magnet sharing a signature run the same lax.switch branch.
    def sig_disp(sig):
        if sig not in sig_index:
            sig_index[sig] = DISP_MAGNET0 + len(sigs)
            sigs.append(sig)
        return sig_index[sig]

    cols = {k: [] for k in ("disp", "L", "k0", "k1", "k2", "k3", "h", "kq")}
    knl_rows, ksl_rows, edge_rows = [], [], []
    for nm in ordered_names:
        e = ed.get(nm)
        cls = type(e).__name__ if e is not None else None
        length = float(getattr(e, "length", 0.0) or 0.0)
        # Row defaults: an identity (no-op) row carrying no strengths and no
        # live-kq link (kqi == -1). Each branch below overrides what it needs.
        disp, L, k0, k1f, k2, k3, h, kqi = DISP_IDENT, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1
        knl = np.zeros(W)
        ksl = np.zeros(W)
        edges = (0.0, 0.0, 0.0, 0.0)
        if e is None:
            pass
        # Active thick magnet: encode it verbatim under its signature's code.
        elif (
            cls in _MAGNET_CLASSES
            and length > 0.0
            and (nm in kq_index or _magnet_active(e))
        ):
            cfg = resolve_magnet(e)
            is_bend = cls in ("Bend", "RBend")
            disp = sig_disp(
                (cfg["drift_model"], cfg["integrator"], int(cfg["num_kicks"]), is_bend)
            )
            L, k0, k1f, k2, k3, h = (
                cfg["length"],
                cfg["k0"],
                cfg["k1"],
                cfg["k2"],
                cfg["k3"],
                cfg["h"],
            )
            uknl = np.asarray(getattr(e, "knl", [0.0]), dtype=float)
            uksl = np.asarray(getattr(e, "ksl", [0.0]), dtype=float)
            knl[: len(uknl)] = uknl
            ksl[: len(uksl)] = uksl
            if is_bend:
                edges = bend_edge_coeffs(e)
            # Link this row to the live kq vector so jacfwd reaches its k1.
            if nm in kq_index:
                kqi = kq_index[nm]
        # Inactive thick element (off magnet, etc.): collapse to a pure drift.
        elif bool(getattr(e, "isthick", False)) and length > 0.0:
            disp, L = DISP_DRIFT, length
        # Thin element: a single multipole kick if it carries any strength,
        # otherwise it stays the default identity row.
        else:
            uknl = np.asarray(getattr(e, "knl", [0.0]), dtype=float)
            uksl = np.asarray(getattr(e, "ksl", [0.0]), dtype=float)
            if np.any(uknl != 0.0) or np.any(uksl != 0.0):
                disp = DISP_THIN
                knl[: len(uknl)] = uknl
                ksl[: len(uksl)] = uksl
        for k, v in zip(
            ("disp", "L", "k0", "k1", "k2", "k3", "h", "kq"),
            (disp, L, k0, k1f, k2, k3, h, kqi),
        ):
            cols[k].append(v)
        knl_rows.append(knl)
        ksl_rows.append(ksl)
        edge_rows.append(edges)

    er = np.array(edge_rows, dtype=float) if edge_rows else np.zeros((0, 4))
    enc = Enc(
        jnp.array(cols["disp"], dtype=jnp.int32),
        jnp.array(cols["L"]),
        jnp.array(cols["k0"]),
        jnp.array(cols["k1"]),
        jnp.array(cols["k2"]),
        jnp.array(cols["k3"]),
        jnp.array(cols["h"]),
        jnp.array(cols["kq"], dtype=jnp.int32),
        jnp.array(np.array(knl_rows)),
        jnp.array(np.array(ksl_rows)),
        jnp.array(er[:, 0]),
        jnp.array(er[:, 1]),
        jnp.array(er[:, 2]),
        jnp.array(er[:, 3]),
    )
    return enc, sigs


def compress_encoding(enc, boundary_rows):
    """Drop identity rows and merge runs of consecutive drifts.

    Boundary rows are target positions that must not be merged away, so that
    scan output index i still corresponds to the optics after a specific element.
    Returns (compressed Enc, orig_to_comp) where ``orig_to_comp[i]`` is the
    compressed row whose output holds the optics *after* original element i.
    Magnet rows are untouched, so the ``sigs`` list still applies. Mirror of
    ``jax_optics.compress_encoding``.
    """
    disp = np.asarray(enc.disp)
    L = np.asarray(enc.L)
    n = len(disp)
    boundary = {int(b) for b in boundary_rows}

    # Plan the kept rows: each is either ("copy", original_index) or a merged
    # ("drift", accumulated_length).  Build the actual arrays in a second pass.
    plan = []
    orig_to_comp = np.empty(n, dtype=np.int64)
    acc = 0.0

    def flush():
        nonlocal acc
        if acc > 0.0:
            plan.append(("drift", acc))
            acc = 0.0

    for i in range(n):
        if disp[i] == DISP_IDENT:
            pass  # drop no-op markers
        elif disp[i] == DISP_DRIFT:
            acc += float(L[i])  # accumulate into the running drift
        else:
            flush()
            plan.append(("copy", i))  # a magnet / thin kick: keep it verbatim
        if i in boundary:
            flush()  # a read point must keep its own output row
        orig_to_comp[i] = len(plan) - 1
    flush()

    # Materialise: a copied row keeps every field; a merged drift is a plain
    # drift (DISP_DRIFT + its length, all other fields at their defaults).
    W = np.asarray(enc.knl).shape[1] if n else 4

    def field(name):
        a = np.asarray(getattr(enc, name))
        zero = np.zeros(W) if a.ndim == 2 else 0
        return np.array([a[i] if kind == "copy" else zero for kind, i in plan])

    disp_c = np.array(
        [DISP_DRIFT if kind == "drift" else int(disp[i]) for kind, i in plan]
    )
    L_c = np.array([v if kind == "drift" else float(L[v]) for kind, v in plan])

    enc_c = Enc(
        jnp.array(disp_c, dtype=jnp.int32),
        jnp.array(L_c),
        jnp.array(field("k0")),
        jnp.array(field("k1fix")),
        jnp.array(field("k2")),
        jnp.array(field("k3")),
        jnp.array(field("h")),
        jnp.array(field("kqidx"), dtype=jnp.int32),
        jnp.array(field("knl")),
        jnp.array(field("ksl")),
        jnp.array(field("r21i")),
        jnp.array(field("r43i")),
        jnp.array(field("r21o")),
        jnp.array(field("r43o")),
    )
    return enc_c, orig_to_comp


# ---------------------------------------------------------------------------
# Optics provider for the matching backend (``use_jax_integrators=True``): the
# counterpart of ``jax_optics.build_section_twiss``, with a live kq vector so
# jacfwd over the varied-quad strengths works.
# ---------------------------------------------------------------------------
def _branch_list(sigs, beta0):
    """The per-row element maps, indexed by ``disp`` code: identity / drift /
    thin multipole, then one ``track_element_jax`` branch per magnet signature.
    A magnet reads k1 from the live ``kq`` vector when ``kqidx >= 0``, else from
    ``k1fix``."""

    def _make_magnet(sig):
        dm, integ, nk, is_bend = sig
        if is_bend:
            integ = BEND_SCHEME

        def branch(s, e, kq):
            k1 = jnp.where(e.kqidx >= 0, kq[jnp.maximum(e.kqidx, 0)], e.k1fix)
            return track_element_jax(
                s, e.L, dm, integ, nk,
                e.k0, k1, e.k2, e.k3, e.h, e.knl, e.ksl,
                e.r21i, e.r43i, e.r21o, e.r43o, beta0,
            )

        return branch

    return [
        lambda s, e, kq: s,  # DISP_IDENT
        lambda s, e, kq: drift_exact_jax(s, e.L, beta0),  # DISP_DRIFT
        lambda s, e, kq: thin_multipole_kick_jax(s, e.knl, e.ksl),  # DISP_THIN
    ] + [_make_magnet(sig) for sig in sigs]


def build_emap(enc, sigs, beta0):
    """Return ``emap(state, row, kq)`` dispatching one ``Enc`` row on its
    ``disp`` code via ``lax.switch`` (used by the sequential orbit scan, where a
    switch is a real conditional - only the matching branch runs per step)."""
    branches = _branch_list(sigs, beta0)

    def emap(state, row, kq):
        return lax.switch(row.disp, branches, state, row, kq)

    return emap


def build_section_twiss(enc, sigs, beta0, p0, s0):
    """Factorized Twiss target builder; drop-in for
    ``jax_optics.build_section_twiss``. Returns ``targets(kq, rows, qidx)``:
    frozen-orbit scan -> per-element R -> Twiss scan -> gather ``[rows, qidx]``.
    ``hist[i]`` is the optics *after* row ``i`` (the ``name_to_row = i-1``
    convention). R is built grouped by ``disp`` code so each row runs only its
    own map (avoiding the vmap(lax.switch) trap)."""
    emap = build_emap(enc, sigs, beta0)
    branches = _branch_list(sigs, beta0)
    disp = np.asarray(enc.disp)
    # static partition of the rows by dispatch code (known at trace time)
    groups = {int(d): np.where(disp == d)[0] for d in np.unique(disp)}
    n_rows = len(disp)

    def all_params(kq):
        # Orbit: a sequential scan - the switch here dispatches one branch/step.
        def obody(s, e):
            return emap(s, e, kq), s

        _, orbits_in = lax.scan(obody, s0, enc)
        orbits_in = lax.stop_gradient(orbits_in)

        # Per-element R = jacfwd of each row's own map at its frozen orbit,
        # computed group-by-group so that no row runs a branch it does not use.
        Rs = jnp.zeros((n_rows, 6, 6))
        for d, idx in groups.items():
            branch = branches[d]
            sub_enc = jax.tree_util.tree_map(lambda a: a[idx], enc)
            sub_R = jax.vmap(
                lambda e, o: jax.jacfwd(lambda s: branch(s, e, kq))(o)
            )(sub_enc, orbits_in[idx])
            Rs = Rs.at[idx].set(sub_R)

        def tbody(params, R):
            pn = propagate_twiss(R, params)
            return pn, pn

        _, hist = lax.scan(tbody, p0, Rs)
        return hist

    @jax.jit
    def targets(kq, rows, qidx):
        return all_params(kq)[rows, qidx]

    return targets
