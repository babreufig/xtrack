"""Unified exact-physics JAX Jacobian backend for ``line.match(use_jax=True)``.

The target categories are handled inside the ``JaxJacobian`` class, which
dispatches by the target types/quantities to build the appropriate observable
for the JAX differentiation. The categories are:
  * "optics": twiss parameters at named places in the line, plus relative phase
    advance between two places (the "TRP" target, which is a difference of two
    optics targets).
  * "global": global quantities derived from the one-turn matrix, currently
    limited to tunes and chromaticities (qx, qy, dqx, dqy).
  * "orbit": single-pass trajectory (x, px, y, py) at named places.

Set the environment variable ``XT_JAX_CACHE_DIR`` to enable JAX's XLA compile
cache (compile once, reuse across runs/matches of the same structure).
"""

import os

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from .jax_optics import (
    encode_section,
    compress_encoding,
    build_section_twiss,
    pack_rows,
    emap_jax,
    scalar_chain,
    kick_chain,
    TW_INDEX,
    ET_DRIFT,
    ET_QUAD,
    ET_BEND,
    ET_SEXT,
    ET_KICK,
    ET_CORR,
    ET_IDENT,
)

jax.config.update("jax_enable_x64", True)

# Opt-in persistent XLA compile cache (the global backend is compile-bound).
_CACHE_DIR = os.environ.get("XT_JAX_CACHE_DIR")
if _CACHE_DIR:
    jax.config.update("jax_compilation_cache_dir", _CACHE_DIR)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)

GLOBAL_QTYS = ("qx", "qy", "dqx", "dqy")
ORBIT_QTYS = ("x", "px", "y", "py")


class JaxTwissResult:
    """Minimal twiss-like result holding JAX-computed values at named places.

    Supports ``res[quantity, place]`` (and integer ``0``/``-1`` resolving to the
    section start/end) so it can stand in for the xtrack twiss table when the
    matching merit function evaluates its targets - the residual is then read
    from the exact JAX maps instead of a fresh twiss, while every other piece of
    the merit function logic (transforms, tolerances, weights) runs unchanged.
    """

    def __init__(self, table, start_name, end_name):
        self._t = table
        self._start = start_name
        self._end = end_name

    def __getitem__(self, key):
        quantity, place = key
        if isinstance(place, (int, np.integer)):
            place = self._start if place == 0 else (self._end if place == -1 else place)
        return self._t[(quantity, place)]


_OPTICS_QTYS = ("betx", "bety", "alfx", "alfy", "mux", "muy", "dx", "dy", "dpx", "dpy")
_COORD = {"x": 0, "px": 1, "y": 2, "py": 3}


class JaxJacobian:
    """Exact-physics ``d(target)/d(knob)`` Jacobian, dispatched by ``kind``.

    ``jacobian()`` returns the unweighted ``(n_target, n_vary)`` matrix at the
    line's current strengths (weights are applied by the caller).  Built once
    (encode + XLA compile) and reused every Newton step.
    """

    def __init__(self, line, tw, kind, targets, vary_names, TargetRelPhaseAdvance):
        self.line = line
        self._ed = line.element_dict
        self.kind = kind
        self.vary_names = list(vary_names)
        self.beta0 = float(tw.particle_on_co.beta0[0])
        self.ordered = list(tw.name[:-1])
        self.n_vary = len(vary_names)
        # Optional residual (primal) evaluator; set only where implemented
        # (optics).  None -> the caller keeps using the normal twiss residual.
        self._values = None
        {
            "optics": self._build_optics,
            "global": self._build_global,
            "orbit": self._build_orbit,
        }[kind](tw, targets, TargetRelPhaseAdvance)

    def jacobian(self):
        return self._compute()

    def has_values(self):
        """True if this backend can evaluate target values (the residual)."""
        return self._values is not None

    def values(self):
        """JAX-computed target values at the line's current strengths.

        Returns a :class:`JaxTwissResult` usable in place of the twiss table.
        The primal forward pass is far cheaper than a section twiss, which is
        what makes a JAX residual worthwhile.
        """
        if self._values is None:
            raise NotImplementedError(
                "JAX residual is only implemented for optics targets"
            )
        return self._values()

    # -- shared driver: compile jacfwd-over-knobs of an observable -----------
    def _finalize(self, observable, p0):
        jac_fn = jax.jit(jax.jacfwd(observable, argnums=1))
        jac_fn(jnp.asarray(p0()), jnp.zeros(self.n_vary))  # warm compile
        n_vary = self.n_vary

        def compute():
            return np.array(jac_fn(jnp.asarray(p0()), jnp.zeros(n_vary)))

        self._compute = compute

    # -- optics: twiss at places + relative phase advance -------------------
    def _build_optics(self, tw, targets, TRP):
        ed = self._ed
        ordered = self.ordered
        p0 = jnp.array(
            [
                tw["betx", 0],
                tw["alfx", 0],
                tw["mux", 0],
                tw["bety", 0],
                tw["alfy", 0],
                tw["muy", 0],
                tw["dx", 0],
                tw["dpx", 0],
                tw["dy", 0],
                tw["dpy", 0],
            ]
        )
        s0 = jnp.array(
            [
                tw["x", 0],
                tw["px", 0],
                tw["y", 0],
                tw["py", 0],
                tw["zeta", 0],
                tw["delta", 0],
            ]
        )

        quads = [n for n in ordered if type(ed.get(n)).__name__ == "Quadrupole"]
        varied, dk1 = scalar_chain(self.line, self.vary_names, "k1", quads)
        kq_index = {q: i for i, q in enumerate(varied)}
        enc = encode_section(self.line, ordered, kq_index)

        # atoms (unique quantity@place) + per-target sign combos
        section_start, section_end = tw.name[0], tw.name[-2]
        atom_index, atoms, combos = {}, [], []

        def atom_id(q, place):
            key = (q, place)
            if key not in atom_index:
                atom_index[key] = len(atoms)
                atoms.append(key)
            return atom_index[key]

        for tt in targets:
            if isinstance(tt, TRP):
                q = tt.var
                end = section_end if tt.end == "__ele_stop__" else tt.end
                combo = [(1.0, atom_id(q, end))]
                if tt.start not in ("__ele_start__", section_start):
                    combo.append((-1.0, atom_id(q, tt.start)))
                combos.append(combo)
            else:
                quantity, place = tt.tar
                combos.append([(1.0, atom_id(quantity, place))])

        # optics "at place" = optics after the preceding element; entrance
        # places are knob-independent (zero derivative).
        name_to_row = {nm: i - 1 for i, nm in enumerate(ordered)}
        orig_rows, atom_zero = [], []
        for q, place in atoms:
            r = name_to_row[place]
            atom_zero.append(r < 0)
            orig_rows.append(max(r, 0))
        atom_zero = np.array(atom_zero, dtype=bool)

        cenc, orig_to_comp = compress_encoding(enc, set(orig_rows))
        rows = jnp.array([int(orig_to_comp[r]) for r in orig_rows], dtype=jnp.int32)
        qidx = jnp.array([TW_INDEX[q] for q, _ in atoms], dtype=jnp.int32)
        targets_fn = build_section_twiss(cenc, self.beta0, p0, s0)
        jac_fn = jax.jit(jax.jacfwd(lambda kq: targets_fn(kq, rows, qidx)))
        jac_fn(jnp.asarray([float(ed[q].k1) for q in varied]))  # warm

        n_target = len(targets)

        def compute():
            kq = jnp.asarray([float(ed[q].k1) for q in varied])
            ja = np.asarray(jac_fn(kq)) @ dk1  # (n_atom, n_vary)
            ja[atom_zero, :] = 0.0
            jac = np.zeros((n_target, self.n_vary))
            for it, combo in enumerate(combos):
                for sign, ai in combo:
                    jac[it] += sign * ja[ai]
            return jac

        self._compute = compute

        # --- residual (primal target values) -------------------------------
        # Every (quantity, place) any target's eval reads - a superset of the
        # jacobian atoms (it also includes phase-advance start places, whose
        # derivative is zero but whose value the residual still needs).
        val_index, val_keys = {}, []

        def val_id(q, place):
            if (q, place) not in val_index:
                val_index[(q, place)] = len(val_keys)
                val_keys.append((q, place))

        for tt in targets:
            if isinstance(tt, TRP):
                q = tt.var
                end = section_end if tt.end == "__ele_stop__" else tt.end
                start = section_start if tt.start == "__ele_start__" else tt.start
                val_id(q, end)
                val_id(q, start)
            else:
                quantity, place = tt.tar
                val_id(quantity, place)

        # Places at the section entrance (row < 0) are knob-independent; their
        # value is the fixed initial optics p0, read directly (not from a scan
        # row, which would hold the optics *after* the first element).
        p0_np = np.asarray(p0)
        v_rows_orig = [name_to_row[place] for _, place in val_keys]
        v_entrance = np.array([r < 0 for r in v_rows_orig], dtype=bool)
        v_entrance_val = np.array([p0_np[TW_INDEX[q]] for q, _ in val_keys])
        v_rows = jnp.array(
            [int(orig_to_comp[max(r, 0)]) for r in v_rows_orig], dtype=jnp.int32
        )
        v_qidx = jnp.array([TW_INDEX[q] for q, _ in val_keys], dtype=jnp.int32)
        val_fn = jax.jit(lambda kq: targets_fn(kq, v_rows, v_qidx))
        val_fn(jnp.asarray([float(ed[q].k1) for q in varied]))  # warm

        def values():
            kq = jnp.asarray([float(ed[q].k1) for q in varied])
            v = np.array(val_fn(kq))  # writable copy
            v[v_entrance] = v_entrance_val[v_entrance]
            table = {key: float(vi) for key, vi in zip(val_keys, v)}
            return JaxTwissResult(table, section_start, section_end)

        self._values = values

    # -- global: tune / chromaticity from the one-turn matrix ---------------
    def _build_global(self, tw, targets, TRP):
        ed = self._ed
        ordered = self.ordered
        beta0 = self.beta0
        want = [tt.tar for tt in targets]
        svals = np.asarray(tw.s)
        Dx = np.asarray(tw.dx)[:-1]
        Dpx = np.asarray(tw.dpx)[:-1]
        Dy = np.asarray(tw.dy)[:-1]
        Dpy = np.asarray(tw.dpy)[:-1]

        quads = [n for n in ordered if type(ed.get(n)).__name__ == "Quadrupole"]
        sexts = [n for n in ordered if type(ed.get(n)).__name__ == "Sextupole"]
        varied_q, dp_q = scalar_chain(self.line, self.vary_names, "k1", quads)
        varied_s, dp_s = scalar_chain(self.line, self.vary_names, "k2", sexts)
        n_vq = len(varied_q)
        qpos = {n: i for i, n in enumerate(varied_q)}
        spos = {n: i for i, n in enumerate(varied_s)}
        dp_dknob = jnp.asarray(
            np.vstack([dp_q, dp_s])
            if (n_vq or varied_s)
            else np.zeros((0, self.n_vary))
        )

        # Encode all quads + all sextupoles (fixed ones carry baked strengths),
        # one row per element, then merge drifts (no boundaries -> full merge).
        # Edges OFF (bend body only) is this backend's model: encode no edge
        # coeffs, so emap_jax's bend_with_edges reduces to a plain bend.
        rows = []
        for i, n in enumerate(ordered):
            e = ed.get(n)
            cls = type(e).__name__ if e is not None else None
            span = float(svals[i + 1] - svals[i])
            D = {"Dx": Dx[i], "Dpx": Dpx[i], "Dy": Dy[i], "Dpy": Dpy[i]}
            if cls == "Quadrupole" and float(e.k1) != 0.0:
                rows.append(
                    {
                        "etype": ET_QUAD,
                        "L": float(e.length),
                        "k1fix": float(e.k1),
                        "kqidx": qpos.get(n, -1),
                        **D,
                    }
                )
            elif cls in ("Bend", "RBend"):
                rows.append(
                    {
                        "etype": ET_BEND,
                        "L": float(e.length),
                        "k0": float(e.k0),
                        "h": float(e.h),
                        **D,
                    }
                )
            elif cls == "Sextupole":
                k2f = 0.0 if n in spos else float(e.k2)
                rows.append(
                    {
                        "etype": ET_SEXT,
                        "L": float(e.length),
                        "k2fix": k2f,
                        "ksidx": spos.get(n, -1),
                        **D,
                    }
                )
            elif e is not None and not getattr(e, "isthick", False):
                knl = np.asarray(getattr(e, "knl", [0.0]), dtype=float)
                ksl = np.asarray(getattr(e, "ksl", [0.0]), dtype=float)
                kn1 = float(knl[1]) if len(knl) > 1 else 0.0
                ks1 = float(ksl[1]) if len(ksl) > 1 else 0.0
                if kn1 or ks1:
                    rows.append({"etype": ET_KICK, "knl1": kn1, "ksl1": ks1, **D})
                elif span > 0.0:
                    rows.append({"etype": ET_DRIFT, "L": span})
                else:
                    rows.append({"etype": ET_IDENT})
            elif span > 0.0:
                rows.append({"etype": ET_DRIFT, "L": span})
            else:
                rows.append({"etype": ET_IDENT})
        enc, _ = compress_encoding(pack_rows(rows), set())
        need_chroma = any(q in ("dqx", "dqy") for q in want)

        def tunes(kq, ks, delta):
            kqp = jnp.concatenate([kq, jnp.zeros(1)])
            ksp = jnp.concatenate([ks, jnp.zeros(1)])
            corr0 = jnp.zeros(1)  # no correctors in the global backend

            # Factorized: one vmapped jacfwd for every element's R (parallel),
            # then the one-turn product via parallel scan.
            def Rof(e):
                pt = jnp.array(
                    [e.Dx * delta, e.Dpx * delta, e.Dy * delta, e.Dpy * delta]
                )

                def f(t):
                    s = jnp.array([t[0], t[1], t[2], t[3], 0.0, delta])
                    return emap_jax(s, e, kqp, ksp, corr0, beta0)[0:4]

                return jax.jacfwd(f)(pt)

            Rs = jax.vmap(Rof)(enc)  # (N, 4, 4)
            M = lax.associative_scan(lambda Al, Ar: Ar @ Al, Rs)[-1]
            qx = jnp.arccos(jnp.clip(0.5 * (M[0, 0] + M[1, 1]), -1.0, 1.0)) / (
                2 * jnp.pi
            )
            qy = jnp.arccos(jnp.clip(0.5 * (M[2, 2] + M[3, 3]), -1.0, 1.0)) / (
                2 * jnp.pi
            )
            return jnp.array([qx, qy])

        def observable(p_base, dknob):
            p = p_base + dp_dknob @ dknob
            kq, ks = p[:n_vq], p[n_vq:]
            q0 = tunes(kq, ks, 0.0)
            ch = (
                jax.jacfwd(lambda d: tunes(kq, ks, d))(0.0)
                if need_chroma
                else jnp.zeros(2)
            )
            lut = {"qx": q0[0], "qy": q0[1], "dqx": ch[0], "dqy": ch[1]}
            return jnp.array([lut[q] for q in want])

        def p0():
            return np.array(
                [float(ed[n].k1) for n in varied_q]
                + [float(ed[n].k2) for n in varied_s]
            )

        self._finalize(observable, p0)

    # -- orbit: single-pass trajectory at named places ----------------------
    def _build_orbit(self, tw, targets, TRP):
        ed = self._ed
        ordered = self.ordered
        beta0 = self.beta0
        want = [tt.tar for tt in targets]
        svals = np.asarray(tw.s)
        s0 = jnp.array(
            [
                tw["x", 0],
                tw["px", 0],
                tw["y", 0],
                tw["py", 0],
                tw["zeta", 0],
                tw["delta", 0],
            ]
        )

        params, hidx, vidx, dp = kick_chain(self.line, self.vary_names, ordered)
        dp_dknob = jnp.asarray(dp)
        corr_names = set(hidx) | set(vidx)

        # One row per element (NO merging -> place lookup stays index-aligned).
        # Edges OFF (bend body only): encode no edge coeffs, so emap_jax's
        # bend_with_edges reduces to a plain bend.
        rows = []
        for i, n in enumerate(ordered):
            e = ed.get(n)
            cls = type(e).__name__ if e is not None else None
            span = float(svals[i + 1] - svals[i])
            knl = np.asarray(getattr(e, "knl", [0.0]), dtype=float)
            ksl = np.asarray(getattr(e, "ksl", [0.0]), dtype=float)
            kn1 = float(knl[1]) if len(knl) > 1 else 0.0
            ks1 = float(ksl[1]) if len(ksl) > 1 else 0.0
            if n in corr_names:
                rows.append(
                    {
                        "etype": ET_CORR,
                        "L": span,
                        "hidx": hidx.get(n, -1),
                        "vidx": vidx.get(n, -1),
                    }
                )
            elif cls == "Quadrupole" and float(e.k1) != 0.0:
                rows.append(
                    {"etype": ET_QUAD, "L": float(e.length), "k1fix": float(e.k1)}
                )
            elif cls in ("Bend", "RBend"):
                rows.append(
                    {
                        "etype": ET_BEND,
                        "L": float(e.length),
                        "k0": float(e.k0),
                        "h": float(e.h),
                    }
                )
            elif e is not None and not getattr(e, "isthick", False) and (kn1 or ks1):
                rows.append({"etype": ET_KICK, "knl1": kn1, "ksl1": ks1})
            else:
                rows.append({"etype": ET_DRIFT, "L": max(span, 0.0)})
        enc = pack_rows(rows)

        name_to_row = {nm: i for i, nm in enumerate(ordered)}
        rows_idx = jnp.array([name_to_row[place] for _, place in want], dtype=jnp.int32)
        coord_idx = jnp.array([_COORD[q] for q, _ in want], dtype=jnp.int32)

        def observable(p_base, dknob):
            # corrector kicks are the only live vector; quads/sexts are baked in.
            corr = jnp.concatenate([p_base + dp_dknob @ dknob, jnp.zeros(1)])
            kq0 = jnp.zeros(1)
            ks0 = jnp.zeros(1)

            def body(s, e):
                sn = emap_jax(s, e, kq0, ks0, corr, beta0)
                return sn, sn

            _, hist = lax.scan(body, s0, enc)
            return hist[rows_idx, coord_idx]

        def p0():
            def kick(n, c):
                arr = getattr(ed.get(n), c, None)
                return float(arr[0]) if (arr is not None and len(arr)) else 0.0

            return np.array([kick(n, c) for (n, c) in params])

        self._finalize(observable, p0)


# ===========================================================================
# Target classification (shared by match.py dispatch).
# ===========================================================================
def classify_jax_targets(targets, TargetRelPhaseAdvance):
    """Return 'optics' | 'global' | 'orbit' if all targets share one supported
    category, else None."""
    categories = set()
    for tt in targets:
        if isinstance(tt, TargetRelPhaseAdvance):
            categories.add("optics")
        elif isinstance(tt.tar, tuple):
            q = tt.tar[0]
            if q in _OPTICS_QTYS:
                categories.add("optics")
            elif q in ORBIT_QTYS:
                categories.add("orbit")
            else:
                return None
        else:
            if tt.tar in GLOBAL_QTYS:
                categories.add("global")
            else:
                return None
    return categories.pop() if len(categories) == 1 else None
