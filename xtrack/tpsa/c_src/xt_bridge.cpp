/* Bridge entry points: run xtrack element physics on a single particle whose
 * coordinates are TPSAs (XT_FLAVOR_TPSA) or plain doubles (XT_FLAVOR_NUM).
 * One source, compiled twice. The element loop and the typeid switch live in C, so a
 * whole line is one Python->C call, as in Xtrack. All field-shaped and cross-boundary code is
 * generated (gen_bridge.py), this file glues them together. */

#include "xt_local_particle.hpp"                        /* flavor select + LocalParticle struct + glue */

/* the generated C-API uses the C99 keyword `restrict`, which does not exist in C++,
 * but is supported by most C++ compilers. */
#define restrict __restrict
#include "generated/xt_element_capi.h"                  /* <El>Data + ElementRefData + XtBridgeParticle */
#include "generated/xt_local_particle_gen.hpp"          /* field accessors (via XtBridgeParticle_get_*) */

#include "xtrack/headers/track.h"
#include "xtrack/headers/particle_states.h"             /* XT_LOST_ON_APERTURE (aperture loss) */
#include "xtrack/particles/local_particle_custom_api.h" /* update_delta/ptau/pzeta, add_to_energy (XT_NUM) */
#include "generated/xt_dispatch.inc"                    /* physics includes + XT_TYPEID + xt_bridge_dispatch() */

#if defined(XT_FLAVOR_TPSA)
  #define XT_F(base) base##_tpsa
  /* Route A knob kernel (xt_bridge_knob.cpp, linked into the tpsa .so): the
   * mad::tpsa-strength variant of a knob-using element, dispatched by real typeid.
   * Declared here so this TU (and cffi's wrapper for set_table) can call them, and
   * defined in the linked knob object. */
  extern "C" void xt_knob_dispatch(int64_t type_id, void* el, void* part);
  extern "C" void xt_knob_set_table(const void** addrs, const void** tpsas,
                                    const void* proto, int64_t n);
#else
  #define XT_F(base) base##_num
#endif

/* Point a LocalParticle at the bridge struct: coord pointers + ref/int variables
 * via the generated XtBridgeParticle_get_* accessors (coords are tpsa_t* / double* addresses
 * stored as UInt64. refs/ints stay reachable through the XtBridgeParticle `bp`). */
static inline void lp_bind(LocalParticle* part, XtBridgeParticle p){
    part->x = (XT_COORD*)(uintptr_t)XtBridgeParticle_get_x(p);
    part->px = (XT_COORD*)(uintptr_t)XtBridgeParticle_get_px(p);
    part->y = (XT_COORD*)(uintptr_t)XtBridgeParticle_get_y(p);
    part->py = (XT_COORD*)(uintptr_t)XtBridgeParticle_get_py(p);
    part->zeta = (XT_COORD*)(uintptr_t)XtBridgeParticle_get_zeta(p);
    part->delta = (XT_COORD*)(uintptr_t)XtBridgeParticle_get_delta(p);
    part->bp = p;                                               /* opaque; refs/ints read via accessors */
    part->state = XtBridgeParticle_getp_state(p);               /* &state field: check_is_active reads it */
    part->line_length = XtBridgeParticle_get_line_length(p);    /* RF revolution time */
    part->ipart = 0; part->endpart = 1;
    part->_num_active_particles = 1; part->_num_lost_particles = 0;
    part->track_flags = XtBridgeParticle_get_track_flags(p);
}

/* Allocate the C-owned derived coords (rvv,rpp,ptau,s), point the LocalParticle at
 * them. s, ax and ay start at 0. They live in the caller's frame and outlive the tracking.
 * Caller must call LocalParticle_update_delta to fill rvv, rpp, ptau. */
#if defined(XT_FLAVOR_TPSA)
  /* each derived coord tpsa borrows x's descriptor (any live coord carries the right one). */
  #define XT_DECL_DERIVED(part) \
      mad::tpsa _rvv{mad::tpsa_ref((part).x)}, _rpp{mad::tpsa_ref((part).x)}, \
                _ptau{mad::tpsa_ref((part).x)}, _s{mad::tpsa_ref((part).x)}, \
                _ax{mad::tpsa_ref((part).x)}, _ay{mad::tpsa_ref((part).x)}; \
      _s = 0.0; _ax = 0.0; _ay = 0.0; \
      (part).rvv = _rvv.ptr(); (part).rpp = _rpp.ptr(); \
      (part).ptau = _ptau.ptr(); (part).s = _s.ptr(); \
      (part).ax = _ax.ptr(); (part).ay = _ay.ptr();
#else
  #define XT_DECL_DERIVED(part) \
      double _rvv, _rpp, _ptau, _s = 0.0, _ax = 0.0, _ay = 0.0; \
      (part).rvv = &_rvv; (part).rpp = &_rpp; (part).ptau = &_ptau; (part).s = &_s; \
      (part).ax = &_ax; (part).ay = &_ay;
#endif

extern "C"
void XT_F(xt_bridge_track_element)(int64_t type_id, void* el, void* p_){
    XtBridgeParticle p = (XtBridgeParticle) p_;   /* opaque buffer pointer from Python */
    LocalParticle part;
    lp_bind(&part, p);
    XT_DECL_DERIVED(part);
    LocalParticle_update_delta(&part, LocalParticle_get_delta(&part));  /* refresh rvv,rpp,ptau */
    xt_bridge_dispatch(type_id, el, &part);
}

/* Record the full map into slot `slot` of an XtBridgeTpsaMonitor: mad_tpsa_copy of every
 * coordinate series into that slot's preallocated tpsa (Python preallocates and hands us
 * the tpsa_t* addresses). A ParticlesMonitor cannot do this:
 * its ParticlesData store is six doubles per slot, i.e. the const part only.
 * The _num flavor has no map to record -> no-op (never reached, flag 3 is TPSA-only). */
#if defined(XT_FLAVOR_TPSA)
static inline void xt_tpsa_monitor_record(XtBridgeTpsaMonitor mon, LocalParticle* part,
                                          int64_t slot){
    const int64_t nc = XtBridgeTpsaMonitor_get_n_coords(mon);
    if (slot < 0 || slot >= XtBridgeTpsaMonitor_get_n_slots(mon)) return;
    XT_COORD* const src[6] = {part->x, part->px, part->y, part->py,
                              part->zeta, part->delta};
    for (int64_t j = 0; j < nc; ++j){
        XT_COORD* dst = (XT_COORD*)(uintptr_t)
            XtBridgeTpsaMonitor_get_coords(mon, slot*nc + j);
        mad_tpsa_copy(src[j], dst);   /* whole polynomial, all orders */
    }
}
#else
static inline void xt_tpsa_monitor_record(XtBridgeTpsaMonitor, LocalParticle*, int64_t){}
#endif

/* Track [ele_start, ele_start + num_elements] once, with optional monitoring.
 * flag_monitor follows the same convention as the native track_line kernel:
 *   flag_monitor == 0  no monitor (`mon_` may be NULL)
 *                == 1  record once, before the range (at_turn slot, at_turn == 0 here)
 *                == 2  element-by-element: record before every element + once at the end,
 *                      the monitor being in ebe_mode, whose store slot is at_element.
 *                == 3  element-by-element into an XtBridgeTpsaMonitor: the FULL map per
 *                      slot, not just its const part (TPSA flavor only).
 * `mon_` is a ParticlesMonitorData for flags 1-2 and an XtBridgeTpsaMonitor for flag 3.
 * `observe` (flag 3 only, may be NULL) selects which positions to record: a length
 * num_elements+1 array over the range (index k = before element ele_start+k, index
 * num_elements = the end). The full map is recorded at positions where observe[k] != 0,
 * into consecutive monitor slots. NULL records every position (EBE), so the slot counter
 * then coincides with the position index.
 * at_element is maintained only in EBE mode: it counts elements tracked (0-based from
 * ele_start, like in xtrack's increment_at_element from a frsh particle). On loss, it
 * is the absolute line index, which Python maps back to the name.
 * knob_dispatch:
 *   NULL -> pure-double path (bit-identical, _num always passes NULL).
 *   Non-NULL -> knob_dispatch[ii - ele_start] is 0 for a normal element or
 * real_typeid + 1 for an instance affected by knobs, which routes to the tpsa-strength kernel
 * (xt_knob_dispatch). The knob table (xt_knob_set_table) must be pushed from Python first. */
extern "C"
void XT_F(xt_bridge_track_line)(void* ref_, int64_t ele_start, int64_t num_elements,
                               void* p_, void* mon_, int64_t flag_monitor,
                               const int64_t* observe, const int64_t* knob_dispatch){
    XtBridgeParticle p = (XtBridgeParticle) p_;
    ElementRefData ref = (ElementRefData) ref_;
    ParticlesMonitorData mon = (ParticlesMonitorData) mon_;      /* flags 1-2 */
    XtBridgeTpsaMonitor tpsa_mon = (XtBridgeTpsaMonitor) mon_;   /* flag 3 */
    LocalParticle part;
    lp_bind(&part, p);
    XT_DECL_DERIVED(part);   /* locals reused across the loop; s accumulates */
    /* Refresh derived coords (rvv,rpp,ptau) from delta once before the loop. Native
     * track_line never re-derives mid-line: the physics carries them as state and keeps
     * them consistent. Per-element re-derivation would break the _num<->native
     * identity for any ptau-writing element. */
    LocalParticle_update_delta(&part, LocalParticle_get_delta(&part));
    if (flag_monitor == 1){
        ParticlesMonitor_track_local_particle(mon, &part);
    }
    int64_t slot = 0;   /* flag 3: next TpsaMonitor slot; advances only on a recorded position */
    for (int64_t ii = ele_start; ii < ele_start + num_elements; ++ii){
        if (flag_monitor == 2){
            XtBridgeParticle_set_at_element(p, ii - ele_start);
            ParticlesMonitor_track_local_particle(mon, &part);
        }
        else if (flag_monitor == 3 && (observe == nullptr || observe[ii - ele_start])){
            xt_tpsa_monitor_record(tpsa_mon, &part, slot++);
        }
        void* el = ElementRefData_member_elements(ref, ii);
        int64_t type_id = ElementRefData_typeid_elements(ref, ii);
#if defined(XT_FLAVOR_TPSA)
        if (knob_dispatch && knob_dispatch[ii - ele_start]){
            xt_knob_dispatch(type_id, el, &part);   /* tpsa-strength kernel (knobbed) */
        } else
#endif
        xt_bridge_dispatch(type_id, el, &part);
        if (XtBridgeParticle_get_state(p) <= 0){  /* loss: a map past its loss point is meaningless */
            XtBridgeParticle_set_at_element(p, ii);
            return;
        }
    }
    if (flag_monitor == 2){   /* final slot: the map at the end of the range */
        XtBridgeParticle_set_at_element(p, num_elements);
        ParticlesMonitor_track_local_particle(mon, &part);
    }
    else if (flag_monitor == 3 && (observe == nullptr || observe[num_elements])){
        xt_tpsa_monitor_record(tpsa_mon, &part, slot++);
    }
}
