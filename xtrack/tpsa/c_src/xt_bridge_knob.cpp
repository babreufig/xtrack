/* Route A knob object (second translation unit): the mad::tpsa-STRENGTH variant of the
 * knob-using magnet kernels, compiled as its own translation unit (-DXT_FLAVOR_TPSA -DXT_KNOBS)
 * and linked into the tpsa bridge .so. Namespace isolation of two strength variants in one TU
 * fails (argument dependent lookup on the global LocalParticle* type -> ambiguous calls), so the
 * knob kernel is a separate translation unit. It shares LocalParticle / <El>Data / mad::tpsa layouts
 * with the main bridge (same generated headers, same libgtpsa_core.so), so the main
 * bridge hands a LocalParticle* / ElementData pointer straight across the .o boundary.
 *
 * Exports (extern "C", called from the main bridge's track_line in Python):
 *   xt_knob_dispatch(real_typeid, el, part): run one knob-using element (tpsa strengths)
 *   xt_knob_set_table(addrs, tpsas, proto, n): push the knob table before every track
 * The double kernels + the element loop stay in xt_bridge.cpp (unchanged). */

/* This TU selects its build flavor itself. It is fed to cffi's set_source(sources=...) and
 * so shares the main bridge's compile arguments, which don't carry the per-file -DXT_KNOBS that
 * makes strengths mad::tpsa. The main bridge TU never sees XT_KNOBS (separate TU) and stays double. */
#ifndef XT_FLAVOR_TPSA
#define XT_FLAVOR_TPSA 1
#endif
#ifndef XT_KNOBS
#define XT_KNOBS 1
#endif
#ifndef XTRACK_MULTIPOLE_NO_SYNRAD
#define XTRACK_MULTIPOLE_NO_SYNRAD 1
#endif

/* Standard headers mad_tpsa.hpp relies on but does not include itself. The main bridge
 * gets these transitively from cffi's <Python.h> preamble, but this TU is compiled
 * standalone (its own .o, linked in), so it must pull them explicitly. */
#include <cstddef>
#include <cstdint>
#include <string>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <cmath>

#include "xt_local_particle.hpp"                        /* XT_STRENGTH = mad::tpsa (XT_KNOBS) + LocalParticle */

#define restrict __restrict
#include "generated/xt_element_capi.h"                  /* <El>Data + accessors (layout shared w/ main bridge) */
#include "generated/xt_local_particle_gen.hpp"          /* field accessors */

#include "xt_knob.hpp"                                   /* address table + xt_knob() + xt_cur_addr slots */

#include "xtrack/headers/track.h"
#include "xtrack/headers/particle_states.h"
#include "xtrack/particles/local_particle_custom_api.h"
#include "generated/xt_knob_dispatch.inc"               /* knobbable includes + xt_knob_dispatch() */
