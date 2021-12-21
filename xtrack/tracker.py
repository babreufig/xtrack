from pathlib import Path
import numpy as np
from scipy.optimize import fsolve

from .general import _pkg_root
from .line_frozen import LineFrozen
from .base_element import _handle_per_particle_blocks
from .twiss_from_tracker import twiss_from_tracker

import xobjects as xo
import xpart as xp

from .beam_elements import Drift
from .line import Line

def _check_is_collective(ele):
    iscoll = not hasattr(ele, 'iscollective') or ele.iscollective
    return iscoll

class Tracker:

    def __init__(
        self,
        _context=None,
        _buffer=None,
        _offset=None,
        line=None,
        sequence=None,
        track_kernel=None,
        element_classes=None,
        particles_class=None,
        skip_end_turn_actions=False,
        particles_monitor_class=None,
        global_xy_limit=1.0,
        local_particle_src=None,
        save_source_as=None,
    ):

        if sequence is not None:
            raise ValueError(
                    "`Tracker(... sequence=... ) is deprecated use `line=`)")

        # Check if there are collective elements
        self.iscollective = False
        for ee in line.elements:
            if _check_is_collective(ee):
                self.iscollective = True
                break

        if self.iscollective:
            self._init_track_with_collective(
                _context=_context,
                _buffer=_buffer,
                _offset=_offset,
                line=line,
                track_kernel=track_kernel,
                element_classes=element_classes,
                particles_class=particles_class,
                skip_end_turn_actions=skip_end_turn_actions,
                particles_monitor_class=particles_monitor_class,
                global_xy_limit=global_xy_limit,
                local_particle_src=local_particle_src,
                save_source_as=save_source_as)
        else:
            self._init_track_no_collective(
                _context=_context,
                _buffer=_buffer,
                _offset=_offset,
                line=line,
                track_kernel=track_kernel,
                element_classes=element_classes,
                particles_class=particles_class,
                skip_end_turn_actions=skip_end_turn_actions,
                particles_monitor_class=particles_monitor_class,
                global_xy_limit=global_xy_limit,
                local_particle_src=local_particle_src,
                save_source_as=save_source_as)

    def _init_track_with_collective(
        self,
        _context=None,
        _buffer=None,
        _offset=None,
        line=None,
        track_kernel=None,
        element_classes=None,
        particles_class=None,
        skip_end_turn_actions=False,
        particles_monitor_class=None,
        global_xy_limit=1.0,
        local_particle_src=None,
        save_source_as=None,
    ):

        assert _offset is None
        assert track_kernel is None
        assert element_classes is None

        self.skip_end_turn_actions = skip_end_turn_actions
        self.particles_class = particles_class
        self.global_xy_limit = global_xy_limit
        self.local_particle_src = local_particle_src
        self.save_source_as = save_source_as

        if _buffer is None:
            if _context is None:
                _context = xo.context_default
            _buffer = _context.new_buffer()
        self._buffer = _buffer

        # Split the sequence
        parts = []
        this_part = Line(elements=[], element_names=[])
        for nn, ee in zip(line.element_names, line.elements):
            if not _check_is_collective(ee):
                this_part.append_element(ee, nn)
            else:
                if len(this_part.elements)>0:
                    this_part.iscollective=False
                    parts.append(this_part)
                parts.append(ee)
                this_part = Line(elements=[], element_names=[])
        if len(this_part.elements)>0:
            this_part.iscollective=False
            parts.append(this_part)

        # Transform non collective elements into xtrack elements 
        noncollective_xelements = []
        for ii, pp in enumerate(parts):
            if not _check_is_collective(pp):
                tempxtline = LineFrozen(_buffer=_buffer,
                                   line=pp)
                pp.elements = tempxtline.elements
                noncollective_xelements += pp.elements
            else:
                if hasattr(pp, 'isthick') and pp.isthick:
                    ldrift = pp.length
                else:
                    ldrift = 0.

                noncollective_xelements.append(
                    Drift(_buffer=_buffer, length=ldrift))

        # Build tracker for all non collective elements
        supertracker = Tracker(_buffer=_buffer,
                line=Line(elements=noncollective_xelements),
                    particles_class=particles_class,
                    particles_monitor_class=particles_monitor_class,
                    global_xy_limit=global_xy_limit,
                    local_particle_src=local_particle_src,
                    save_source_as=save_source_as
                    )

        # Build trackers for non collective parts
        for ii, pp in enumerate(parts):
            if not _check_is_collective(pp):
                parts[ii] = Tracker(_buffer=_buffer,
                                    line=pp,
                                    element_classes=supertracker.element_classes,
                                    track_kernel=supertracker.track_kernel,
                                    particles_class=particles_class,
                                    particles_monitor_class=particles_monitor_class,
                                    global_xy_limit=global_xy_limit,
                                    local_particle_src=local_particle_src,
                                    skip_end_turn_actions=True)

        # Make a "marker" element to increase at_element
        self._zerodrift = Drift(_context=_buffer.context, length=0)

        self.line = line
        self._supertracker = supertracker
        self._parts = parts
        self.track = self._track_with_collective
        self.particles_class = supertracker.particles_class
        self.particles_monitor_class = supertracker.particles_monitor_class


    def _init_track_no_collective(
        self,
        _context=None,
        _buffer=None,
        _offset=None,
        line=None,
        track_kernel=None,
        element_classes=None,
        particles_class=None,
        skip_end_turn_actions=False,
        particles_monitor_class=None,
        global_xy_limit=1.0,
        local_particle_src=None,
        save_source_as=None,
    ):
        if particles_class is None:
            particles_class = xp.Particles

        if particles_monitor_class is None:
            import xtrack as xt  # I have to do it like this
                                 # to avoid circular import #TODO to be solved
            particles_monitor_class = xt.ParticlesMonitor

        if local_particle_src is None:
            local_particle_src = xp.gen_local_particle_api()

        self.global_xy_limit = global_xy_limit

        frozenline = LineFrozen(_context=_context, _buffer=_buffer, _offset=_offset,
                    line=line)

        context = frozenline._buffer.context

        if track_kernel is None:
            # Kernel relies on element_classes ordering
            assert element_classes is None

        if element_classes is None:
            # Kernel relies on element_classes ordering
            assert track_kernel=='skip' or track_kernel is None
            element_classes = frozenline._ElementRefClass._reftypes + [
                particles_monitor_class.XoStruct,
            ]

        line._freeze()
        self.line = line
        self._line_frozen = frozenline
        ele_offsets = np.array([ee._offset for ee in frozenline.elements], dtype=np.int64)
        ele_typeids = np.array(
            [element_classes.index(ee._xobject.__class__) for ee in frozenline.elements],
            dtype=np.int64,
        )
        ele_offsets_dev = context.nparray_to_context_array(ele_offsets)
        ele_typeids_dev = context.nparray_to_context_array(ele_typeids)

        self.particles_class = particles_class
        self.particles_monitor_class = particles_monitor_class
        self.ele_offsets_dev = ele_offsets_dev
        self.ele_typeids_dev = ele_typeids_dev
        self.num_elements = len(frozenline.elements)
        self.global_xy_limit = global_xy_limit
        self.skip_end_turn_actions = skip_end_turn_actions
        self.local_particle_src = local_particle_src
        self.element_classes = element_classes
        self._buffer = frozenline._buffer

        if track_kernel == 'skip':
            self.track_kernel = None
        elif track_kernel is None:
            self._build_kernel(save_source_as)
        else:
            self.track_kernel = track_kernel

        self.track=self._track_no_collective

    def find_closed_orbit(self, particle_co_guess):

        particle_co_guess = particle_co_guess.copy(
                            _context=self._buffer.context)

        res = fsolve(lambda p: p - _one_turn_map(p, particle_co_guess, self),
              x0=np.array([particle_co_guess._xobject.x[0],
                           particle_co_guess._xobject.px[0],
                           particle_co_guess._xobject.y[0],
                           particle_co_guess._xobject.py[0],
                           particle_co_guess._xobject.zeta[0],
                           particle_co_guess._xobject.delta[0]]))

        particle_on_co = particle_co_guess.copy()
        particle_on_co.x = res[0]
        particle_on_co.px = res[1]
        particle_on_co.y = res[2]
        particle_on_co.py = res[3]
        particle_on_co.zeta = res[4]
        particle_on_co.delta = res[5]

        return particle_on_co

    def compute_one_turn_matrix_finite_differences(
            self, particle_on_co,
            dx=1e-7, dpx=1e-10, dy=1e-7, dpy=1e-10,
            dzeta=1e-6, ddelta=1e-7):

        particle_on_co = particle_on_co.copy(
                            _context=self._buffer.context)
        # Find R matrix
        p0 = np.array([
               particle_on_co._xobject.x[0],
               particle_on_co._xobject.px[0],
               particle_on_co._xobject.y[0],
               particle_on_co._xobject.py[0],
               particle_on_co._xobject.zeta[0],
               particle_on_co._xobject.delta[0]])
        II = np.eye(6)
        RR = np.zeros((6, 6), dtype=np.float64)
        for jj, dd in enumerate([dx, dpx, dy, dpy, dzeta, ddelta]):
            RR[:,jj]=(_one_turn_map(p0+II[jj]*dd, particle_on_co, self)-
                      _one_turn_map(p0-II[jj]*dd, particle_on_co, self))/(2*dd)

        return RR

    def twiss(self, particle_ref, r_sigma=0.01,
        nemitt_x=1e-6, nemitt_y=2.5e-6,
        n_theta=1000, delta_disp=1e-5, delta_chrom = 1e-4):

        return twiss_from_tracker(self, particle_ref, r_sigma=0.01,
            nemitt_x=1e-6, nemitt_y=2.5e-6,
            n_theta=1000, delta_disp=1e-5, delta_chrom = 1e-4)

    def get_backtracker(self, _context=None, _buffer=None):

        assert not self.iscollective

        if _buffer is None:
            if _context is None:
                _context = xo.context_default
            _buffer = _context.new_buffer()

        line = Line(elements=[], element_names=[])
        for nn, ee in zip(self.line.element_names[::-1],
                          self.line.elements[::-1]):
            line.append_element(
                    ee.get_backtrack_element(_buffer=_buffer), nn)

        return self.__class__(
                    _buffer=_buffer,
                    line=line,
                    track_kernel=self.track_kernel,
                    element_classes=self.element_classes,
                    particles_class=self.particles_class,
                    skip_end_turn_actions=self.skip_end_turn_actions,
                    particles_monitor_class=self.particles_monitor_class,
                    global_xy_limit=self.global_xy_limit,
                    local_particle_src=self.local_particle_src,
                )

    def _build_kernel(self, save_source_as):

        context = self._line_frozen._buffer.context

        sources = []
        kernels = {}
        cdefs = []

        sources.append(
                f"#define XTRACK_GLOBAL_POSLIMIT ({self.global_xy_limit})")
        sources.append(_pkg_root.joinpath("headers/constants.h"))


        # Local particles
        sources.append(self.local_particle_src)

        # Elements
        sources.append(_pkg_root.joinpath("tracker_src/tracker.h"))


        for ee in self.element_classes:
            for ss in ee.extra_sources:
                sources.append(ss)

        src_lines = []
        src_lines.append(
            r"""
            /*gpukern*/
            void track_line(
                /*gpuglmem*/ int8_t* buffer,
                /*gpuglmem*/ int64_t* ele_offsets,
                /*gpuglmem*/ int64_t* ele_typeids,
                             ParticlesData particles,
                             int num_turns,
                             int ele_start,
                             int num_ele_track,
                             int flag_end_turn_actions,
                             int flag_tbt_monitor,
                /*gpuglmem*/ int8_t* buffer_tbt_monitor,
                             int64_t offset_tbt_monitor){


            LocalParticle lpart;

            int64_t part_id = 0;                    //only_for_context cpu_serial cpu_openmp
            int64_t part_id = blockDim.x * blockIdx.x + threadIdx.x; //only_for_context cuda
            int64_t part_id = get_global_id(0);                    //only_for_context opencl


            /*gpuglmem*/ int8_t* tbt_mon_pointer =
                            buffer_tbt_monitor + offset_tbt_monitor;
            ParticlesMonitorData tbt_monitor =
                            (ParticlesMonitorData) tbt_mon_pointer;

            int64_t part_capacity = ParticlesData_get__capacity(particles);
            if (part_id<part_capacity){
            Particles_to_LocalParticle(particles, &lpart, part_id);

            int64_t isactive = check_is_active(&lpart);

            for (int64_t iturn=0; iturn<num_turns; iturn++){

                if (!isactive){
                    break;
                }

                if (flag_tbt_monitor){
                    ParticlesMonitor_track_local_particle(tbt_monitor, &lpart);
                }

                for (int64_t ee=ele_start; ee<ele_start+num_ele_track; ee++){

                        /*gpuglmem*/ int8_t* el = buffer + ele_offsets[ee];
                        int64_t ee_type = ele_typeids[ee];

                        switch(ee_type){
        """
        )

        for ii, cc in enumerate(self.element_classes):
            ccnn = cc.__name__.replace("Data", "")
            src_lines.append(
                f"""
                        case {ii}:
"""
            )
            if ccnn == "Drift":
                src_lines.append(
                    """
                            global_aperture_check(&lpart);

                            """
                )
            src_lines.append(
                f"""
                            {ccnn}_track_local_particle(({ccnn}Data) el, &lpart);
                            break;"""
            )

        src_lines.append(
            """
                        } //switch
                    isactive = check_is_active(&lpart);
                    if (!isactive){
                        break;
                    }
                    increment_at_element(&lpart);
                } // for elements
                if (flag_end_turn_actions>0){
                    if (isactive){
                        increment_at_turn(&lpart);
                    }
                }
            } // for turns
            }// if partid
        }//kernel
        """
        )

        source_track = "\n".join(src_lines)
        sources.append(source_track)

        kernel_descriptions = {
            "track_line": xo.Kernel(
                args=[
                    xo.Arg(xo.Int8, pointer=True, name="buffer"),
                    xo.Arg(xo.Int64, pointer=True, name="ele_offsets"),
                    xo.Arg(xo.Int64, pointer=True, name="ele_typeids"),
                    xo.Arg(self.particles_class.XoStruct, name="particles"),
                    xo.Arg(xo.Int32, name="num_turns"),
                    xo.Arg(xo.Int32, name="ele_start"),
                    xo.Arg(xo.Int32, name="num_ele_track"),
                    xo.Arg(xo.Int32, name="flag_end_turn_actions"),
                    xo.Arg(xo.Int32, name="flag_tbt_monitor"),
                    xo.Arg(xo.Int8, pointer=True, name="buffer_tbt_monitor"),
                    xo.Arg(xo.Int64, name="offset_tbt_monitor"),
                ],
            )
        }

        # Internal API can be exposed only on CPU
        if not isinstance(context, xo.ContextCpu):
            kernels = {}
        kernels.update(kernel_descriptions)

        sources = _handle_per_particle_blocks(sources)

        # Compile!
        context.add_kernels(
            sources,
            kernels,
            extra_classes=self.element_classes,
            save_source_as=save_source_as,
            specialize=True,
        )

        self.track_kernel = context.kernels.track_line


    def _track_with_collective(
        self,
        particles,
        ele_start=0,
        num_elements=None,
        num_turns=1,
        turn_by_turn_monitor=None,
    ):

        assert ele_start == 0
        assert num_elements is None

        (flag_tbt, monitor, buffer_monitor, offset_monitor
             ) = self._get_monitor(particles, turn_by_turn_monitor, num_turns)

        for tt in range(num_turns):
            if flag_tbt:
                monitor.track(particles)

            for pp in self._parts:
                pp.track(particles)
                if not isinstance(pp, Tracker):
                    self._zerodrift.track(particles, increment_at_element=True)


            # Increment at_turn and reset at_element
            # (use the supertracker to perform only end-turn actions)
            self._supertracker.track(particles,
                               ele_start=self._supertracker.num_elements,
                               num_elements=0)

        self.record_last_track = monitor


    def _track_no_collective(
        self,
        particles,
        ele_start=0,
        num_elements=None,
        num_turns=1,
        turn_by_turn_monitor=None,
    ):

        if num_elements is None:
            # get to the end of the turn
            num_elements = self.num_elements - ele_start

        assert num_elements + ele_start <= self.num_elements

        if self.skip_end_turn_actions:
            flag_end_turn_actions=False
        else:
            flag_end_turn_actions = (
                    num_elements + ele_start == self.num_elements)

        (flag_tbt, monitor, buffer_monitor, offset_monitor
             ) = self._get_monitor(particles, turn_by_turn_monitor, num_turns)

        self.track_kernel.description.n_threads = particles._capacity
        self.track_kernel(
            buffer=self._line_frozen._buffer.buffer,
            ele_offsets=self.ele_offsets_dev,
            ele_typeids=self.ele_typeids_dev,
            particles=particles._xobject,
            num_turns=num_turns,
            ele_start=ele_start,
            num_ele_track=num_elements,
            flag_end_turn_actions=flag_end_turn_actions,
            flag_tbt_monitor=flag_tbt,
            buffer_tbt_monitor=buffer_monitor,
            offset_tbt_monitor=offset_monitor,
        )

        self.record_last_track = monitor

    def _get_monitor(self, particles, turn_by_turn_monitor, num_turns):

        if turn_by_turn_monitor is None or turn_by_turn_monitor is False:
            flag_tbt = 0
            monitor = None
            buffer_monitor = particles._buffer.buffer  # I just need a valid buffer
            offset_monitor = 0
        elif turn_by_turn_monitor is True:
            flag_tbt = 1
            # TODO Assumes at_turn starts from zero, to be generalized
            monitor = self.particles_monitor_class(
                _context=particles._buffer.context,
                start_at_turn=0,
                stop_at_turn=num_turns,
                particle_id_range=particles.get_active_particle_id_range()
            )
            buffer_monitor = monitor._buffer.buffer
            offset_monitor = monitor._offset
        elif isinstance(turn_by_turn_monitor, self.particles_monitor_class):
            flag_tbt = 1
            monitor = turn_by_turn_monitor
            buffer_monitor = monitor._buffer.buffer
            offset_monitor = monitor._offset
        else:
            raise ValueError('Please provide a valid monitor object')

        return flag_tbt, monitor, buffer_monitor, offset_monitor


    def _slow_track_ebe(self,part):
        out=[]
        for ii in range(len(self.line.elements)):
            out.append(part.copy())
            self.track(part,ele_start=ii,num_elements=1)
        return out

def _one_turn_map(p, particle_ref, tracker):
    part = particle_ref.copy()
    part.x = p[0]
    part.px = p[1]
    part.y = p[2]
    part.py = p[3]
    part.zeta = p[4]
    part.delta = p[5]

    tracker.track(part)
    p_res = np.array([
           part._xobject.x[0],
           part._xobject.px[0],
           part._xobject.y[0],
           part._xobject.py[0],
           part._xobject.zeta[0],
           part._xobject.delta[0]])
    return p_res


