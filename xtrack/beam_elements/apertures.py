import numpy as np

import xobjects as xo

from ..base_element import BeamElement
from ..general import _pkg_root


class LimitRect(BeamElement):
    _xofields = {
        'min_x': xo.Float64,
        'max_x': xo.Float64,
        'min_y': xo.Float64,
        'max_y': xo.Float64,
        }

    def get_backtrack_element(self, _context=None, _buffer=None, _offset=None):
        return self.__class__(
                    min_x=self.min_x,
                    max_x=self.max_x,
                    min_y=self.min_y,
                    max_y=self.max_y,
                    _context=_context, _buffer=_buffer, _offset=_offset)

LimitRect.XoStruct.extra_sources = [
        _pkg_root.joinpath('beam_elements/apertures_src/limitrect.h')]


class LimitEllipse(BeamElement):
    _xofields = {
            'a_squ': xo.Float64,
            'b_squ': xo.Float64,
            'a_b_squ': xo.Float64,
            }

    def __init__(self, a_squ=None, b_squ=None, **kwargs):
        if a_squ is None and "a" in kwargs:
            a = kwargs.get("a")
            if a is not None and a > 0.0:
                a_squ = a * a
        if a_squ is None:
            a_squ = 1.0

        if b_squ is None and "b" in kwargs:
            b = kwargs.get("b")
            if b is not None and b > 0.0:
                b_squ = b * b
        if b_squ is None:
            b_squ = 1.0

        if a_squ > 0.0 and b_squ > 0.0:
            a_b_squ = a_squ * b_squ
            kwargs['a_squ'] = a_squ
            kwargs['b_squ'] = b_squ
            kwargs['a_b_squ'] = a_squ * b_squ
            super().__init__(**kwargs)
        else:
            raise ValueError("a_squ and b_squ have to be positive definite")

    def set_half_axes(self, a, b):
        return self.set_half_axes_squ(a * a, b * b)

    def set_half_axes_squ(self, a_squ, b_squ):
        self.a_squ = a_squ
        self.b_squ = b_squ
        self.a_b_squ = a_squ * b_squ
        return self

    def get_backtrack_element(self, _context=None, _buffer=None, _offset=None):
        return self.__class__(
                    a_squ=self.a_squ,
                    b_squ=self.b_squ,
                    a_b_squ=self.a_b_squ,
                    _context=_context, _buffer=_buffer, _offset=_offset)

LimitEllipse.XoStruct.extra_sources = [
        _pkg_root.joinpath('beam_elements/apertures_src/limitellipse.h')]


class LimitPolygon(BeamElement):
    _xofields = {
        'x_vertices': xo.Float64[:],
        'y_vertices': xo.Float64[:],
        'x_normal': xo.Float64[:],
        'y_normal': xo.Float64[:],
        'resc_fac': xo.Float64
        }

    def __init__(self, x_vertices, y_vertices, **kwargs):

        assert len(x_vertices) == len(y_vertices)

        super().__init__(
                x_vertices=x_vertices,
                y_vertices=y_vertices,
                x_normal = len(x_vertices),
                y_normal = len(x_vertices),
                resc_fac = 1.,
                **kwargs)

        lengths = np.sqrt(np.diff(self.x_closed)**2
                        + np.diff(self.y_closed)**2)

        assert np.all(lengths>0)


        if self.area < 0:
            raise ValueError(
                    "The area of the polygon is negative!\n"
                    "Vertices must be provided with counter-clockwise order!")

        Nx = -np.diff(self.y_closed)
        Ny = np.diff(self.x_closed)

        norm_N = np.sqrt(Nx**2 + Ny**2)
        Nx = Nx / norm_N
        Ny = Ny / norm_N

        ctx = self._buffer.context
        self.x_normal = ctx.nparray_to_context_array(Nx)
        self.y_normal = ctx.nparray_to_context_array(Ny)


    @property
    def x_closed(self):
        ctx = self._buffer.context
        xx = ctx.nparray_from_context_array(self.x_vertices)
        return np.concatenate([xx, np.array([xx[0]])])

    @property
    def y_closed(self):
        ctx = self._buffer.context
        yy = ctx.nparray_from_context_array(self.y_vertices)
        return np.concatenate([yy, np.array([yy[0]])])

    def impact_point_and_normal(self, x_in, y_in, z_in,
                                x_out, y_out, z_out):

        ctx = self._buffer.context

        if 'LimitPolygon_impact_point_and_normal' not in ctx.kernels.keys():
            ctx.add_kernels(
                sources = (['#define NO_LIMITPOLYGON_TRACK_LOCAL_PARTICLE']
                             + self.XoStruct.extra_sources),
                kernels =  self.XoStruct.custom_kernels)

        x_inters = ctx.zeros(shape=x_in.shape, dtype=np.float64)
        y_inters = ctx.zeros(shape=x_in.shape, dtype=np.float64)
        z_inters = ctx.zeros(shape=x_in.shape, dtype=np.float64)
        Nx_inters = ctx.zeros(shape=x_in.shape, dtype=np.float64)
        Ny_inters = ctx.zeros(shape=x_in.shape, dtype=np.float64)
        i_found = ctx.zeros(shape=x_in.shape, dtype=np.int64)

        ctx.kernels.LimitPolygon_impact_point_and_normal(el=self,
                x_in=x_in, y_in=y_in, z_in=z_in,
                x_out=x_out, y_out=y_out, z_out=z_out,
                n_impacts=len(x_in), x_inters=x_inters,
                y_inters=y_inters, z_inters=z_inters,
                Nx_inters=Nx_inters, Ny_inters=Ny_inters,
                i_found=i_found)

        assert np.all(i_found>=0)

        return x_inters, y_inters, z_inters, Nx_inters, Ny_inters, i_found

    @property
    def area(self):
        return -0.5 * np.sum((self.y_closed[1:] + self.y_closed[:-1])
                                * (self.x_closed[1:] - self.x_closed[:-1]))
    @property
    def centroid(self):
        x = self.x_vertices
        y = self.x_vertices
        cx = 1/(6*self.area)*np.sum((x[:-1]+x[1:])*(x[:-1]*y[1:]-x[1:]*y[:-1]))
        cy = 1/(6*self.area)*np.sum((y[:-1]+y[1:])*(y[:-1]*x[1:]-y[1:]*x[:-1]))
        return (cx,cy)

LimitPolygon.XoStruct.extra_sources = [
        _pkg_root.joinpath('beam_elements/apertures_src/limitpolygon.h')]

LimitPolygon.XoStruct.custom_kernels = {
    'LimitPolygon_impact_point_and_normal': xo.Kernel(
        args = [xo.Arg(LimitPolygon.XoStruct, name='el'),
                xo.Arg(xo.Float64, pointer=True, name='x_in'),
                xo.Arg(xo.Float64, pointer=True, name='y_in'),
                xo.Arg(xo.Float64, pointer=True, name='z_in'),
                xo.Arg(xo.Float64, pointer=True, name='x_out'),
                xo.Arg(xo.Float64, pointer=True, name='y_out'),
                xo.Arg(xo.Float64, pointer=True, name='z_out'),
                xo.Arg(xo.Int64,   pointer=False, name='n_impacts'),
                xo.Arg(xo.Float64, pointer=True, name='x_inters'),
                xo.Arg(xo.Float64, pointer=True, name='y_inters'),
                xo.Arg(xo.Float64, pointer=True, name='z_inters'),
                xo.Arg(xo.Float64, pointer=True, name='Nx_inters'),
                xo.Arg(xo.Float64, pointer=True, name='Ny_inters'),
                xo.Arg(xo.Int64,   pointer=True, name='i_found')],
        n_threads='n_impacts')}
