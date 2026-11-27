"""KnobParameters: hold line variables as GTPSA parameters, and restore them.

Assigning a parameter to a line variable makes xdeps re-evaluate the dependent
expressions in the TPSA algebra, so the driven element strengths hold Taylor series in
the knob and a map tracked on the same descriptor carries d(coord)/d(knob).
"""

from __future__ import annotations

from typing import Any

from madng_tpsa import Descriptor, Tpsa
from xdeps.refs import AttrRef


def _scalar_value(value: Any) -> float:
    if isinstance(value, Tpsa):
        return float(value.const_part)
    return float(value)


def _tpsa_enabled(element: Any) -> bool:
    return bool(getattr(getattr(element, "_xobject", None), "_tpsa_enabled", 0))


class KnobParameters:
    """Manage line knobs as TPSA parameters, so tracked maps carry knob derivatives.

    The knobs are the line variables named in ``descriptor.param_names``.
    ``apply()`` puts the parameters in, ``apply(values)`` moves them to new knob
    values, ``apply_doubles(values)`` and ``teardown()`` put plain doubles back in the
    elements the knobs reach.

    While the parameters are in, the knobs must be moved with ``apply``: assigning a
    float to ``line.vars`` re-propagates constants and drops the knob dependence.

    With ``affine=True`` the driven fields must be affine in the knobs, and their
    expressions, including the variables in their coefficients, must not change while
    the object lives. One xdeps pass builds the series, later switches only move the
    constant parts and the knob variables stay floats. A series replaced while the
    parameters are in raises at the next ``apply`` or ``apply_doubles``. This flag is
    only to increase performance, and is thus optional.
    """

    def __init__(self, line: Any, descriptor: Descriptor, affine: bool = False) -> None:
        self.line = line
        self.names = descriptor.param_names
        self.descriptor = descriptor
        self.affine = affine
        for name in self.names:
            if name not in line.vars:
                raise KeyError(f"knob {name!r} is not a line variable")
        self._applied = False
        self._reached = self._reached_elements()
        # Teardown disables the reached elements, which would drop a series set elsewhere.
        enabled = [name for name in self._reached if _tpsa_enabled(line.element_dict[name])]
        if enabled:
            raise ValueError(f"Elements reached by the knobs are already TPSA-enabled: {enabled}")
        self._kept_series = None

    def _reached_elements(self) -> list[str]:
        element_refs = self.line.element_refs
        deps = self.line._xdeps_manager.find_deps([self.line.vars[name] for name in self.names])
        return sorted({ref._owner._key for ref in deps
                       if isinstance(ref, AttrRef) and ref._owner._owner is element_refs})

    def __len__(self) -> int:
        return len(self.names)

    @property
    def applied(self) -> bool:
        """Do the reached elements hold the knob parameters?"""
        return self._applied

    def apply(self, values: list[float] | None = None) -> None:
        """Assign parameter ``k+1`` to knob ``k``, seeded at its current value."""
        if values is None:
            values = [_scalar_value(self.line[name]) for name in self.names]
        if len(values) != len(self.names):
            raise ValueError("Exactly one value per knob is required")
        if not self.affine:
            self._apply_parameters(values)
            return
        elements = self.line.element_dict
        if self._kept_series is None:
            self._apply_parameters(values)
            self._kept_series = {
                name: {field: getattr(elements[name], field)
                       for field in elements[name]._float_or_tpsa_fields}
                for name in self._reached}
            # Floats in while enabled: a field driven by several knobs would get a Tpsa
            # in between, which enables a disabled element again.
            for name, value in zip(self.names, values, strict=True):
                self.line.vars[name] = _scalar_value(value)
            for name, kept in self._kept_series.items():
                for field, series in kept.items():
                    setattr(elements[name], field, series)
            self._keep_and_disable_series()
        elif self._applied:
            self._keep_and_disable_series()
        for name, value in zip(self.names, values, strict=True):
            self.line.vars[name] = _scalar_value(value)
        for name in self._reached:
            elements[name].enable_tpsa(self.descriptor, reuse_cached=True)
        self._applied = True

    def _apply_parameters(self, values: list[float]) -> None:
        old_values = [_scalar_value(self.line[name]) for name in self.names]
        try:
            for index, (name, value) in enumerate(zip(self.names, values, strict=True), start=1):
                self.line.vars[name] = self.descriptor.param(index, _scalar_value(value))
        except Exception:
            self._drop_parameters(old_values)
            raise
        self._applied = True

    def _keep_and_disable_series(self) -> None:
        elements = self.line.element_dict
        replaced = [(name, field) for name, kept in self._kept_series.items()
                    for field, series in kept.items() if getattr(elements[name], field) is not series]
        if replaced:
            raise RuntimeError("Series replaced while the parameters were in, e.g. by "
                               f"assigning a knob outside apply: {replaced}")
        for name in self._reached:
            elements[name].disable_tpsa(keep_series=True)
        self._applied = False

    def teardown(self) -> None:
        """Plain doubles back in the variables and in every element they reached."""
        if self.affine:
            for name in self._reached:
                self.line.element_dict[name].disable_tpsa()
            self._kept_series = None
            self._applied = False
            return
        if not self._applied:
            return
        self._drop_parameters([_scalar_value(self.line[name]) for name in self.names])

    def apply_doubles(self, values: list[float]) -> None:
        """Set the knobs to floats and drop TPSA storage from the elements they drive."""
        if len(values) != len(self.names):
            raise ValueError("Exactly one value per knob is required")
        if not self.affine:
            self._drop_parameters(values)
            return
        if self._applied:
            self._keep_and_disable_series()
        for name, value in zip(self.names, values, strict=True):
            self.line.vars[name] = _scalar_value(value)

    def _drop_parameters(self, values: list[float]) -> None:
        for name, value in zip(self.names, values, strict=True):
            self.line.vars[name] = _scalar_value(value)
        # xdeps re-propagates the new floats, but an enabled element stores them as
        # constant series, so the switch back to doubles is explicit by disabling tpsa
        # for the reached elements.
        elements = self.line.element_dict
        for name in self._reached:
            element = elements[name]
            if (_tpsa_enabled(element)
                    and all(getattr(element, field).is_constant()
                            for field in element._float_or_tpsa_fields)):
                element.disable_tpsa()
        self._applied = False

    def driven_elements(self) -> list[tuple[str, str]]:
        """``(element, attribute)`` of every field currently holding a series."""
        elements = self.line.element_dict
        return sorted((name, field) for name in self._reached if _tpsa_enabled(elements[name])
                      for field in elements[name]._float_or_tpsa_fields)

    def strength_jacobian(self) -> dict[tuple[str, str], list[float]]:
        """``d strength / d knob`` per driven field.

        Distinct from ``ParticlesTpsa.param_jacobian`` (``d coord / d knob``): this is the
        upstream element-strength sensitivity that seeds the map.
        """
        jacobian = {}
        for element_name, attr in self.driven_elements():
            value = getattr(self.line.element_dict[element_name], attr)
            if isinstance(value, Tpsa):
                jacobian[(element_name, attr)] = value.param_grad()
        return jacobian

    def __repr__(self) -> str:
        return (f"KnobParameters(names={self.names}, applied={self._applied}, "
                f"affine={self.affine})")
