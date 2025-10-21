// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2025.                 //
// ######################################### //

#ifndef XTRACK_TPSAMAP_H
#define XTRACK_TPSAMAP_H

#include <headers/track.h>

GPUFUN
double taylor_expansion_single(ArrNRefArrNInt8 monomials, ArrNFloat64 coefficients, double const* base_coordinates,
                               ArrNFloat64 machine_vals, ArrNFloat64 map_machine_vals, LocalParticle *part);

GPUFUN
void TPSAMap_track_local_particle(TPSAMapData el, LocalParticle* part0){

    double const length = TPSAMapData_get_length(el);
    double const* base_coordinates = TPSAMapData_getp1_base_coordinates(el, 0);
    ArrNFloat64 machine_vals = TPSAMapData_getp_map_machine_vals(el);
    ArrNFloat64 map_machine_vals = TPSAMapData_getp_map_machine_vals(el);

    START_PER_PARTICLE_BLOCK(part0, part);
        LocalParticle_add_to_s(part, length);
        double x = taylor_expansion_single(TPSAMapData_getp_x_monomials(el), TPSAMapData_getp_x_coefficients(el), base_coordinates, machine_vals, map_machine_vals, part);
        double px = taylor_expansion_single(TPSAMapData_getp_px_monomials(el), TPSAMapData_getp_px_coefficients(el), base_coordinates, machine_vals, map_machine_vals, part);
        double y = taylor_expansion_single(TPSAMapData_getp_y_monomials(el), TPSAMapData_getp_y_coefficients(el), base_coordinates, machine_vals, map_machine_vals, part);
        double py = taylor_expansion_single(TPSAMapData_getp_py_monomials(el), TPSAMapData_getp_py_coefficients(el), base_coordinates, machine_vals, map_machine_vals, part);
        double zeta = taylor_expansion_single(TPSAMapData_getp_zeta_monomials(el), TPSAMapData_getp_zeta_coefficients(el), base_coordinates, machine_vals, map_machine_vals, part);
        double delta = taylor_expansion_single(TPSAMapData_getp_delta_monomials(el), TPSAMapData_getp_delta_coefficients(el), base_coordinates, machine_vals, map_machine_vals, part);

        LocalParticle_set_x(part, x);
        LocalParticle_set_px(part, px);
        LocalParticle_set_y(part, y);
        LocalParticle_set_py(part, py);
        LocalParticle_set_zeta(part, zeta);
        LocalParticle_set_delta(part, delta);

    END_PER_PARTICLE_BLOCK;
}

GPUFUN
double taylor_expansion_single(ArrNRefArrNInt8 monomials, ArrNFloat64 coefficients, double const* base_coordinates,
                               ArrNFloat64 machine_vals, ArrNFloat64 map_machine_vals, LocalParticle *part) {
    double total = 0.0;
    double x = LocalParticle_get_x(part);
    double px = LocalParticle_get_px(part);
    double y = LocalParticle_get_y(part);
    double py = LocalParticle_get_py(part);
    double zeta = LocalParticle_get_zeta(part);
    double delta = LocalParticle_get_delta(part);
    double actual_coordinates[6] = {x, px, y, py, zeta, delta};

    for (int i = 0; i < ArrNRefArrNInt8_len(monomials); i++) {
        // Calculate the order
        int order = 0;
        for (int j = 0; j < ArrNRefArrNInt8_len1(monomials, i); j++) {
            order += ArrNRefArrNInt8_get(monomials, i, j);
        }

        // Calculate factorial term
        int factorial_term = 1;
        for (int k = 1; k < order + 1; k++) {
            factorial_term *= k;
        }

        double term = (1.0 / factorial_term) * ArrNFloat64_get(coefficients, i);
        for (int j = 0; j < 6; j++) {
            if (ArrNRefArrNInt8_get(monomials, i, j) != 0) {
                // power = ArrNRefArrNInt8_get(monomials, i, j); Potentially faster
                term *= pow(actual_coordinates[j] - base_coordinates[j], ArrNRefArrNInt8_get(monomials, i, j));
            }
        }
        for (int j = 6; j < ArrNRefArrNInt8_len1(monomials, i); j++) {
            if (ArrNRefArrNInt8_get(monomials, i, j) != 0) {
                int machine_index = j - 6;
                term *= pow(ArrNFloat64_get(machine_vals, machine_index) - ArrNFloat64_get(map_machine_vals, machine_index), ArrNRefArrNInt8_get(monomials, i, j));
            }
        }
        total += term;
    }

    return total;
}

#endif
