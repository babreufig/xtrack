// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_BEND_H
#define XTRACK_BEND_H


// model = 0: adaptive
// model = 1: full (for backward compatibility)
// model = 2: bend-kick-bend
// model = 3: rot-kick-rot
// model = 4: expanded

/*gpufun*/
void track_multipolar_kick_bend(
    LocalParticle* part, int64_t order, double inv_factorial_order,
    /*gpuglmem*/ const double* knl,
    /*gpuglmem*/ const double* ksl,
    double kick_weight, double k0, double k1, double h, double length){

    double const k1l = k1 * length * kick_weight;
    double const k0l = k0 * length * kick_weight;

    // dipole kick
    double dpx = -k0l;
    double dpy = 0;

    // quadrupole kick
    double const x = LocalParticle_get_x(part);
    double const y = LocalParticle_get_y(part);
    dpx += -k1l * x;
    dpy +=  k1l * y;

    // k0h correction can be computed from this term in the hamiltonian
    // H = 1/2 h k0 x^2
    // (see MAD 8 physics manual, eq. 5.15, and apply Hamilton's eq. dp/ds = -dH/dx)
    dpx += -k0l * h * x;

    // k1h correction can be computed from this term in the hamiltonian
    // H = 1/3 hk1 x^3 - 1/2 hk1 xy^2
    // (see MAD 8 physics manual, eq. 5.15, and apply Hamilton's eq. dp/ds = -dH/dx)

    dpx += h * k1l * (-x * x + 0.5 * y * y);
    dpy += h * k1l * x * y;
    LocalParticle_add_to_px(part, dpx);
    LocalParticle_add_to_py(part, dpy);

    multipolar_kick(part, order, inv_factorial_order, knl, ksl, kick_weight);
}

#define N_KICKS_YOSHIDA 7


/*gpufun*/
void Bend_track_local_particle(
        BendData el,
        LocalParticle* part0
) {
    double length = BendData_get_length(el);

    #ifdef XSUITE_BACKTRACK
        length = -length;
    #endif

    const double k0 = BendData_get_k0(el);
    const double k1 = BendData_get_k1(el);
    const double h = BendData_get_h(el);

    int64_t num_multipole_kicks = BendData_get_num_multipole_kicks(el);
    const int64_t order = BendData_get_order(el);
    const double inv_factorial_order = BendData_get_inv_factorial_order(el);

    const int64_t model = BendData_get_model(el);

    /*gpuglmem*/ const double *knl = BendData_getp1_knl(el, 0);
    /*gpuglmem*/ const double *ksl = BendData_getp1_ksl(el, 0);

    if (model==0 || model==1 || model==2 || model==3){

            int64_t num_slices;
            if (num_multipole_kicks == 0) { // num_multipole_kicks needs to be determined

                if (fabs(h) < 1e-8){
                    num_multipole_kicks = 0; // straight magnet
                }
                else{
                    double b_circum = 2 * 3.14159 / fabs(h);
                    num_multipole_kicks = fabs(length) / b_circum / 1e-4; // 0.1 mrad per kick (on average)
                }
            }

            if (num_multipole_kicks < 8) {
                num_slices = 1;
            }
            else{
                num_slices = num_multipole_kicks / N_KICKS_YOSHIDA + 1;
            }

            const double slice_length = length / (num_slices);
            const double kick_weight = 1. / num_slices;
            const double d_yoshida[] =
                         {0x1.91abc4988937bp-2, 0x1.052468fb75c74p-1,
                         -0x1.e25bd194051b9p-2, 0x1.199cec1241558p-4 };
                        //  {1/8.0, 1/8.0, 1/8.0, 1/8.0}; // Uniform, for debugging
            const double k_yoshida[] =
                         {0x1.91abc4988937bp-1, 0x1.e2743579895b4p-3,
                         -0x1.2d7c6f7933b93p+0, 0x1.50b00cfb7be3ep+0 };
                        //  {1/7.0, 1/7.0, 1/7.0, 1/7.0}; // Uniform, for debugging

            // printf("num_slices = %ld\n", num_slices);
            // printf("slice_length = %e\n", slice_length);
            // printf("check = %d\n", check);

            double k0_kick = 0;
            double k0_drift = 0;
            if (model ==0 || model==1 || model==3){
                // Slice is short w.r.t. bending radius
                k0_kick = k0;
                k0_drift = 0;
            }
            else {
                // method is 2
                // Force bend-kick-bend
                k0_kick = 0;
                k0_drift = k0;
            }

            // printf("k0_kick = %e\n", k0_kick);

            // Check if it can be handled without slicing
            int no_slice_needed = 0;
            if (k0_kick == 0 && k1 == 0){
                int multip_present = 0;
                for (int mm=0; mm<=order; mm++){
                    if (knl[mm] != 0 || ksl[mm] != 0){
                        multip_present = 1;
                        break;
                    }
                }
                if (!multip_present){
                    no_slice_needed = 1;
                }
            }

            if (no_slice_needed){
                // printf("No slicing needed\n");
                //start_per_particle_block (part0->part)
                    track_thick_bend(part, length, k0_drift, h);
                //end_per_particle_block
            }
            else{
                for (int ii = 0; ii < num_slices; ii++) {
                    //start_per_particle_block (part0->part)
                        track_thick_bend(part, slice_length * d_yoshida[0], k0_drift, h);
                        track_multipolar_kick_bend(
                            part, order, inv_factorial_order, knl, ksl,
                            kick_weight * k_yoshida[0], k0_kick, k1, h, length);
                        track_thick_bend(part, slice_length * d_yoshida[1], k0_drift, h);
                        track_multipolar_kick_bend(
                            part, order, inv_factorial_order, knl, ksl,
                            kick_weight * k_yoshida[1], k0_kick, k1, h, length);
                        track_thick_bend(part, slice_length * d_yoshida[2], k0_drift, h);
                        track_multipolar_kick_bend(
                            part, order, inv_factorial_order, knl, ksl,
                            kick_weight * k_yoshida[2], k0_kick, k1, h, length);
                        track_thick_bend(part, slice_length * d_yoshida[3], k0_drift, h);
                        track_multipolar_kick_bend(
                            part, order, inv_factorial_order, knl, ksl,
                            kick_weight * k_yoshida[3], k0_kick, k1, h, length);
                        track_thick_bend(part, slice_length * d_yoshida[3], k0_drift, h);
                        track_multipolar_kick_bend(
                            part, order, inv_factorial_order, knl, ksl,
                            kick_weight * k_yoshida[2], k0_kick, k1, h, length);
                        track_thick_bend(part, slice_length * d_yoshida[2], k0_drift, h);
                        track_multipolar_kick_bend(
                            part, order, inv_factorial_order, knl, ksl,
                            kick_weight * k_yoshida[1], k0_kick, k1, h, length);
                        track_thick_bend(part, slice_length * d_yoshida[1], k0_drift, h);
                        track_multipolar_kick_bend(
                            part, order, inv_factorial_order, knl, ksl,
                            kick_weight * k_yoshida[0], k0_kick, k1, h, length);
                        track_thick_bend(part, slice_length * d_yoshida[0], k0_drift, h);
                    //end_per_particle_block
                }
            }

    }
    if (model==4){
        const double slice_length = length / (num_multipole_kicks + 1);
        const double kick_weight = 1. / num_multipole_kicks;
        //start_per_particle_block (part0->part)
            track_thick_cfd(part, slice_length, k0, k1, h);

            for (int ii = 0; ii < num_multipole_kicks; ii++) {
                multipolar_kick(part, order, inv_factorial_order, knl, ksl, kick_weight);
                track_thick_cfd(part, slice_length, k0, k1, h);
            }
        //end_per_particle_block
    }

}

#endif // XTRACK_TRUEBEND_H