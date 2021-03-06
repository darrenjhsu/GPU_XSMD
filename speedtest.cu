#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include "kernel.cu"
#include "speedtest.hh"
#include "env_param.hh"
#include "mol_param.hh"
#include "scat_param.hh"
#include "coord_ref.hh"
//#include "raster8.hh"

int main () {
    cudaFree(0);
    float *S_calc,
          *Force;
    int   *close_num, 
          *close_idx;
    float *V;
    float *FF_table; 

    // Declare cuda pointers //
    float *d_coord;          // Coordinates 3 x num_atom
    float *d_Force;          // Force 3 x num_atom
    int   *d_Ele;              // Element list.

    float *d_q_S_ref_dS;     /* q vector, reference scattering pattern, and 
                                measured difference pattern to fit.
                                Since they are of same size they're grouped */
                                
    float *d_Aq;             // Prefactor for each q
    float *d_S_calc;         // Calculated scattering curve

    float *d_S_calcc,        // Some intermediate matrices
          *d_f_ptxc, 
          *d_f_ptyc, 
          *d_f_ptzc;
    
    float *d_V,              // Exposed surf area (in num of dots) 
          *d_V_s;            // Exposed surf area (in real A^2)

    float *d_WK;             // Waasmaier-Kirfel parameters 

    int   *d_close_flag,     // Flags for atoms close to an atom
          *d_close_num,      // Num of atoms close to an atom
          *d_close_idx;      // Their atomic index
 
    float *d_vdW;            // van der Waals radii

    float *d_FF_table,       // Form factors for each atom type at each q
          *d_FF_full;        /* Form factors for each atom at each q, 
                                considering the SASA an atom has. */

    // If using HyPred mode, then an array of c2 is needed. //
    // float *d_c2;
    
    // set various memory chunk sizes
    int size_coord       = 3 * num_atom * sizeof(float);
    int size_atom        = num_atom * sizeof(int);
    int size_atom2       = num_atom2 * sizeof(int);
    int size_atom2f      = num_atom2 * sizeof(float);
    int size_atom2xatom2 = 1024 * num_atom2 * sizeof(int); // For d_close_flag
    int size_q           = num_q * sizeof(float); 
    int size_qxatom2     = num_q2 * num_atom2 * sizeof(float);
    int size_FF_table    = (num_ele + 1) * num_q * sizeof(float); // +1 for solvent
    int size_WK          = 11 * num_ele * sizeof(float);
    int size_vdW         = (num_ele + 1) * sizeof(float); // +1 for solvent
    // int size_c2          = 10 * sizeof(float); // Only for HyPred

    // Allocate local memories
    S_calc = (float *)malloc(size_q);
    Force = (float *)malloc(size_coord);
    close_idx = (int *)malloc(size_atom2xatom2);
    close_num = (int *)malloc(size_atom2);
    V = (float *)malloc(size_atom2f);
    FF_table = (float *)malloc(size_FF_table);

    // Allocate cuda memories
    cudaMalloc((void **)&d_Aq,         size_q);
    cudaMalloc((void **)&d_coord,      size_coord); // 40 KB
    cudaMalloc((void **)&d_Force,      size_coord); // 40 KB
    cudaMalloc((void **)&d_Ele,        size_atom);
    cudaMalloc((void **)&d_q_S_ref_dS, 3 * size_q);
    cudaMalloc((void **)&d_S_calc,     size_q); // Will be computed on GPU
    cudaMalloc((void **)&d_f_ptxc,     size_qxatom2);
    cudaMalloc((void **)&d_f_ptyc,     size_qxatom2);
    cudaMalloc((void **)&d_f_ptzc,     size_qxatom2);
    cudaMalloc((void **)&d_S_calcc,    size_qxatom2);
    cudaMalloc((void **)&d_V,          size_atom2f);
    cudaMalloc((void **)&d_V_s,        size_atom2f);
    cudaMalloc((void **)&d_close_flag, size_atom2xatom2);
    cudaMalloc((void **)&d_close_num,  size_atom2);
    cudaMalloc((void **)&d_close_idx,  size_atom2xatom2);
    cudaMalloc((void **)&d_vdW,        size_vdW);
    cudaMalloc((void **)&d_FF_table,   size_FF_table);
    cudaMalloc((void **)&d_FF_full,    size_qxatom2);
    cudaMalloc((void **)&d_WK,         size_WK);
    //cudaMalloc((void **)&d_c2,         size_c2); // Only for HyPred

    // Initialize some matrices
    cudaMemset(d_close_flag, 0,   size_qxatom2);
    cudaMemset(d_Force,      0.0, size_coord);
    cudaMemset(d_Aq,         0.0, size_q);
    cudaMemset(d_S_calc,     0.0, size_q);
    cudaMemset(d_f_ptxc,     0.0, size_qxatom2);
    cudaMemset(d_f_ptyc,     0.0, size_qxatom2);   
    cudaMemset(d_f_ptzc,     0.0, size_qxatom2);
    cudaMemset(d_S_calcc,    0.0, size_qxatom2);
    cudaMemset(d_close_num,  0,   size_atom2);
    cudaMemset(d_close_idx,  0,   size_atom2xatom2);
    cudaMemset(d_FF_full,    0.0, size_qxatom2);

    // Copy necessary data
    cudaMemcpy(d_coord,      coord_init, size_coord, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vdW,        vdW,        size_vdW,   cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ele,        Ele,        size_atom,  cudaMemcpyHostToDevice);
    cudaMemcpy(d_q_S_ref_dS, q_S_ref_dS, 3 * size_q, cudaMemcpyHostToDevice);
    cudaMemcpy(d_WK,         WK,         size_WK,    cudaMemcpyHostToDevice);
    // Only for HyPred
    //cudaMemcpy(d_c2,         c2,         size_c2,    cudaMemcpyHostToDevice);



    float sigma2 = 1.0;
    float alpha = 1.0;
     
    dist_calc<<<1024, 1024>>>(
        d_coord, 
        d_close_num, 
        d_close_flag,
        d_close_idx, 
        num_atom,
        num_atom2); 

    cudaMemcpy(close_num, d_close_num, size_atom2, cudaMemcpyDeviceToHost);
    cudaMemcpy(close_idx, d_close_idx, size_atom2xatom2, cudaMemcpyDeviceToHost);
    surf_calc<<<1024,512>>>(
        d_coord, 
        d_Ele, 
        d_close_num, 
        d_close_idx, 
        d_vdW, 
        num_atom, 
        num_atom2, 
        num_raster, 
        sol_s, 
        d_V);


    sum_V<<<1,1024>>>(
        d_V, 
        d_V_s, 
        num_atom, 
        num_atom2, 
        d_Ele, 
        d_vdW);

    cudaMemcpy(V, d_V, size_atom2f, cudaMemcpyDeviceToHost);

    // Print surf info
    /*float wa_c2 = 0.0; 
    float V_sum = 0.0;
    for (int i = 0; i < num_atom; i++) {
        printf("%3d atoms are close to atom %5d, %.6f of surf being exposed.\n", close_num[i], i, V[i]);
        V_sum += V[i];
        wa_c2 += V[i] * c2[Ele[i]];
        for (int j = 0; j < close_num[i]; j++) {
        //for (int j = 0; j < 30; j++) {
            printf("%5d, ", close_idx[i*1024+j]);
        }
        printf("\n");
    }
    printf("Weighed average c2 is %.5f . \n", wa_c2 / V_sum);*/
    
    FF_calc<<<320, 32>>>(
        d_q_S_ref_dS, 
        d_WK, 
        d_vdW, 
        num_q, 
        num_ele, 
        c1, 
        r_m, 
        d_FF_table);

    create_FF_full_FoXS<<<320, 1024>>>(
        d_FF_table, 
        d_V,
        c2, 
        d_Ele, 
        d_FF_full, 
        num_q, 
        num_ele, 
        num_atom, 
        num_atom2);

    cudaMemcpy(FF_table, d_FF_table, size_FF_table, cudaMemcpyDeviceToHost);

    scat_calc<<<320, 1024>>>(
        d_coord, 
        d_Ele,
        d_q_S_ref_dS, 
        d_S_calc, 
        num_atom,  
        num_q,     
        num_ele,  
        d_Aq, 
        alpha,    
        k_chi,     
        sigma2,    
        d_f_ptxc, 
        d_f_ptyc, 
        d_f_ptzc, 
        d_S_calcc, 
        num_atom2, 
        d_FF_full);

    cudaMemcpy(S_calc, d_S_calc, size_q,     cudaMemcpyDeviceToHost);

    force_calc<<<1024, 512>>>(
        d_Force, 
        num_atom, 
        num_q, 
        d_f_ptxc, 
        d_f_ptyc, 
        d_f_ptzc, 
        num_atom2, 
        num_q2, 
        d_Ele);

    cudaMemcpy(Force,  d_Force,  size_coord, cudaMemcpyDeviceToHost);


    float chi = 0.0;
    float chi2 = 0.0;
    float chi_ref = 0.0;
    for (int ii = 0; ii < num_q; ii++) {
        chi = q_S_ref_dS[ii+2*num_q] - (S_calc[ii] - q_S_ref_dS[ii+num_q]);
        printf("q = %.3f: chi is: %.3f, dS is: %.3f, S_calc is: %.3f, S_ref is: %.3f\n", q_S_ref_dS[ii], chi, q_S_ref_dS[ii+2*num_q], S_calc[ii], q_S_ref_dS[ii+num_q]); 
        chi2 += chi * chi;
        chi_ref+= q_S_ref_dS[ii+2*num_q] * q_S_ref_dS[ii+2*num_q];
    }
    /*for (int ii = 0; ii < 3 * num_atom; ii++) {
        printf("%d: %.8f ", ii/3, Force[ii]);
        if ((ii+1) % 3 == 0) printf("\n");
    }
    for (int ii = 0; ii < num_atom; ii++) {
        printf("grad: %4d: %7.4f %7.4f %7.4f \n", ii, surf_grad[3*ii], surf_grad[3*ii+1], surf_grad[3*ii+2]);
    }*/
    printf("chi square is %.5e ( %.3f % )\n", chi2, chi2 / chi_ref * 100);
    /*for (int ii = 0; ii < 1; ii++) {
        printf("S0 = %.5e \n", S_calc[ii]);
    }*/

    // Print surface points
    /*
            printf("CRYST1    0.000    0.000    0.000  90.00  90.00  90.00 P 1           1\n");
    int idx = 0;
    for (int ii = 0; ii < num_atom * num_raster; ii++) {
        if (surf[3*ii] != 0) {
            printf("ATOM  %5d  XXX XXX P   1     %7.3f %7.3f %7.3f  0.00  0.00      P1\n", idx, surf[3*ii], surf[3*ii+1], surf[3*ii+2]);
            idx++;
        }
    }
    */
    cudaFree(d_coord); 
    cudaFree(d_Force); 
    cudaFree(d_Ele); 
    cudaFree(d_q_S_ref_dS); 
    cudaFree(d_Aq);
    cudaFree(d_S_calc); 
    cudaFree(d_f_ptxc); cudaFree(d_f_ptyc); cudaFree(d_f_ptzc);
    cudaFree(d_S_calcc); 
    cudaFree(d_WK);
    cudaFree(d_V); cudaFree(d_V_s); 
    cudaFree(d_close_flag); cudaFree(d_close_num); cudaFree(d_close_idx);
    cudaFree(d_vdW);
    cudaFree(d_FF_table); cudaFree(d_FF_full);
    //cudaFree(d_c2);


    free(S_calc); free(Force);
    free(close_num); free(close_idx);
    free(V); free(FF_table);

    return 0;
}
