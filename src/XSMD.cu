
#include <stdio.h>
#include <math.h>
#include "kernel.cu"
#include "XSMD.hh"
#include "mol_param.hh"
#include "env_param.hh"
#include "scat_param.hh"
#include "WaasKirf.hh"


void XSMD_calc (float *coord, float *Force, double *scat, int frame_num, double *EMA_norm) {
if (frame_num % 1000 == 0) {
    // In this code pointers with d_ are device pointers. 

    // Declare local pointers //
    // The calculated scattering pattern for this snapshot.
    float *S_calc;

    // Declare cuda pointers //
    float *d_coord;          // Coordinates 3 x num_atom
    float *d_Force;          // Force 3 x num_atom
    int   *d_Ele;              // Element list.

    float *d_q_S_ref_dS;     /* q vector, reference scattering pattern, and 
                                measured difference pattern to fit.
                                Since they are of same size they're grouped */
                                
    float *d_Aq;             // Prefactor for each q
    double *d_S_calc;         // Calculated scattering curve

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
    
    // Compute the exponential moving average normalization constant.
    // Here this final 500.0 is to say we average over 500 snapshots,
    // each snapshot taken every 1000 steps (the first if statement of this kernel).
    // So we have tau = 1.0 ns for exponential averaging.
    EMA_norm = EMA_norm * exp(-1.0/500.0) + 1;
    

    // If using HyPred mode, then an array of c2 is needed. //
    float *d_c2;
    
    // set various memory chunk sizes
    int size_coord       = 3 * num_atom * sizeof(float);
    int size_atom        = num_atom * sizeof(int);
    int size_atom2       = num_atom2 * sizeof(int);
    int size_atom2f      = num_atom2 * sizeof(float);
    int size_atom2xatom2 = 1024 * num_atom2 * sizeof(int); // For d_close_flag
    int size_q           = num_q * sizeof(float); 
    int size_double_q    = num_q * sizeof(double);
    int size_qxatom2     = num_q2 * num_atom2 * sizeof(float);
    int size_FF_table    = (num_ele + 1) * num_q * sizeof(float); // +1 for solvent
    int size_WK          = 11 * num_ele * sizeof(float);
    int size_vdW         = (num_ele + 1) * sizeof(float); // +1 for solvent
    int size_c2          = 10 * sizeof(float); // Only for HyPred

    // Allocate local memories
    S_calc = (float *)malloc(size_q);

    // Allocate cuda memories
    cudaMalloc((void **)&d_Aq,         size_q);
    cudaMalloc((void **)&d_coord,      size_coord); // 40 KB
    cudaMalloc((void **)&d_Force,      size_coord); // 40 KB
    cudaMalloc((void **)&d_Ele,        size_atom);
    cudaMalloc((void **)&d_q_S_ref_dS, 3 * size_q);
    cudaMalloc((void **)&d_S_calc,     size_double_q); // Will be computed on GPU
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
    cudaMalloc((void **)&d_c2,         size_c2); // Only for HyPred

    // Initialize some matrices
    cudaMemset(d_close_flag, 0,   size_qxatom2);
    cudaMemset(d_Force,      0.0, size_coord);
    cudaMemset(d_Aq,         0.0, size_q);
    cudaMemset(d_S_calc,     0.0, size_double_q);
    cudaMemset(d_f_ptxc,     0.0, size_qxatom2);
    cudaMemset(d_f_ptyc,     0.0, size_qxatom2);   
    cudaMemset(d_f_ptzc,     0.0, size_qxatom2);
    cudaMemset(d_S_calcc,    0.0, size_qxatom2);
    cudaMemset(d_close_num,  0,   size_atom2);
    cudaMemset(d_close_idx,  0,   size_atom2xatom2);
    cudaMemset(d_FF_full,    0.0, size_qxatom2);

    // Copy necessary data
    cudaMemcpy(d_coord,      coord,      size_coord, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vdW,        vdW,        size_vdW,   cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ele,        Ele,        size_atom,  cudaMemcpyHostToDevice);
    cudaMemcpy(d_q_S_ref_dS, q_S_ref_dS, 3 * size_q, cudaMemcpyHostToDevice);
    cudaMemcpy(d_WK,         WK,         size_WK,    cudaMemcpyHostToDevice);
    // Only for HyPred
    cudaMemcpy(d_c2,         c2_H,       size_c2,    cudaMemcpyHostToDevice);

    float sigma2 = 1.0;
    float alpha = 1.0;
     
    dist_calc<<<1024, 1024>>>(
        d_coord, 
        d_close_num, 
        d_close_flag,
        d_close_idx, 
        num_atom,
        num_atom2); 

    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if(error!=cudaSuccess)
    {
       fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
       exit(-1);
    }

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

    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if(error!=cudaSuccess)
    {
       fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
       exit(-1);
    }

    sum_V<<<1,1024>>>(
        d_V, 
        d_V_s, 
        num_atom, 
        num_atom2, 
        d_Ele, 
        d_vdW);

    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if(error!=cudaSuccess)
    {
       fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
       exit(-1);
    }

    FF_calc<<<320, 32>>>(
        d_q_S_ref_dS, 
        d_WK, 
        d_vdW, 
        num_q, 
        num_ele, 
        c1, 
        r_m, 
        d_FF_table);

    create_FF_full_HyPred<<<320, 1024>>>(
        d_FF_table, 
        d_V,
        c2 
        d_c2, 
        d_Ele, 
        d_FF_full, 
        num_q, 
        num_ele, 
        num_atom, 
        num_atom2);

/*    create_FF_full_FoXS<<<320, 1024>>>(
        d_FF_table, 
        d_V,
        c2, 
        d_Ele, 
        d_FF_full, 
        num_q, 
        num_ele, 
        num_atom, 
        num_atom2);
*/
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if(error!=cudaSuccess)
    {
       fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
       exit(-1);
    }

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

    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if(error!=cudaSuccess)
    {
       fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
       exit(-1);
    }

    cudaMemcpyAsync(S_calc, d_S_calc, size_q,     cudaMemcpyDeviceToHost);

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

    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if(error!=cudaSuccess)
    {
       fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
       exit(-1);
    }

    cudaMemcpy(Force,  d_Force,  size_coord, cudaMemcpyDeviceToHost);

    float chi = 0.0;
    float chi2 = 0.0;
    float chi_ref = 0.0;
    for (int ii = 0; ii < num_q; ii++) {
        chi = q_S_ref_dS[ii+2*num_q] - (S_calc[ii] - q_S_ref_dS[ii+num_q]);
        chi2 += chi * chi;
        chi_ref+= q_S_ref_dS[ii+2*num_q] * q_S_ref_dS[ii+2*num_q];
    }
    printf("chi square is %.5e ( %.3f % )\n", chi2, chi2 / chi_ref * 100);

 
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
    free(S_calc);
}
}
