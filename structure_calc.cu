#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include "kernel.cu"
#include "speedtest.hh"
#include "param.hh"
#include "coord_ref.hh"
//#include "raster8.hh"

int main () {
    cudaFree(0); 
    float *d_Aq, *d_coord, *d_Force, *d_FF;
    int *d_Ele;
    float *d_q_S_ref_dS, *d_S_calc;
    float *S_calc1, *S_calc2;
    float *d_S_calcc, *d_f_ptxc, *d_f_ptyc, *d_f_ptzc;
    float *d_dx, *d_dy, *d_dz;
    float *d_raster, *d_V, *d_r2;
    float *d_WK;
    float *Force;
    int *d_close_flag, *d_close_num, *d_close_idx;
    float *d_vdW;
    int *close_num, *close_idx;
    float *V;
    float *d_FF_table;
    float *d_surf, *surf, *d_surf_grad;

    // set various memory chunk sizes
    int size_coord = 3 * num_atom * sizeof(float);
    int size_atom = num_atom * sizeof(int);
    int size_atom2 = num_atom2 * sizeof(int);
    int size_atom2f = num_atom2 * sizeof(float);
    int size_atomxatom = num_atom * num_atom * sizeof(float);
    int size_atom2xatom2 = num_atom2 * num_atom2 * sizeof(int);
    printf("size_atomxatom = %d. \n", size_atomxatom);
    int size_q = num_q * sizeof(float); 
    int size_FF = num_ele * num_q * sizeof(float);
    int size_qxatom2 = num_q2 * num_atom2 * sizeof(float); // check if overflow
    int size_raster = num_raster * 3 * sizeof(float);
    int size_FF_table = (num_ele+1) * num_q * sizeof(float);
    int size_surf = num_atom * num_raster * 3 * sizeof(float);
    int size_WK = 11 * num_ele * sizeof(float);
    int size_vdW = (num_ele+1) * sizeof(float);


    // Allocate local memories
    Force = (float *)malloc(size_coord);
    close_idx = (int *)malloc(size_atom2xatom2);
    close_num = (int *)malloc(size_atom2);
    V = (float *)malloc(size_atom2f);
    S_calc1 = (float *)malloc(size_q);
    S_calc2 = (float *)malloc(size_q);
    surf = (float *)malloc(size_surf);

    // Allocate cuda memories
    cudaMalloc((void **)&d_Aq,     size_q);
    cudaMalloc((void **)&d_coord,  size_coord); // 40 KB
    cudaMalloc((void **)&d_Force,  size_coord); // 40 KB
    cudaMalloc((void **)&d_FF,     size_FF);  // 10 KB ?
    cudaMalloc((void **)&d_Ele,    size_atom);
    cudaMalloc((void **)&d_q_S_ref_dS, 3 * size_q);
    cudaMalloc((void **)&d_S_calc, size_q); // Will be computed on GPU
    cudaMalloc((void **)&d_f_ptxc, size_qxatom2);
    cudaMalloc((void **)&d_f_ptyc, size_qxatom2);
    cudaMalloc((void **)&d_f_ptzc, size_qxatom2);
    cudaMalloc((void **)&d_S_calcc, size_qxatom2);
    cudaMalloc((void **)&d_V, size_atom2f);
    cudaMalloc((void **)&d_dx, size_atomxatom);
    cudaMalloc((void **)&d_dy, size_atomxatom);
    cudaMalloc((void **)&d_dz, size_atomxatom);
    cudaMalloc((void **)&d_r2, size_atomxatom);
    cudaMalloc((void **)&d_close_flag, size_atom2xatom2);
    cudaMalloc((void **)&d_close_num, size_atom2);
    cudaMalloc((void **)&d_close_idx, size_atom2xatom2);
    cudaMalloc((void **)&d_vdW, size_vdW);
    cudaMalloc((void **)&d_FF_table, size_FF_table);
    cudaMalloc((void **)&d_WK, size_WK);
    cudaMalloc((void **)&d_surf, size_surf);
    cudaMalloc((void **)&d_surf_grad, size_coord);
    // Initialize some matrices
    cudaMemset(d_close_flag, 0, size_qxatom2);
    cudaMemset(d_Force, 0.0, size_coord);
    cudaMemset(d_Aq, 0.0, size_q);
    cudaMemset(d_S_calc, 0.0, size_q);
    cudaMemset(d_f_ptxc,0.0, size_qxatom2);
    cudaMemset(d_f_ptyc,0.0, size_qxatom2);   
    cudaMemset(d_f_ptzc,0.0, size_qxatom2);
    cudaMemset(d_S_calcc,0.0, size_qxatom2);
    cudaMemset(d_close_num, 0, size_atom2);
    cudaMemset(d_close_idx, 0, size_atom2xatom2);
    cudaMemset(d_surf, 0.0, size_surf);
    // Copy necessary data
    cudaMemcpy(d_coord, coord_ref, size_coord,    cudaMemcpyHostToDevice);
    cudaMemcpy(d_vdW, vdW, size_vdW, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ele,    Ele,    size_atom,   cudaMemcpyHostToDevice);
    cudaMemcpy(d_q_S_ref_dS,  q_S_ref_dS, 3 * size_q,      cudaMemcpyHostToDevice);
    cudaMemcpy(d_WK,     WK,     size_WK,     cudaMemcpyHostToDevice);

    float sigma2 = 1.0;
    float alpha = 1.0;
     
    dist_calc<<<1024, 1024>>>(d_coord, d_dx, d_dy, d_dz, d_r2, d_close_flag, num_atom, num_atom2); 
    pre_scan_close<<<2048,1024>>>(d_close_flag, d_close_num, d_close_idx, num_atom2);
    cudaMemcpy(close_num, d_close_num, size_atom2, cudaMemcpyDeviceToHost);
    surf_calc<<<1024,512>>>(d_coord, d_Ele, d_r2, d_close_num, d_close_idx, d_vdW, num_atom, num_atom2, num_raster, sol_s, d_V, d_surf, d_surf_grad, offset);
    sum_V<<<1,1024>>>(d_V, num_atom, num_atom2, d_Ele, d_vdW);
    FF_calc<<<320, 32>>>(d_q_S_ref_dS, d_WK, d_vdW, num_q, num_ele, c1, r_m, d_FF_table); 
    scat_calc<<<320, 1024>>>(d_coord,  d_Force,   d_Ele,     d_WK,     d_q_S_ref_dS, 
                             d_S_calc, num_atom,  num_q,     num_ele,  d_Aq, 
                             alpha,    k_chi,     sigma2,    d_f_ptxc, d_f_ptyc, 
                             d_f_ptzc, d_S_calcc, num_atom2, num_q2,   d_vdW,
                             c1,       c2,        d_V,       r_m,      d_FF_table,
                             d_surf_grad);
    cudaMemcpy(S_calc1,d_S_calc, size_q,     cudaMemcpyDeviceToHost);
    cudaMemcpy(surf,   d_surf,   size_surf,  cudaMemcpyDeviceToHost);
    force_calc<<<1024, 512>>>(d_Force, num_atom, num_q, d_f_ptxc, d_f_ptyc, d_f_ptzc, num_atom2, num_q2, d_Ele);
  
    // Initialize some matrices
    cudaMemset(d_close_flag, 0, size_qxatom2);
    cudaMemset(d_Force, 0.0, size_coord);
    cudaMemset(d_Aq, 0.0, size_q);
    cudaMemset(d_S_calc, 0.0, size_q);
    cudaMemset(d_f_ptxc,0.0, size_qxatom2);
    cudaMemset(d_f_ptyc,0.0, size_qxatom2);   
    cudaMemset(d_f_ptzc,0.0, size_qxatom2);
    cudaMemset(d_S_calcc,0.0, size_qxatom2);
    cudaMemset(d_close_num, 0, size_atom2);
    cudaMemset(d_close_idx, 0, size_atom2xatom2);
    cudaMemset(d_surf, 0.0, size_surf);

    // Do the next structure
    cudaMemcpy(d_coord, coord_init, size_coord,    cudaMemcpyHostToDevice);

    dist_calc<<<1024, 1024>>>(d_coord, d_dx, d_dy, d_dz, d_r2, d_close_flag, num_atom, num_atom2); 
    pre_scan_close<<<2048,1024>>>(d_close_flag, d_close_num, d_close_idx, num_atom2);
    cudaMemcpy(close_num, d_close_num, size_atom2, cudaMemcpyDeviceToHost);
    surf_calc<<<1024,512>>>(d_coord, d_Ele, d_r2, d_close_num, d_close_idx, d_vdW, num_atom, num_atom2, num_raster, sol_s, d_V, d_surf, d_surf_grad, offset);
    sum_V<<<1,1024>>>(d_V, num_atom, num_atom2, d_Ele, d_vdW);
    FF_calc<<<320, 32>>>(d_q_S_ref_dS, d_WK, d_vdW, num_q, num_ele, c1, r_m, d_FF_table); 
    scat_calc<<<320, 1024>>>(d_coord,  d_Force,   d_Ele,     d_WK,     d_q_S_ref_dS, 
                             d_S_calc, num_atom,  num_q,     num_ele,  d_Aq, 
                             alpha,    k_chi,     sigma2,    d_f_ptxc, d_f_ptyc, 
                             d_f_ptzc, d_S_calcc, num_atom2, num_q2,   d_vdW,
                             c1,       c2,        d_V,       r_m,      d_FF_table,
                             d_surf_grad);
    cudaMemcpy(S_calc2,d_S_calc, size_q,     cudaMemcpyDeviceToHost);
    cudaMemcpy(surf,   d_surf,   size_surf,  cudaMemcpyDeviceToHost);
    force_calc<<<1024, 512>>>(d_Force, num_atom, num_q, d_f_ptxc, d_f_ptyc, d_f_ptzc, num_atom2, num_q2, d_Ele);
 

    printf("float q_S_ref_dS[%d] = {", 3*num_q);
    for (int ii = 0; ii < num_q; ii++) {
        printf("%f, ",q_S_ref_dS[ii]);
    }
    printf("\n");
    for (int ii = 0; ii < num_q; ii++) {
        printf("%f, ",S_calc2[ii]);
    }
    printf("\n");
    for (int ii = 0; ii < num_q; ii++) {
        printf("%f", S_calc1[ii]-S_calc2[ii]);
        if (ii < num_q - 1) printf(", ");
    }
    printf("};\n");
   
    for (int ii = 0; ii < num_q; ii ++) {
        printf("%f, ",S_calc2[ii]);
    } 

    // Calculating chi square
    /*
    float chi = 0.0;
    float chi2 = 0.0;
    float chi_ref = 0.0;
    for (int ii = 0; ii < num_q; ii++) {
        chi = q_S_ref_dS[ii+2*num_q] - (S_calc[ii] - q_S_ref_dS[ii+num_q]);
        printf("q = %.3f: chi is: %.3f, dS is: %.3f, S_calc is: %.3f, S_ref is: %.3f\n", q_S_ref_dS[ii], chi, q_S_ref_dS[ii+2*num_q], S_calc[ii], q_S_ref_dS[ii+num_q]); 
        chi2 += chi * chi;
        chi_ref+= q_S_ref_dS[ii+2*num_q] * q_S_ref_dS[ii+2*num_q];
    }
    printf("chi square is %.5e ( %.3f \% )\n", chi2, chi2 / chi_ref * 100);
    */

    // Print surface points for PDB exhibition
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

    // Free cuda and local memories
    cudaFree(d_coord); cudaFree(d_Force); 
    cudaFree(d_Ele); cudaFree(d_FF); 
    cudaFree(d_q_S_ref_dS); 
    cudaFree(d_S_calc); cudaFree(d_Aq);
    cudaFree(d_f_ptxc); cudaFree(d_f_ptyc); cudaFree(d_f_ptzc);
    cudaFree(d_S_calcc); cudaFree(d_WK);
    cudaFree(d_dx); cudaFree(d_dy); cudaFree(d_dz);
    cudaFree(d_V); 
    cudaFree(d_r2);
    cudaFree(d_close_flag); cudaFree(d_close_num); cudaFree(d_close_idx);
    cudaFree(d_vdW);
    free(S_calc1); free(S_calc2); free(close_num); free(close_idx);

    return 0;
}
