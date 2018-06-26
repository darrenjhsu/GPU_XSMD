
#include <stdio.h>
#include <math.h>
#include "kernel.cu"
#include "XSMD.hh"
#include "param.hh"

void XSMD_calc (double *coord, double *Force) {
    //int *Ele, double *FF, double *q, double *S_ref, double *dS, double *S_calc, int num_atom, int num_q, int num_ele, double k_chi)
    double *d_Aq, *d_coord, *d_Force, *d_FF, *d_q;
    int *d_Ele;
    double *d_S_ref, *d_dS, *d_S_calc;
    double *S_calc;
    double *d_S_calcc, *d_f_ptxc, *d_f_ptyc, *d_f_ptzc;
    //int *a, *d_a; 
    //a = (int *)malloc(sizeof(int)); 
    //cudaMalloc((void **)&d_a,sizeof(int));
    //cudaMemset(d_a, 0, sizeof(int));
    int size_coord = 3 * num_atom * sizeof(double);
    int size_atom = num_atom * sizeof(int);
    int size_atom2 = num_atom * sizeof(double);
    int size_q = num_q * sizeof(double); 
    int size_FF = num_ele * num_q * sizeof(double);
    int size_qxatom2 = num_q2 * num_atom2 * sizeof(double); // check if overflow
    // Initialize Force array
    //Force = (double *)malloc(size_coord);
    for (int ii = 0; ii<3*num_atom; ii++) {
        Force[ii] = 0.0;
    }

    S_calc = (double *)malloc(size_q);
    //for (int ii = 0; ii < num_q; ii++) {
    //    S_calc[ii] = 0.0;
    //}

    cudaMalloc((void **)&d_Aq,     size_q);
    cudaMemset(d_Aq, 0.0, size_q);
    cudaMalloc((void **)&d_coord,  size_coord); // 40 KB
    cudaMalloc((void **)&d_Force,  size_coord); // 40 KB
    cudaMemset(d_Force, 0.0, size_coord);
    cudaMalloc((void **)&d_FF,     size_FF);  // 10 KB ?
    cudaMalloc((void **)&d_q,      size_q);  
    cudaMalloc((void **)&d_Ele,    size_atom);
    cudaMalloc((void **)&d_S_ref,  size_q);
    cudaMalloc((void **)&d_dS,     size_q);
    cudaMalloc((void **)&d_S_calc, size_q); // Will be computed on GPU
    cudaMemset(d_S_calc, 0.0, size_q);
    cudaMalloc((void **)&d_f_ptxc, size_qxatom2);
    cudaMemset(d_f_ptxc,0.0, size_qxatom2);
    cudaMalloc((void **)&d_f_ptyc, size_qxatom2);
    cudaMemset(d_f_ptyc,0.0, size_qxatom2);   
    cudaMalloc((void **)&d_f_ptzc, size_qxatom2);
    cudaMemset(d_f_ptzc,0.0, size_qxatom2);
    cudaMalloc((void **)&d_S_calcc, size_qxatom2);
    cudaMemset(d_S_calcc,0.0, size_qxatom2);   
    cudaMemcpy(d_coord, coord, size_coord, cudaMemcpyHostToDevice);
    //cudaMemcpy(d_Force, Force, size_coord, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q,     q,     size_q,     cudaMemcpyHostToDevice);
    cudaMemcpy(d_FF,    FF,    size_FF,    cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ele,   Ele,   size_atom,  cudaMemcpyHostToDevice);
    cudaMemcpy(d_S_ref, S_ref, size_q,     cudaMemcpyHostToDevice);
    cudaMemcpy(d_dS,    dS,    size_q,     cudaMemcpyHostToDevice);

    //k_chi = 5e-10;
    double sigma2 = 1.0;
    double alpha = 1.0;
    //printf("About to start force_calc...\n");
    scat_calc<<<512, 128>>>(d_coord, d_Force, d_Ele, d_FF, d_q, d_S_ref, d_dS, d_S_calc, num_atom, num_q, num_ele, d_Aq, alpha, k_chi, sigma2, d_f_ptxc, d_f_ptyc, d_f_ptzc, d_S_calcc, num_atom2, num_q2);
    //printf("force_calc finished! \n");
    force_calc<<<128, 512>>>(d_Force, d_q, num_atom, num_q, d_f_ptxc, d_f_ptyc, d_f_ptzc, num_atom2, num_q2);
    cudaMemcpy(S_calc, d_S_calc, size_q,     cudaMemcpyDeviceToHost);
    cudaMemcpy(Force,  d_Force,  size_coord, cudaMemcpyDeviceToHost);
    //cudaMemcpy(a,      d_a,      sizeof(int),cudaMemcpyDeviceToHost);
    double chi = 0.0;
    double chi2 = 0.0;
    for (int ii = 0; ii < num_q; ii++) {
        chi = dS[ii] - (S_calc[ii] - S_ref[ii]);
        chi2 += chi * chi;
    }
    /*for (int ii = 0; ii < 3 * num_atom; ii++) {
        printf("%.8f ", Force[ii]);
        if ((ii+1) % 3 == 0) printf("\n");
    }*/
    printf("chi square is %.5e ( %.3f \% )\n",chi2, chi2 / 7.80177e+10 * 100);
    /*for (int ii = 0; ii < 1; ii++) {
        printf("%.5e and a is %d\n", S_calc[ii], *a);
    }*/

    cudaFree(d_coord); cudaFree(d_Force); cudaFree(d_q);
    cudaFree(d_Ele); cudaFree(d_FF); cudaFree(d_S_ref); 
    cudaFree(d_dS); cudaFree(d_S_calc); cudaFree(d_Aq);
    cudaFree(d_f_ptxc); cudaFree(d_f_ptyc); cudaFree(d_f_ptzc);
    cudaFree(d_S_calcc);
    //cudaFree(d_a); free(a);
    free(S_calc);


}
