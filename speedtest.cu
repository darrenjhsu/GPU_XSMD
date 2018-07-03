#include <stdio.h>
#include <math.h>
#include "kernel.cu"
#include "speedtest.hh"
#include "param.hh"
#include "coord_ref.hh"


int main () {
    //int *Ele, float *FF, float *q, float *S_ref, float *dS, float *S_calc, int num_atom, int num_q, int num_ele, float k_chi)
    //for (int ii = 0; ii < num_atom; ii ++) printf("%.3f, %.3f, %.3f\n",coord_ref[ii*3],coord_ref[ii*3+1],coord_ref[ii*3+2]) ;
    float *d_Aq, *d_coord, *d_Force, *d_FF;
    int *d_Ele;
    float *d_q_S_ref_dS, *d_S_calc;
    float *S_calc;
    float *d_S_calcc, *d_f_ptxc, *d_f_ptyc, *d_f_ptzc;
    //float *d_rot_pt, *d_rot;
    float *d_WK;
    float *Force;
    //int *d_bond_pp;
    //int *a, *d_a; 
    //a = (int *)malloc(sizeof(int)); 
    //cudaMalloc((void **)&d_a,sizeof(int));
    //cudaMemset(d_a, 0, sizeof(int));
    int size_coord = 3 * num_atom * sizeof(float);
    int size_atom = num_atom * sizeof(int);
    int size_atom2 = num_atom * sizeof(float);
    int size_q = num_q * sizeof(float); 
    int size_FF = num_ele * num_q * sizeof(float);
    int size_qxatom2 = num_q2 * num_atom2 * sizeof(float); // check if overflow
    //printf("size_qxatom2 = %d. \n", size_qxatom2);
    /*int size_bond_pp = 3 * num_pp * sizeof(int);
    int size_rot = num_pp * sizeof(int);
    int size_rotxatom2 = num_pp * num_atom2 * sizeof(float);*/
    int size_WK = 11 * num_ele * sizeof(float);
    // Initialize Force array
    Force = (float *)malloc(size_coord);
    /*for (int ii = 0; ii<3*num_atom; ii++) {
        Force[ii] = 0.0;
    }*/
    /*for (int ii = 0; ii < 66; ii ++) {
        printf("CPU: WK element %d is %.3f\n", ii, WK[ii]);
    }*/
    S_calc = (float *)malloc(size_q);
    //for (int ii = 0; ii < num_q; ii++) {
    //    S_calc[ii] = 0.0;
    //}
    
    cudaMalloc((void **)&d_Aq,     size_q);
    cudaMemset(d_Aq, 0.0, size_q);
    cudaMalloc((void **)&d_coord,  size_coord); // 40 KB
    cudaMalloc((void **)&d_Force,  size_coord); // 40 KB
    cudaMemset(d_Force, 0.0, size_coord);
    cudaMalloc((void **)&d_FF,     size_FF);  // 10 KB ?
    cudaMalloc((void **)&d_Ele,    size_atom);
    cudaMalloc((void **)&d_q_S_ref_dS, 3 * size_q);
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
    /*cudaMalloc((void **)&d_rot, size_rot);
    cudaMemset(d_rot,0.0, size_rot);
    cudaMalloc((void **)&d_rot_pt, size_rotxatom2);
    cudaMemset(d_rot_pt,0.0, size_rotxatom2);
    cudaMalloc((void **)&d_bond_pp, size_bond_pp);*/
    cudaMalloc((void **)&d_WK, size_WK);
    cudaMemcpy(d_coord, coord_ref, size_coord,    cudaMemcpyHostToDevice);
    //cudaMemcpy(d_Force, Force, size_coord, cudaMemcpyHostToDevice);
    //cudaMemcpy(d_q,      q,      size_q,      cudaMemcpyHostToDevice);
    //cudaMemcpy(d_FF,     FF,     size_FF,     cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ele,    Ele,    size_atom,   cudaMemcpyHostToDevice);
    cudaMemcpy(d_q_S_ref_dS,  q_S_ref_dS, 3* size_q,      cudaMemcpyHostToDevice);
    //cudaMemcpy(d_dS,     dS,     size_q,      cudaMemcpyHostToDevice);
    //cudaMemcpy(d_bond_pp,bond_pp,size_bond_pp,cudaMemcpyHostToDevice);
    cudaMemcpy(d_WK,     WK,     size_WK,     cudaMemcpyHostToDevice);
    //printf("Finished copying.\n");

    //k_chi = 5e-10;
    float sigma2 = 1.0;
    float alpha = 1.0;
    //printf("About to start force_calc...\n");
    //scat_calc<<<512, 128>>>(d_coord, d_Force, d_Ele, d_FF, d_q, d_S_ref, d_dS, d_S_calc, num_atom, num_q, num_ele, d_Aq, alpha, k_chi, sigma2, d_f_ptxc, d_f_ptyc, d_f_ptzc, d_S_calcc, num_atom2, num_q2);
    scat_calc<<<320, 1024>>>(d_coord, d_Force, d_Ele, d_WK, d_q_S_ref_dS, d_S_calc, num_atom, num_q, num_ele, d_Aq, alpha, k_chi, sigma2, d_f_ptxc, d_f_ptyc, d_f_ptzc, d_S_calcc, num_atom2, num_q2);
    //printf("force_calc finished! \n");
    //printf("%d \n",cudaDeviceSynchronize());
    cudaMemcpy(S_calc, d_S_calc, size_q,     cudaMemcpyDeviceToHost);
    force_calc<<<1024, 512>>>(d_Force, num_atom, num_q, d_f_ptxc, d_f_ptyc, d_f_ptzc, num_atom2, num_q2);
    
    //printf("%d \n",cudaDeviceSynchronize());
    //force_proj<<<32, 128>>>(d_coord, d_Force, d_rot, d_rot_pt, d_bond_pp, num_pp, num_atom, num_atom2);
    //cudaMemcpy(rot,    d_rot,    size_rot,   cudaMemcpyDeviceToHost);
    //pp_assign<<<1, 128>>>(d_coord, d_Force, d_rot, d_bond_pp, num_pp, num_atom);
    cudaMemcpy(Force,  d_Force,  size_coord, cudaMemcpyDeviceToHost);

 

    //cudaMemcpy(a,      d_a,      sizeof(int),cudaMemcpyDeviceToHost);
    float chi = 0.0;
    float chi2 = 0.0;
    float chi_ref = 0.0;
    for (int ii = 0; ii < num_q; ii++) {
        chi = q_S_ref_dS[ii+2*num_q] - (S_calc[ii] - q_S_ref_dS[ii+num_q]);
        //printf("%d: chi is: %.3f, dS is: %.3f, S_calc is: %.3f, S_ref is: %.3f\n", ii, chi, dS[ii], S_calc[ii], S_ref[ii]); 
        chi2 += chi * chi;
        chi_ref+= q_S_ref_dS[ii+2*num_q] * q_S_ref_dS[ii+2*num_q];
    }
    /*for (int ii = 0; ii < 3 * num_atom; ii++) {
        printf("%.8f ", Force[ii]);
        if ((ii+1) % 3 == 0) printf("\n");
    }*/
    printf("chi square is %.5e ( %.3f \% )\n", chi2, chi2 / chi_ref * 100);
    /*for (int ii = 0; ii < 1; ii++) {
        printf("S0 = %.5e \n", S_calc[ii]);
    }*/

    cudaFree(d_coord); cudaFree(d_Force); //cudaFree(d_q);
    cudaFree(d_Ele); cudaFree(d_FF); 
    cudaFree(d_q_S_ref_dS); 
    // cudaFree(d_dS);
    cudaFree(d_S_calc); cudaFree(d_Aq);
    cudaFree(d_f_ptxc); cudaFree(d_f_ptyc); cudaFree(d_f_ptzc);
    cudaFree(d_S_calcc); cudaFree(d_WK);

    //cudaFree(d_rot); cudaFree(d_rot_pt); cudaFree(d_bond_pp);
    //cudaFree(d_a); free(a);
    free(S_calc);
    //printf("So the fault is at NAMD?\n");

    return 0;
}
