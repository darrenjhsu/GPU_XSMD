
#include <math.h>
#include "param.hh"
#include "WaasKirf.hh"

__global__ void scat_calc (double *coord, double *Force, int *Ele, double *FF, double *q, double *S_ref, double *dS, double *S_calc, int num_atom, int num_q, int num_ele, double *Aq, double alpha, double k_chi, double sigma2, double *f_ptxc, double *f_ptyc, double *f_ptzc, double *S_calcc, int num_atom2, int num_q2) {
//__global__ void scat_calc (double *coord, double *Force, int *Ele, double *WK, double *q, double *S_ref, double *dS, double *S_calc, int num_atom, int num_q, int num_ele, double *Aq, double alpha, double k_chi, double sigma2, double *f_ptxc, double *f_ptyc, double *f_ptzc, double *S_calcc, int num_atom2, int num_q2) {
    //__shared__ double FF_pt[4];
    
    if (blockIdx.x > num_q) return; // out of q range
    if (threadIdx.x > num_atom) return; // out of atom numbers (not happening)
        
    /*if (blockIdx.x == 0) {
        for (int jj = threadIdx.x; jj < num_atom; jj += blockDim.x) {
            f_ptx[jj] = 0.0;
            f_pty[jj] = 0.0;
            f_ptz[jj] = 0.0;
        }
        for (int jj = threadIdx.x; jj < num_q; jj += blockDim.x) {
            Aq[jj] = 0.0;
            S_calc[jj] = 0.0;
        }
    }
    __syncthreads();
    */


    //for (int ii = blockIdx.x * blockDim.x + threadIdx.x; ii < num_q; ii += blockDim.x * gridDim.x) {
    for (int ii = blockIdx.x; ii < num_q; ii += gridDim.x) {
          //       0 - 512          300          512
        double q_pt = q[ii];
        // Calculate Form factor for this block (or q vector)
        /*for (int jj = threadIdx.x; jj < num_ele; jj += blockDim.x) {
            FF_pt[jj] = WK[jj*11] * exp(-WK[jj*11+6] * q_pt * q_pt) + \
                        WK[jj*11+1] * exp(-WK[jj*11+7] * q_pt * q_pt) + \
                        WK[jj*11+2] * exp(-WK[jj*11+8] * q_pt * q_pt) + \
                        WK[jj*11+3] * exp(-WK[jj*11+9] * q_pt * q_pt) + \
                        WK[jj*11+4] * exp(-WK[jj*11+10] * q_pt * q_pt) + \
                        WK[jj*11+5];
        }
        __syncthreads();*/
        // Calculate scattering for Aq
        //for (int jj = blockIdx.y * blockDim.y + threadIdx.y; jj < num_atom; jj += blockIdx.y * gridDim.y) {
        for (int jj = threadIdx.x; jj < num_atom; jj += blockDim.x) {
              //       0 - 1023          1749            1024
            double atom1x = coord[3*jj+0];
            double atom1y = coord[3*jj+1];
            double atom1z = coord[3*jj+2];
            int atom1t = Ele[jj]; // atom1 element type
            double atom1FF = FF[ii*num_ele+atom1t]; // atom1 form factor at q
            //double atom1FF = FF_pt[atom1t]; // atom1 form factor at q
            for (int kk = 0; kk < num_atom; kk++) {
                int atom2t = Ele[kk];
                if (q_pt == 0.0) {
                    S_calcc[ii*num_atom2+jj] += atom1FF * FF[ii*num_ele+atom2t];
                    //S_calcc[ii*num_atom2+jj] += atom1FF * FF_pt[atom2t];
                    //*a = 1;
                } else if (kk == jj) {
                    S_calcc[ii*num_atom2+jj] += atom1FF * FF[ii*num_ele+atom2t];
                    //S_calcc[ii*num_atom2+jj] += atom1FF * FF_pt[atom2t];
                } else {
                    double dx = coord[3*kk+0] - atom1x;
                    double dy = coord[3*kk+1] - atom1y;
                    double dz = coord[3*kk+2] - atom1z;
                    double r = sqrt(dx*dx+dy*dy+dz*dz);
                    S_calcc[ii*num_atom2+jj] += atom1FF * FF[ii*num_ele+atom2t] * sin(q_pt * r) / q_pt / r;
                    double prefac = atom1FF * FF[ii*num_ele+atom2t] * (cos(q_pt * r) - sin(q_pt * r) / q_pt / r) / r / r;
                    //S_calcc[ii*num_atom2+jj] += atom1FF * FF_pt[atom2t] * sin(q_pt * r) / q_pt / r;
                    //double prefac = atom1FF * FF_pt[atom2t] * (cos(q_pt * r) - sin(q_pt * r) / q_pt / r) / r / r;
                    f_ptxc[ii*num_atom2+jj] += prefac * dx;
                    f_ptyc[ii*num_atom2+jj] += prefac * dy;
                    f_ptzc[ii*num_atom2+jj] += prefac * dz;
                }
            }
            
        }

        // S_calc[ii] += S_pt;
        // Tree-like summation of S_calcc to get S_calc
        for (int stride = num_atom2 / 2; stride > 0; stride >>= 1) {
            __syncthreads();
            for(int iAccum = threadIdx.x; iAccum < stride; iAccum += blockDim.x) {
                S_calcc[ii * num_atom2 + iAccum] += S_calcc[ii * num_atom2 + stride + iAccum];
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            S_calc[ii] = S_calcc[ii * num_atom2];
            Aq[ii] = k_chi / 2.0 / sigma2 * ( dS[ii] - alpha * (S_calc[ii] - S_ref[ii]));
        }
        __syncthreads();
        // Multiply f_pt{x,y,z}c(q) by Aq(q) * 8 * alpha * k_chi / sigma2
        for (int jj = threadIdx.x; jj < num_atom; jj += blockDim.x) {
            f_ptxc[ii * num_atom2 + jj] *= (Aq[ii] * 4.0 * alpha);
            f_ptyc[ii * num_atom2 + jj] *= (Aq[ii] * 4.0 * alpha);
            f_ptzc[ii * num_atom2 + jj] *= (Aq[ii] * 4.0 * alpha);
        }
        __syncthreads();
        // Call another device function (block = atom_num, threads = num_q)
        // to column sum f_pt{x,y,z}c for Force[jj] 
        /*for (int jj = threadIdx.x; jj < num_atom; jj += blockDim.x) {
              //       0 - 1023          1749            1024
            //Force[jj] += 8 * alpha * k_chi / sigma2 * Aq[ii];
            Force[jj] = 0.0;
        }*/
    }
}


__global__ void force_calc (double *Force, double *q, int num_atom, int num_q, double *f_ptxc, double *f_ptyc, double *f_ptzc, int num_atom2, int num_q2) {
    // Do column tree sum of f_ptxc for f_ptx for every atom, then assign threadIdx.x == 0 (3 * num_atoms) to Force. Force is num_atom * 3. 
    if (blockIdx.x > num_atom) return;
    for (int ii = blockIdx.x; ii < num_atom; ii += gridDim.x) {
        for (int stride = num_atom2 / 2; stride > 0; stride >>= 1) {
            __syncthreads();
            for(int iAccum = threadIdx.x; iAccum < stride; iAccum += blockDim.x) {
                f_ptxc[ii + iAccum * num_atom2] += f_ptxc[ii + iAccum * num_atom2 + stride * num_atom2];
                f_ptyc[ii + iAccum * num_atom2] += f_ptyc[ii + iAccum * num_atom2 + stride * num_atom2];
                f_ptzc[ii + iAccum * num_atom2] += f_ptzc[ii + iAccum * num_atom2 + stride * num_atom2];
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            Force[ii*3    ] = -f_ptxc[ii];
            Force[ii*3 + 1] = -f_ptyc[ii];
            Force[ii*3 + 2] = -f_ptzc[ii];
        }
        __syncthreads();
    }

}
