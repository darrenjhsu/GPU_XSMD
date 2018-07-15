
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "param.hh"
#include "WaasKirf.hh"
#define PI 3.14159265359


/*__device__ float dot (float a1, float a2, float a3, float b1, float b2, float b3) {
    return (a1 * b1 + a2 * b2 + a3 * b3);
}

__device__ float cross2 (float a2, float a3, float b2, float b3) {
    return (a2 * b3 - a3 * b2);
}*/
 
//__global__ void scat_calc (float *coord, float *Force, int *Ele, float *FF, float *q, float *S_ref, float *dS, float *S_calc, int num_atom, int num_q, int num_ele, float *Aq, float alpha, float k_chi, float sigma2, float *f_ptxc, float *f_ptyc, float *f_ptzc, float *S_calcc, int num_atom2, int num_q2) {
/*__global__ void build_cxsxdx (float *cxsxdx_table, float *sx_table, float *qr_vec, float qr_max, float qr_step, int num_bin) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    if (idx >= num_bin) return;
    if (blockIdx.x == 0 && threadIdx.x == 0) printf("qr_max = %.3f, qr_step = %.3f, num_bin = %d. \n", qr_max, qr_step, num_bin);
    for (int ii = idx; ii < num_bin; ii += stride) {
        //Initialize table and qr_vec
        if (ii == 0) {
            qr_vec[ii] = 0.0;
            cxsxdx_table[ii] = 0.0;
            sx_table[ii] = 0.0;
        }
        qr_vec[ii] = (float)ii * qr_step;
        cxsxdx_table[ii] = cos(qr_vec[ii]) - sin(qr_vec[ii]) / qr_vec[ii];
        sx_table[ii] = sin(qr_vec[ii]);
    }

}
*/

__global__ void dist_calc (float *coord, float *dx, float *dy, float *dz, float *r2, int *close_flag, int num_atom, int num_atom2) {
    // r2 is the square of distances
    // close_flag is a num_atom2 x num_atom2 matrix initialized to 0.
    //__shared__ float coord_s[5247];
    __shared__ float x_ref, y_ref, z_ref;
    if (blockIdx.x >= num_atom) return;
    if (threadIdx.x >= num_atom) return;
    // Put coord into coord_s
    //for (int ii = threadIdx.x; ii < num_atom; ii += blockDim.x) {
    //    coord_s[ii] = coord[ii];
    //}
    // Calc distance
    for (int ii = blockIdx.x; ii < num_atom; ii += gridDim.x) {
        if (threadIdx.x == 0) {
            x_ref = coord[3*ii  ];
            y_ref = coord[3*ii+1];
            z_ref = coord[3*ii+2];
        }
    
        __syncthreads();
        for (int jj = threadIdx.x; jj < num_atom; jj += blockDim.x) {
            dx[ii*num_atom+jj] = coord[3*jj  ] - x_ref;
            dy[ii*num_atom+jj] = coord[3*jj+1] - y_ref;
            dz[ii*num_atom+jj] = coord[3*jj+2] - z_ref;
            float r2t = (coord[3*jj  ] - x_ref) * (coord[3*jj  ] - x_ref) + 
                        (coord[3*jj+1] - y_ref) * (coord[3*jj+1] - y_ref) + 
                        (coord[3*jj+2] - z_ref) * (coord[3*jj+2] - z_ref); 
 
            r2[ii*num_atom+jj] = r2t;
            if (r2t < 29.0) close_flag[ii*num_atom2+jj] = 1; // roughly 2 A vdW radii + 1.4 A water vdW
        }
    }
}

__global__ void pre_scan_close (int *close_flag, int *close_num, int *close_idx, int num_atom2) {
    // close_flag: A num_atom2 x num_atom2 boolean matrix
    // close_num: A num_atom2 int vector 
    // close_idx: A num_atom2 x num_atom2 int matrix, row i of which only the first close_num[i] elements are defined. (Otherwise it's 0). 
    // Do prefix sum for getting the index out - now up to 2048 ( = 1024 threads * 2 ) elements.
    if (blockIdx.x >= num_atom2) return;
    __shared__ int temp[2048];
    int idx = threadIdx.x; 
    int offset = 1;
    temp[2 * idx]     = close_flag[blockIdx.x * num_atom2 + 2 * idx];
    temp[2 * idx + 1] = close_flag[blockIdx.x * num_atom2 + 2 * idx + 1];
    for (int d = num_atom2>>1; d > 0; d >>= 1) {
        __syncthreads();
        if (idx < d) {
            int ai = offset * (2 * idx + 1) - 1;
            int bi = offset * (2 * idx + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    if (idx == 0) {
        close_num[blockIdx.x] = temp[num_atom2 - 1];
        temp[num_atom2 - 1] = 0;
    }
    for (int d = 1; d < num_atom2; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (idx < d) {
            int ai = offset * (2 * idx + 1) - 1;
            int bi = offset * (2 * idx + 2) - 1;
            int t    = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }

    __syncthreads();

    // Finally assign the indices
    if (close_flag[blockIdx.x * num_atom2 + 2 * idx] == 1) {
        close_idx[blockIdx.x * num_atom2 + temp[2*idx]] = 2*idx;
    }
    if (close_flag[blockIdx.x * num_atom2 + 2 * idx + 1] == 1) {
        close_idx[blockIdx.x * num_atom2 + temp[2*idx+1]] = 2*idx+1;
    }
}

__global__ void surf_calc (float *coord, int *Ele, float *r2, int *close_num, int *close_idx, float *vdW, int num_atom, int num_atom2, int num_raster, float sol_s, float *V) {
    // num_raster should be a number of 2^n. 
    // sol_s is solvent radius (default = 1.4 A)
    __shared__ float vdW_s; // vdW radius of the center atom
    __shared__ int pts[512]; // All spherical raster points
    __shared__ float L, r;
    if (blockIdx.x >= num_atom) return;
    if (threadIdx.x == 0) L = sqrt(num_raster * PI);
    for (int ii = blockIdx.x; ii < num_atom; ii += gridDim.x) {
        int atom1t = Ele[ii];
        vdW_s = vdW[atom1t];
        r = vdW_s + sol_s;
        for (int jj = threadIdx.x; jj < num_raster; jj += blockDim.x) {
            pts[jj] = 1;
            float h = 1.0 - (2.0 * (float)jj + 1.0) / (float)num_raster;
            float p = acos(h);
            float t = L * p; 
            float x = r * sin(p) * cos(t) + coord[3*ii];
            float y = r * sin(p) * sin(t) + coord[3*ii+1];
            float z = r * cos(p) + coord[3*ii+2];
            //if (ii == 0 && jj == 0) printf("Raster: %.3f, %.3f, %.3f  Ref: %.3f, %.3f, %.3f, \n", x, y, z, coord[3*ii], coord[3*ii+1], coord[3*ii+2]);
            for (int kk = 0; kk < close_num[ii]; kk++) {
                int atom2i = close_idx[ii*num_atom2 + kk];
                
                int atom2t = Ele[atom2i];
                float dr2 = (x - coord[3*atom2i]) * (x - coord[3*atom2i]) +
                            (y - coord[3*atom2i+1]) * (y - coord[3*atom2i+1]) +
                            (z - coord[3*atom2i+2]) * (z - coord[3*atom2i+2]); 
                if (dr2 < vdW[atom2t] * vdW[atom2t]) {
                    pts[jj] = 0;
                    //break;
                }
            }
        }
        /*if (ii == 0 && threadIdx.x == 0) {
            for (int jj = 0; jj < 16; jj++) {
                for (int kk = 0; kk < 32; kk++) {
                    printf("%d ", pts[jj*32+kk]);
                }
                printf("\n");
            }
        }*/
    // Sum pts == 1, calc surf area and assign to V[ii]
        for (int stride = num_raster / 2; stride > 0; stride >>= 1) {
            __syncthreads();
            for(int iAccum = threadIdx.x; iAccum < stride; iAccum += blockDim.x) {
                pts[iAccum] += pts[stride + iAccum];
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            V[ii] = (float)pts[0] / (float)num_raster;// * 4.0 * r * r * PI ;
            //if (ii == 0) printf("Sum pts = %d, V = %.3f. \n", pts[0], V[ii]);
        }
    } 
}

__global__ float FF_calc_dummy (float q, int Ele, float *WK, float vdW) {
    

}




/*
__global__ void border_scat (float *coord, int *Ele, float *r2, float *raster, float *V, int num_atom, int num_atom2, int num_raster, int num_raster2) {
    // raster is the rasterized equivolumetric sphere points w.r.t. center atom as N * 3 array.
    // Calculate border scattering
    if (blockIdx.x >= num_atom) return;
    __shared__ float pts[1024];
    __shared__ int close_flag[1749];
    __shared__ float raster_s[3072];
    __shared__ float x_ref, y_ref, z_ref;
    __shared__ float r2_thres;
    if (threadIdx.x == 0) {r2_min = 2.25; r2_max = 9.00;}
    // Marking those atoms that are close
    for (int ii = blockIdx.x; ii < num_atom; ii += gridDim.x) {
        x_ref = coord[3*ii]; y_ref = coord[3*ii+1]; z_ref = coord[3*ii+2];
        for (int jj = threadIdx.x; jj < num_raster; jj += blockDim.x) {
            raster_s[3*jj] = raster[3*jj] + x_ref;
            raster_s[3*jj+1] = raster[3*jj+1] + y_ref;
            raster_s[3*jj+2] = raster[3*jj+2] + z_ref;
        }
        __syncthreads();
        for (int jj = threadIdx.x; jj < num_atom; jj += blockDim.x) {
            close_flag[jj] = 1;
            if (r2[ii*num_atom+jj] >= 36.0) close_flag[jj] = 0;
        }
        __syncthreads();
    // For each raster points
        for (int jj = threadIdx.x; jj < num_raster; jj += blockDim.x) {
            pts[jj] = 1.0;
            if ((raster[jj*3] * raster[jj*3] + raster[jj*3+1] * raster[jj*3+1] + raster[jj*3+2] * raster[jj*3 + 2]) < r2_min) {
                pts[jj] = 0.0;
                continue;
            }
            for (int kk = 0; kk < num_atom; kk ++) {
                if (close_flag[kk] == 1) {
                    float dr = (raster_s[3*jj] - coord[3*kk]) * 
                               (raster_s[3*jj] - coord[3*kk]) + 
                               (raster_s[3*jj+1] - coord[3*kk+1]) * 
                               (raster_s[3*jj+1] - coord[3*kk+1]) + 
                               (raster_s[3*jj+2] - coord[3*kk+2]) * 
                               (raster_s[3*jj+2] - coord[3*kk+2]);
                    if (dr < r2_min) {
                        pts[jj] = 0.0;
                        //break;
                    } 
                    if (dr < r2_max) {
                        if (pts[jj] > 0.0) pts[jj]++;
                    }
                }   
            }
            if (pts[jj] > 0.0) pts[jj] = 1.0 / pts[jj];
        }
        // Calculate volume 
        for (int stride = num_raster2 / 2; stride > 0; stride >>= 1) {
            __syncthreads();
            for(int iAccum = threadIdx.x; iAccum < stride; iAccum += blockDim.x) {
                pts[iAccum] += pts[stride + iAccum];
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            V[ii] = pts[0];
        }
    }
}
*/
/*
__global__ void V_calc (float *V, int num_atom2) {
    // Integrate V
    for (int stride = num_atom2 / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        for(int iAccum = threadIdx.x; iAccum < stride; iAccum += blockDim.x) {
            V[iAccum] += V[stride + iAccum];
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float vol = V[0] * 4.0 / 3.0 * 3.141592653 * 3.0 * 3.0 * 3.0 / 8.0 / 8.0 / 8.0 / 2.0;
        printf("There are %.3f volume elements, which translates to %.3f A^3.\n", V[0], vol);
    }
}
*/
__global__ void scat_calc (float *coord, float *Force, int *Ele, float *WK, float *q_S_ref_dS, float *S_calc, int num_atom, int num_q, int num_ele, float *Aq, float alpha, float k_chi, float sigma2, float *f_ptxc, float *f_ptyc, float *f_ptzc, float *S_calcc, int num_atom2, int num_q2) {
    __shared__ float q_pt, q_WK;
    __shared__ float FF_pt[7]; // num_ele + 1, the last one for water.
    __shared__ float WK_s[66];
    __shared__ float S_calccs[1024];
    __shared__ float f_ptxcs[1024];
    __shared__ float f_ptycs[1024];
    __shared__ float f_ptzcs[1024];
    //float FF_pt[6]; 
    if (blockIdx.x >= num_q) return; // out of q range
    if (threadIdx.x >= num_atom) return; // out of atom numbers (not happening)
   

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
    //unsigned int t1, t2, t3;
    //for (int ii = blockIdx.x * blockDim.x + threadIdx.x; ii < num_q; ii += blockDim.x * gridDim.x) {
    for (int ii = blockIdx.x; ii < num_q; ii += gridDim.x) {
          //       0 - 512          300          512
        q_pt = q_S_ref_dS[ii];
        //if (threadIdx.x == 0) {
        //    printf("q_pt[%d] = %.3f. \n", ii, q_pt);
        //}
        q_WK = q_pt / 4.0 / PI;
        // Put FF coeff to shared memory
        //if (threadIdx.x == 0)
        for (int jj = threadIdx.x; jj < 11 * num_ele; jj +=blockDim.x) {
            WK_s[jj] = WK[jj];
        }
        __syncthreads();
        /*if (blockIdx.x == 0 && threadIdx.x == 0) {
            for (int jj = 0; jj < 66; jj ++) {
                printf("WK %d before is %.3f. \n", jj, WK[jj]);
                printf("WK coeff %d is %.3f. \n", jj, WK_s[jj]);
            }
        }
        __syncthreads();*/
        // Calculate Form factor for this block (or q vector)
        for (int jj = threadIdx.x; jj < num_ele + 1; jj += blockDim.x) {
            if (jj == num_ele) {
                // water
                FF_pt[jj] = WK_s[3*11] * exp(-WK_s[3*11+6] * q_WK * q_WK) + 
                            WK_s[3*11+1] * exp(-WK_s[3*11+7] * q_WK * q_WK) + 
                            WK_s[3*11+2] * exp(-WK_s[3*11+8] * q_WK * q_WK) + 
                            WK_s[3*11+3] * exp(-WK_s[3*11+9] * q_WK * q_WK) + 
                            WK_s[3*11+4] * exp(-WK_s[3*11+10] * q_WK * q_WK) + 
                            WK_s[3*11+5] + 2.0 * (WK_s[5] + 
                            WK_s[0] * exp(-WK_s[6] * q_WK * q_WK) + 
                            WK_s[1] * exp(-WK_s[7] * q_WK * q_WK) + 
                            WK_s[2] * exp(-WK_s[8] * q_WK * q_WK) + 
                            WK_s[3] * exp(-WK_s[9] * q_WK * q_WK) + 
                            WK_s[4] * exp(-WK_s[10] * q_WK * q_WK)) - 
                            vdW[jj] * vdW[jj] * vdW[jj] * PI * 4.0 / 3.0 * 0.334 *
                            exp(-PI * vdW[jj] * vdw[jj] * q_WK * q_WK);;
            } else { 
                // The last part is for excluded volume
                FF_pt[jj] = WK_s[jj*11] * exp(-WK_s[jj*11+6] * q_WK * q_WK) + 
                            WK_s[jj*11+1] * exp(-WK_s[jj*11+7] * q_WK * q_WK) + 
                            WK_s[jj*11+2] * exp(-WK_s[jj*11+8] * q_WK * q_WK) + 
                            WK_s[jj*11+3] * exp(-WK_s[jj*11+9] * q_WK * q_WK) + 
                            WK_s[jj*11+4] * exp(-WK_s[jj*11+10] * q_WK * q_WK) +
                            WK_s[jj*11+5] -
                            C1 * vdW[jj] * vdW[jj] * vdW[jj] * PI * 4.0 / 3.0 * 0.334 *
                            exp(-PI * vdW[jj] * vdw[jj] * q_WK * q_WK);
            }
            //if (ii == 0) printf("FF for elem %d at q = 0 is %.3f.\n", jj, FF_pt[jj]);
        }
        __syncthreads();
        // Calculate scattering for Aq
        //for (int jj = blockIdx.y * blockDim.y + threadIdx.y; jj < num_atom; jj += blockIdx.y * gridDim.y) {
        for (int jj = threadIdx.x; jj < num_atom; jj += blockDim.x) {
              //       0 - 1023          1749            1024
            int idx = jj % blockDim.x;
            // if (jj==1) printf("idx is %d. \n",idx);
            S_calccs[idx] = 0.0; f_ptxcs[idx] = 0.0; f_ptycs[idx] = 0.0; f_ptzcs[idx] = 0.0;
            float atom1x = coord[3*jj+0];
            float atom1y = coord[3*jj+1];
            float atom1z = coord[3*jj+2];
            int atom1t = Ele[jj]; // atom1 element type
            float atom1FF = FF_pt[atom1t]; // atom1 form factor at q // 6 ms
            atom1FF += c2 * V[jj] * FF_pt[num_ele]; // Correction with border layer scattering
            //float atom1FF = FF[ii*num_ele+atom1t]; // atom1 form factor at q
            for (int kk = 0; kk < num_atom; kk++) {
                int atom2t = Ele[kk]; // 6 ms
                if (q_pt == 0.0) {
                    //S_calcc[ii*num_atom2+jj] += atom1FF * FF[ii*num_ele+atom2t];
                    //S_calcc[ii*num_atom2+jj] += atom1FF * FF_pt[atom2t];
                    S_calccs[idx] += atom1FF * FF_pt[atom2t]; // 6.2 ms
                    //*a = 1;
                } else if (kk == jj) {
                    //S_calcc[ii*num_atom2+jj] += atom1FF * FF[ii*num_ele+atom2t];
                    //S_calcc[ii*num_atom2+jj] += atom1FF * FF_pt[atom2t];
                    S_calccs[idx] += atom1FF * FF_pt[atom2t]; // 7.6 ms
                } else {
                    //if (ii==0&&jj==0&&kk==0) t1 = clock();
                    float dx = coord[3*kk+0] - atom1x;
                    float dy = coord[3*kk+1] - atom1y;
                    float dz = coord[3*kk+2] - atom1z; // 7.6 ms
                    float r = sqrt(dx*dx+dy*dy+dz*dz); // 7.6 ms
                    //if (ii==1&&jj==0&&kk==1) printf("Distance btw jj = 0 and kk = 1 is sqrt (%.3f^2 + %.3f^2 + %.3f^2) = %.3f. \n", dx, dy, dz, r);
                    //int r_bin = (int)(q_pt * r / qr_step);
                    float sqr = sin(q_pt * r); // 22 ms
                    float prefac = atom1FF * FF_pt[atom2t] * (cos(q_pt * r) - sqr / q_pt / r) / r / r; //27 ms
                    //float prefac = atom1FF * FF_pt[atom2t] * cxsxdx_table[r_bin] / r / r;
                    //float prefac = atom1FF * FF[ii*num_ele+atom2t] * (cos(q_pt * r) - sin(q_pt * r) / q_pt / r) / r / r;
                    //S_calcc[ii*num_atom2+jj] += atom1FF * FF[ii*num_ele+atom2t] * sin(q_pt * r) / q_pt / r;
                    //S_calcc[ii*num_atom2+jj] += atom1FF * FF_pt[atom2t] * sqr / q_pt / r;
                    S_calccs[idx] += atom1FF * FF_pt[atom2t] * sqr / q_pt / r; // 51 ms
                    //S_calccs[jj] += atom1FF * FF_pt[atom2t] * sx_table[r_bin] / q_pt / r;
                    // f_ptxc[ii*num_atom2+jj] += prefac * dx;
                    // f_ptyc[ii*num_atom2+jj] += prefac * dy;
                    // f_ptzc[ii*num_atom2+jj] += prefac * dz; // 94 ms
                    f_ptxcs[idx] += prefac * dx;
                    f_ptycs[idx] += prefac * dy; 
                    f_ptzcs[idx] += prefac * dz; // 94 -> 90 ms.
                    //if (ii==0&&jj==0&&kk==num_atom-1) t2 = clock();
                }
            }
            //if (ii==0&&jj==1024) t2=clock();
            S_calcc[ii*num_atom2+jj] = S_calccs[idx];
            //if (ii==0&&jj>0&&jj<10) printf("S_calccs[jj = %d] = %f. \n",jj,S_calccs[idx]);
            f_ptxc[ii*num_atom2+jj] = f_ptxcs[idx];
            f_ptyc[ii*num_atom2+jj] = f_ptycs[idx];
            f_ptzc[ii*num_atom2+jj] = f_ptzcs[idx];
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
            Aq[ii] = k_chi / 2.0 / sigma2 * ( q_S_ref_dS[ii+2*num_q] - alpha * (S_calc[ii] - q_S_ref_dS[ii+num_q]));
            //printf("S_calc[%d] = %.3f. \n", ii, S_calc[ii]);
        }
        __syncthreads();
        // Multiply f_pt{x,y,z}c(q) by Aq(q) * 8 * alpha * k_chi / sigma2
        for (int jj = threadIdx.x; jj < num_atom; jj += blockDim.x) {
            f_ptxc[ii * num_atom2 + jj] *= Aq[ii] * 4.0 * alpha;
            f_ptyc[ii * num_atom2 + jj] *= Aq[ii] * 4.0 * alpha;
            f_ptzc[ii * num_atom2 + jj] *= Aq[ii] * 4.0 * alpha;
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
    /*if (blockIdx.x == 0 && threadIdx.x == 0) {
        t3 = clock();
        printf("Elapsed time: %.3f s for scat calc and %.3f s for overall \n", (float)(t2-t1) / CLOCKS_PER_SEC, (float)(t3-t1) /CLOCKS_PER_SEC);
    }*/
}


__global__ void force_calc (float *Force, int num_atom, int num_q, float *f_ptxc, float *f_ptyc, float *f_ptzc, int num_atom2, int num_q2) {
    // Do column tree sum of f_ptxc for f_ptx for every atom, then assign threadIdx.x == 0 (3 * num_atoms) to Force. Force is num_atom * 3. 
    //if (threadIdx.x == 0) printf("blockIdx.x = %d\n", blockIdx.x);
    if (blockIdx.x >= num_atom) return;
    for (int ii = blockIdx.x; ii < num_atom; ii += gridDim.x) {
        //printf("BlockIdx = %d \n", ii);
        for (int stride = num_q2 / 2; stride > 0; stride >>= 1) {
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
            //Force[ii*3    ] = 0.0;// -f_ptxc[ii];
            //Force[ii*3 + 1] = 0.0;//-f_ptyc[ii];
            //Force[ii*3 + 2] = 0.0;//-f_ptzc[ii];
        }
        __syncthreads();
    }
}

/*__global__ void force_proj (float *coord, float *Force, float *rot, float *rot_pt, int *bond_pp, int num_pp, int num_atom, int num_atom2) {
    if (blockIdx.x >= num_pp) return;
    if (threadIdx.x >= num_atom) return;
    for (int ii = blockIdx.x; ii < num_pp; ii += gridDim.x) {
        // For each pp bond
        // Calculate normalized torsional vector
        float cp1 = 0.0;
        float cp2 = 0.0;
        float cp3 = 0.0; // Cross product
        int E1, E2, E3; // Atom index of the pp bond
        E1 = bond_pp[3*ii]; E2 = bond_pp[3*ii+1]; E3 = bond_pp[3*ii+2];
        //if (ii == 0) printf("Elements are %d %d and %d \n", E1, E2, E3);
        cp1 = cross2(coord[3*E2+1]-coord[3*E1+1], coord[3*E2+2]-coord[3*E1+2],
                     coord[3*E3+1]-coord[3*E2+1], coord[3*E3+2]-coord[3*E2+2]);
        cp2 = cross2(coord[3*E2+2]-coord[3*E1+2], coord[3*E2+0]-coord[3*E1+0],
                     coord[3*E3+2]-coord[3*E2+2], coord[3*E3+0]-coord[3*E2+0]);
        cp3 = cross2(coord[3*E2+0]-coord[3*E1+0], coord[3*E2+1]-coord[3*E1+1],
                     coord[3*E3+0]-coord[3*E2+0], coord[3*E3+1]-coord[3*E2+1]);
        float r = sqrt(cp1 * cp1 + cp2 * cp2 + cp3 * cp3);
        cp1 /= r;
        cp2 /= r;
        cp3 /= r;

        for (int jj = threadIdx.x; jj < num_atom; jj += blockDim.x) {
            // For each atom
            if (jj == E1 || jj == E2) {
                continue;
            } else {
                rot_pt[ii*num_atom2+jj] += dot(cp1, cp2, cp3, coord[3*jj], coord[3*jj+1], coord[3*jj+2]);
            }
        }
    

    // Perform summation for rot
        for (int stride = num_atom2 / 2; stride > 0; stride >>= 1) {
            __syncthreads();
            for(int iAccum = threadIdx.x; iAccum < stride; iAccum += blockDim.x) {
                rot_pt[ii * num_atom2 + iAccum] += rot_pt[ii * num_atom2 + stride + iAccum];
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            rot[ii] = rot_pt[ii * num_atom2];
        }
        __syncthreads();
    }   

}
*/
/*
__global__ void pp_assign (float *coord, float *Force, float *rot, int *bond_pp, int num_pp, int num_atom) {
    if (threadIdx.x >= num_atom) return;
    for (int ii = threadIdx.x; ii < num_atom; ii += blockDim.x) {
        Force[3*ii] = 0.0;
        Force[3*ii+1] = 0.0;
        Force[3*ii+2] = 0.0;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int ii = 0; ii < num_atom; ii ++) {
            printf("Force is now %.3f, %.3f, and %.3f. \n", Force[ii], Force[ii+1], Force[ii+2]);
        }
        printf("rot values: \n");
        for (int ii = 0; ii < num_pp; ii++) {
            printf("%.3f \n", rot[ii]*1e3);
        }
    }
    __syncthreads();
    for (int ii = threadIdx.x; ii < num_pp; ii += blockDim.x) {
        float cp1 = 0.0;
        float cp2 = 0.0;
        float cp3 = 0.0; // Cross product
        int E1, E2, E3; // Atom index of the pp bond
        E1 = bond_pp[3*ii]; E2 = bond_pp[3*ii+1]; E3 = bond_pp[3*ii+2];
        //printf("Element 3 is %d. \n", E3);
        cp1 = cross2(coord[3*E2+1]-coord[3*E1+1], coord[3*E2+2]-coord[3*E1+2],
                     coord[3*E3+1]-coord[3*E2+1], coord[3*E3+2]-coord[3*E2+2]);
        cp2 = cross2(coord[3*E2+2]-coord[3*E1+2], coord[3*E2+0]-coord[3*E1+0],
                     coord[3*E3+2]-coord[3*E2+2], coord[3*E3+0]-coord[3*E2+0]);
        cp3 = cross2(coord[3*E2+0]-coord[3*E1+0], coord[3*E2+1]-coord[3*E1+1],
                     coord[3*E3+0]-coord[3*E2+0], coord[3*E3+1]-coord[3*E2+1]);
        float r = sqrt(cp1 * cp1 + cp2 * cp2 + cp3 * cp3);
        cp1 /= r;
        cp2 /= r;
        cp3 /= r;
        //printf("Vector for E%d is (%.3f, %.3f, %.3f)\n", E3, cp1, cp2, cp3);
        Force[3*E3] = -cp1 * rot[ii] * 1e-3;
        Force[3*E3+1] = -cp2 * rot[ii] * 1e-3;
        Force[3*E3+2] = -cp3 * rot[ii] * 1e-3;
    }
}
*/


