
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
            //if (r2t < 29.0) close_flag[ii*num_atom2+jj] = 1; // roughly 2 A vdW radii + 1.4 A water vdW
            if (r2t < 34.0) close_flag[ii*num_atom2+jj] = 1; // roughly 2 A + 2 A vdW + 2 * 1.8 A probe
            if (ii == jj) close_flag[ii*num_atom2+jj] = 0;
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

__global__ void surf_calc (float *coord, int *Ele, float *r2, int *close_num, int *close_idx, float *vdW, int num_atom, int num_atom2, int num_raster, float sol_s, float *V, float *surf, float *surf_grad, float offset) {
//__global__ void surf_calc (float *coord, int *Ele, float *r2, int *close_num, int *close_idx, float *vdW, int num_atom, int num_atom2, int num_raster, float sol_s, float *V, float *surf) {
    // num_raster should be a number of 2^n. 
    // sol_s is solvent radius (default = 1.8 A)
    // surf_grad is num_atom of gradient vectors for each atom
    // offset is the h in calculation of gradient = ( f(x+h) - f(x-h) ) / 2 h
    __shared__ float vdW_s; // vdW radius of the center atom
    __shared__ int pts[512]; // All spherical raster points
    __shared__ int ptspx[512], ptsmx[512], ptspy[512], ptsmy[512], ptspz[512], ptsmz[512];
                   // plus x   minus x ... for gradient
    __shared__ float L, r;
    
    if (blockIdx.x >= num_atom) return;
    L = sqrt(num_raster * PI);
    for (int ii = blockIdx.x; ii < num_atom; ii += gridDim.x) {
        int atom1t = Ele[ii];
        vdW_s = vdW[atom1t];
        r = vdW_s + sol_s;
        for (int jj = threadIdx.x; jj < num_raster; jj += blockDim.x) {
            pts[jj] = 1; 
           
            ptspx[jj] = 1; ptsmx[jj] = 1; 
            ptspy[jj] = 1; ptsmy[jj] = 1;
            ptspz[jj] = 1; ptsmz[jj] = 1;
            
            float h = 1.0 - (2.0 * (float)jj + 1.0) / (float)num_raster;
            float p = acos(h);
            float t = L * p; 
            float xu = sin(p) * cos(t);
            float yu = sin(p) * sin(t);
            float zu = cos(p);
            // vdW points
            float x = vdW_s * xu + coord[3*ii];
            float y = vdW_s * yu + coord[3*ii+1];
            float z = vdW_s * zu + coord[3*ii+2];
            // Solvent center
            float x2 = r * xu + coord[3*ii];
            float y2 = r * yu + coord[3*ii+1];
            float z2 = r * zu + coord[3*ii+2];
            //if (ii == 0 && jj == 0) printf("Raster: %.3f, %.3f, %.3f  Ref: %.3f, %.3f, %.3f, \n", x, y, z, coord[3*ii], coord[3*ii+1], coord[3*ii+2]);
            for (int kk = 0; kk < close_num[ii]; kk++) {
                int atom2i = close_idx[ii * num_atom2 + kk];
                int atom2t = Ele[atom2i];
                float dx = (x - coord[3*atom2i]);
                float dy = (y - coord[3*atom2i+1]);
                float dz = (z - coord[3*atom2i+2]);
                float dx2 = (x2 - coord[3*atom2i]);
                float dy2 = (y2 - coord[3*atom2i+1]);
                float dz2 = (z2 - coord[3*atom2i+2]);
                float dr2 = dx * dx + dy * dy + dz * dz; 
                float dr22 = dx2 * dx2 + dy2 * dy2 + dz2 * dz2;
                /*(float dr2 = (x - coord[3*atom2i]) * (x - coord[3*atom2i]) + 
                            (y - coord[3*atom2i+1]) * (y - coord[3*atom2i+1]) + 
                            (z - coord[3*atom2i+2]) * (z - coord[3*atom2i+2]);
                float dr22= (x2 - coord[3*atom2i]) * (x2 - coord[3*atom2i]) + 
                            (y2 - coord[3*atom2i+1]) * (y2 - coord[3*atom2i+1]) + 
                            (z2 - coord[3*atom2i+2]) * (z2 - coord[3*atom2i+2]);*/
                // vdW points must not cross into other atom
                if (dr2 < vdW[atom2t] * vdW[atom2t]) pts[jj] = 0;
                // solvent center has to be far enough
                if (dr22 < (vdW[atom2t]+sol_s) * (vdW[atom2t]+sol_s)) pts[jj] = 0;
                
                // Plus x
                dr2 =  (dx + offset)  * (dx + offset) + dy * dy + dz * dz;
                dr22 = (dx2 + offset) * (dx2 + offset) + dy2 * dy2 + dz2 * dz2;
                if (dr2 < vdW[atom2t] * vdW[atom2t]) ptspx[jj] = 0;
                if (dr22 < (vdW[atom2t]+sol_s) * (vdW[atom2t]+sol_s)) ptspx[jj] = 0;
                // Minus x
                dr2 =  (dx - offset)  * (dx - offset) + dy * dy + dz * dz;
                dr22 = (dx2 - offset) * (dx2 - offset) + dy2 * dy2 + dz2 * dz2;
                if (dr2 < vdW[atom2t] * vdW[atom2t]) ptsmx[jj] = 0;
                if (dr22 < (vdW[atom2t]+sol_s) * (vdW[atom2t]+sol_s)) ptsmx[jj] = 0;
                // Plus y
                dr2 =  dx * dx   + (dy + offset)  * (dy + offset) + dz * dz; 
                dr22 = dx2 * dx2 + (dy2 + offset) * (dy2 + offset) + dz2 * dz2;
                if (dr2 < vdW[atom2t] * vdW[atom2t]) ptspy[jj] = 0;
                if (dr22 < (vdW[atom2t]+sol_s) * (vdW[atom2t]+sol_s)) ptspy[jj] = 0;
                // Minus y
                dr2 =  dx * dx   + (dy - offset)  * (dy - offset) + dz * dz; 
                dr22 = dx2 * dx2 + (dy2 - offset) * (dy2 - offset) + dz2 * dz2;
                if (dr2 < vdW[atom2t] * vdW[atom2t]) ptsmy[jj] = 0;
                if (dr22 < (vdW[atom2t]+sol_s) * (vdW[atom2t]+sol_s)) ptsmy[jj] = 0;
                // Plus z
                dr2 =  dx * dx + dy * dy + (dz + offset) * (dz + offset); 
                dr22 = dx2 * dx2 + dy2 * dy2 + (dz2 + offset) * (dz2 + offset);
                if (dr2 < vdW[atom2t] * vdW[atom2t]) ptspz[jj] = 0;
                if (dr22 < (vdW[atom2t]+sol_s) * (vdW[atom2t]+sol_s)) ptspz[jj] = 0;
                // Minus z
                dr2 =  dx * dx + dy * dy + (dz - offset) * (dz - offset); 
                dr22 = dx2 * dx2 + dy2 * dy2 + (dz2 - offset) * (dz2 - offset);
                if (dr2 < vdW[atom2t] * vdW[atom2t]) ptsmz[jj] = 0;
                if (dr22 < (vdW[atom2t]+sol_s) * (vdW[atom2t]+sol_s)) ptsmz[jj] = 0;
                
            }
            /*if (pts[jj] == 1) {
                surf[3*ii*num_raster+jj*3  ] = x;
                surf[3*ii*num_raster+jj*3+1] = y;
                surf[3*ii*num_raster+jj*3+2] = z;
            }*/
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
                
                ptspx[iAccum] += ptspx[stride + iAccum];
                ptsmx[iAccum] += ptsmx[stride + iAccum];
                ptspy[iAccum] += ptspy[stride + iAccum];
                ptsmy[iAccum] += ptsmy[stride + iAccum];
                ptspz[iAccum] += ptspz[stride + iAccum];
                ptsmz[iAccum] += ptsmz[stride + iAccum];
                
                
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            V[ii] = (float)pts[0]/(float)num_raster;// * 4.0 * r * r * PI ;
            
            surf_grad[3*ii  ] = (float)(ptspx[0] - ptsmx[0]) / 2.0 / offset / (float)num_raster;
            surf_grad[3*ii+1] = (float)(ptspy[0] - ptsmy[0]) / 2.0 / offset / (float)num_raster;
            surf_grad[3*ii+2] = (float)(ptspz[0] - ptsmz[0]) / 2.0 / offset / (float)num_raster;
            
            //if (ii == 0) printf("Sum pts = %d, V = %.3f. \n", pts[0], V[ii]);
        }
    }
}

__global__ void sum_V (float *V, int num_atom, int num_atom2, int *Ele, float *vdW) {
    __shared__ float V_s[2048];
    for (int ii = threadIdx.x; ii < num_atom2; ii += blockDim.x) {
        if (ii < num_atom) {
            int atomi = Ele[ii];
            V_s[ii] = V[ii] * 4.0 * PI * vdW[atomi] * vdW[atomi];
        } else {
            V_s[ii] = 0.0;
        }
    }
    for (int stride = num_atom2 / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        for(int iAccum = threadIdx.x; iAccum < stride; iAccum += blockDim.x) {
            V_s[iAccum] += V_s[stride + iAccum];
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) printf("Convex contact area = %.3f A^2.\n", V_s[0]);
    /*for (int ii = threadIdx.x; ii < num_atom2; ii += blockDim.x) {
        V[ii] = V[ii] / V_s[0];
    } */
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

__global__ void FF_calc (float *q_S_ref_dS, float *WK, float *vdW, int num_q, int num_ele, float c1, float r_m, float *FF_table) {
    __shared__ float q_pt, q_WK, C1, expC1;
    __shared__ float FF_pt[7]; // num_ele + 1, the last one for water.
    __shared__ float vdW_s[7];
    __shared__ float WK_s[66]; 
    __shared__ float C1_PI_43_rho;
    if (blockIdx.x >= num_q) return; // out of q range
    for (int ii = blockIdx.x; ii < num_q; ii += gridDim.x) {
        q_pt = q_S_ref_dS[ii];
        q_WK = q_pt / 4.0 / PI;
        // FoXS C1 term
        expC1 = -powf(4.0 * PI / 3.0, 1.5) * q_WK * q_WK * r_m * r_m * (c1 * c1 - 1.0) / 4.0 / PI;
        C1 = powf(c1,3) * exp(expC1);
        C1_PI_43_rho = C1 * PI * 4.0 / 3.0 * 0.334;
        for (int jj = threadIdx.x; jj < 11 * num_ele; jj += blockDim.x) {
            WK_s[jj] = WK[jj];
        }
        __syncthreads();
        // Put FF coeff to shared memory
        //if (threadIdx.x == 0)
        /*if (blockIdx.x == 0 && threadIdx.x == 0) {
            for (int jj = 0; jj < 66; jj ++) {
                printf("WK %d before is %.3f. \n", jj, WK[jj]);
                printf("WK coeff %d is %.3f. \n", jj, WK_s[jj]);
            }
        }
        __syncthreads();*/
        // Calculate Form factor for this block (or q vector)
        for (int jj = threadIdx.x; jj < num_ele + 1; jj += blockDim.x) {
            vdW_s[jj] = vdW[jj];
            if (jj == num_ele) {
                // water
                FF_pt[jj] = WK_s[3*11+5];
                FF_pt[jj] += 2.0 * WK_s[5];
                FF_pt[jj] -= C1_PI_43_rho * powf(vdW_s[jj],3.0) * exp(-PI * vdW_s[jj] * vdW_s[jj] * q_WK * q_WK);
                for (int kk = 0; kk < 5; kk ++) {
                    FF_pt[jj] += WK_s[3*11+kk] * exp(-WK_s[3*11+kk+6] * q_WK * q_WK);
                    FF_pt[jj] += WK_s[kk] * exp(-WK_s[kk+6] * q_WK * q_WK);
                    FF_pt[jj] += WK_s[kk] * exp(-WK_s[kk+6] * q_WK * q_WK);
                }
            } else { 
                FF_pt[jj] = WK_s[jj*11+5];
                // The part is for excluded volume
                FF_pt[jj] -= C1_PI_43_rho * powf(vdW_s[jj],3.0) * exp(-PI * vdW_s[jj] * vdW_s[jj] * q_WK * q_WK);
                for (int kk = 0; kk < 5; kk++) {
                    FF_pt[jj] += WK_s[jj*11+kk] * exp(-WK_s[jj*11+kk+6] * q_WK * q_WK); 
                }
            }
            //if (ii == 0) printf("FF for elem %d at q = 0 is %.3f.\n", jj, FF_pt[jj]);
        FF_table[ii*(num_ele+1)+jj] = FF_pt[jj];
        }
    }
}

__global__ void __launch_bounds__(1024,2) 
    scat_calc (float *coord,  float *Force,   int *Ele,      float *WK,     float *q_S_ref_dS, 
               float *S_calc, int num_atom,   int num_q,     int num_ele,   float *Aq, 
               float alpha,   float k_chi,    float sigma2,  float *f_ptxc, float *f_ptyc, 
               float *f_ptzc, float *S_calcc, int num_atom2, int num_q2,    float *vdW, 
               float c2,       float *V,      float r_m,     float *FF_table,
               float *surf_grad) {
    __shared__ float q_pt;
    __shared__ float FF_pt[7]; // num_ele + 1, the last one for water.
    __shared__ float S_calccs[1024];
    __shared__ float f_ptxcs[1024];
    __shared__ float f_ptycs[1024];
    __shared__ float f_ptzcs[1024];
    __shared__ float atomFF[1749];

    //if (blockIdx.x >= num_q) return; // out of q range
    //if (threadIdx.x >= num_atom) return; // out of atom numbers (not happening)
   

    for (int ii = blockIdx.x; ii < num_q; ii += gridDim.x) {
        //         0 - 301          301          301
        q_pt = q_S_ref_dS[ii];

        // Get form factor for this block (or q vector)
        for (int jj = threadIdx.x; jj < num_ele + 1; jj += blockDim.x) {
            FF_pt[jj] = FF_table[ii*(num_ele+1)+jj];
        }
        __syncthreads();
        float hydration = c2 * FF_pt[num_ele];
        // Calculate atomic form factor for this q
        for (int jj = threadIdx.x; jj < num_atom; jj += blockDim.x) {
            int atomt = Ele[jj];
            atomFF[jj] = FF_pt[atomt];
            atomFF[jj] += hydration * V[jj];
        }
        __syncthreads();
        // Calculate scattering for Aq
        for (int jj = threadIdx.x; jj < num_atom; jj += blockDim.x) {
              //       0 - 1023          1749            1024
            int idx = jj % blockDim.x;
            S_calccs[idx] = 0.0; f_ptxcs[idx] = 0.0; f_ptycs[idx] = 0.0; f_ptzcs[idx] = 0.0;
            float atom1x = coord[3*jj+0];
            float atom1y = coord[3*jj+1];
            float atom1z = coord[3*jj+2];
            //int atom1t = Ele[jj]; // atom1 element type
            // float atom1FF = FF_pt[atom1t]; // atom1 form factor at q // 6 ms
            /*float atom1FF = FF_pt[atom1t]; // atom1 form factor at q // 6 ms
            atom1FF += c2 * V[jj] * FF_pt[num_ele]; // Correction with border layer scattering
            __syncthreads();*/
            //float atom1FF = FF[ii*num_ele+atom1t]; // atom1 form factor at q
            for (int kk = 0; kk < num_atom; kk++) {
                //int atom2t = Ele[kk]; // 6 ms
                /*float atom2FF = FF_pt[atom2t];
                atom2FF += c2 * V[kk] * FF_pt[num_ele];*/
                float FF_kj = atomFF[jj] * atomFF[kk];
                if (q_pt == 0.0 || kk == jj) {
                /*    //S_calcc[ii*num_atom2+jj] += atom1FF * FF[ii*num_ele+atom2t];
                    //S_calcc[ii*num_atom2+jj] += atom1FF * FF_pt[atom2t];
                    //S_calccs[idx] += atom1FF * atom2FF; // 6.2 ms
                    S_calccs[idx] += FF_kj; 
                    //*a = 1;
                } else if (kk == jj) {
                    //S_calcc[ii*num_atom2+jj] += atom1FF * FF[ii*num_ele+atom2t];
                    //S_calcc[ii*num_atom2+jj] += atom1FF * FF_pt[atom2t];
                    //S_calccs[idx] += atom1FF * atom2FF; // 7.6 ms*/
                    S_calccs[idx] += FF_kj;
                } else {
                    //if (ii==0&&jj==0&&kk==0) t1 = clock();
                    float dx = coord[3*kk+0] - atom1x;
                    float dy = coord[3*kk+1] - atom1y;
                    float dz = coord[3*kk+2] - atom1z; // 7.6 ms
                    // float r = sqrt(r2[jj*num_atom+kk]); // 7.6 ms
                    float r = sqrt(dx*dx+dy*dy+dz*dz);
                    float qr = q_pt * r; 
                    float sqr = sin(qr) / qr; // 22 ms
                    float dsqr = cos(qr) - sqr;
                    //float prefac = 2.0 * atom1FF * atom2FF * (cos(q_pt * r) - sqr / q_pt / r) / r / r; //27 ms
                    float prefac = FF_kj * dsqr / r / r; //27 ms
                    prefac += prefac;
                    float gradient = surf_grad[3*kk] * dx;
                    gradient += surf_grad[3*kk+1] * dy;
                    gradient += surf_grad[3*kk+2] * dz;
                    //gradient /= r;
                    gradient *= sqr * atomFF[jj];
                    //gradient *= 0.1;
                    prefac += hydration * gradient;
                    //float prefac = atom1FF * FF_pt[atom2t] * cxsxdx_table[r_bin] / r / r;
                    //float prefac = atom1FF * FF[ii*num_ele+atom2t] * (cos(q_pt * r) - sin(q_pt * r) / q_pt / r) / r / r;
                    //S_calcc[ii*num_atom2+jj] += atom1FF * FF[ii*num_ele+atom2t] * sin(q_pt * r) / q_pt / r;
                    //S_calcc[ii*num_atom2+jj] += atom1FF * FF_pt[atom2t] * sqr / q_pt / r;
                    //S_calccs[idx] += atom1FF * atom2FF * sqr / q_pt / r; // 51 ms
                    S_calccs[idx] += FF_kj * sqr; // 51 ms
                    //S_calccs[jj] += atom1FF * FF_pt[atom2t] * sx_table[r_bin] / q_pt / r;
                    f_ptxcs[idx] += prefac * dx;
                    f_ptycs[idx] += prefac * dy;
                    f_ptzcs[idx] += prefac * dz; // 94 ms
                    // f_ptxcs[idx] += prefac * dx[jj*num_atom+kk];
                    // f_ptycs[idx] += prefac * dy[jj*num_atom+kk]; 
                    // f_ptzcs[idx] += prefac * dz[jj*num_atom+kk]; // 94 -> 90 ms.
                    //if (ii==0&&jj==0&&kk==num_atom-1) t2 = clock();
                }
            }
            //printf("At atom %d we have these values: \n",jj);
            /*if (jj == 737) {
                printf("%d, q = %.3f: %.8f %.8f %.8f\n", 
                jj, q_S_ref_dS[ii],
                //f_ptxc[ii * num_atom2 + jj], 
                //f_ptyc[ii * num_atom2 + jj], 
                //f_ptzc[ii * num_atom2 + jj]);
                f_ptxcs[idx], 
                f_ptycs[idx], 
                f_ptzcs[idx]);
            }*/ 
            //if (ii==0&&jj==1024) t2=clock();
            S_calcc[ii*num_atom2+jj] = S_calccs[idx];
            //if (ii==0&&jj>0&&jj<10) printf("S_calccs[jj = %d] = %f. \n",jj,S_calccs[idx]);
            f_ptxc[ii*num_atom2+jj] = f_ptxcs[idx];
            f_ptyc[ii*num_atom2+jj] = f_ptycs[idx];
            f_ptzc[ii*num_atom2+jj] = f_ptzcs[idx];
            /*if (jj == 737) {
                printf("%d, inner q = %.3f: %.8f %.8f %.8f\n", 
                jj, q_S_ref_dS[ii],
                f_ptxc[ii * num_atom2 + jj], 
                f_ptyc[ii * num_atom2 + jj], 
                f_ptzc[ii * num_atom2 + jj]);
            } */
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
        
        S_calc[ii] = S_calcc[ii * num_atom2];
        __syncthreads();
        Aq[ii] = k_chi / 2.0 / sigma2 * ( q_S_ref_dS[ii+2*num_q] - alpha * (S_calc[ii] - q_S_ref_dS[ii+num_q]));
        __syncthreads();
        // Multiply f_pt{x,y,z}c(q) by Aq(q) * 4 * alpha
        for (int jj = threadIdx.x; jj < num_atom; jj += blockDim.x) {
            f_ptxc[ii * num_atom2 + jj] *= Aq[ii] * 4.0 * alpha;
            f_ptyc[ii * num_atom2 + jj] *= Aq[ii] * 4.0 * alpha;
            f_ptzc[ii * num_atom2 + jj] *= Aq[ii] * 4.0 * alpha;
        }
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


__global__ void force_calc (float *Force, int num_atom, int num_q, float *f_ptxc, float *f_ptyc, float *f_ptzc, int num_atom2, int num_q2, int *Ele) {
    // Do column tree sum of f_ptxc for f_ptx for every atom, then assign threadIdx.x == 0 (3 * num_atoms) to Force. Force is num_atom * 3. 
    //if (threadIdx.x == 0) printf("blockIdx.x = %d\n", blockIdx.x);
    if (blockIdx.x >= num_atom) return;
    for (int ii = blockIdx.x; ii < num_atom; ii += gridDim.x) {
        //printf("BlockIdx = %d \n", ii);
        /*if (ii == 725 && threadIdx.x == 0) {
            printf("At atom %d we have these values: \n",ii);
            for (int jj = 0; jj < num_q2; jj ++) {
                printf("%.8f %.8f %.8f\n", f_ptxc[ii + jj * num_atom2], 
                                           f_ptyc[ii + jj * num_atom2], 
                                           f_ptzc[ii + jj * num_atom2]);
            }
        }
        */
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
            if (Ele[ii]) {
                Force[ii*3    ] = -f_ptxc[ii];
                Force[ii*3 + 1] = -f_ptyc[ii];
                Force[ii*3 + 2] = -f_ptzc[ii];
            }
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


