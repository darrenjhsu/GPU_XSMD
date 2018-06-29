



//__global__ void scat_calc (double *coord, double *Force, int *Ele, double *FF, double *q, double *S_ref, double *dS, double *S_calc, int num_atom, int num_q, int num_ele, double *Aq, double alpha, double k_chi, double sigma2, double *f_ptxc, double *f_ptyc, double *f_ptzc, double *S_calcc, int num_atom2, int num_q2);
__global__ void scat_calc (double *coord, double *Force, int *Ele, double *WK, double *q, double *S_ref, double *dS, double *S_calc, int num_atom, int num_q, int num_ele, double *Aq, double alpha, double k_chi, double sigma2, double *f_ptxc, double *f_ptyc, double *f_ptzc, double *S_calcc, int num_atom2, int num_q2);
__global__ void force_calc (double *Force, double *q, int num_atom, int num_q, double *f_ptxc, double *f_ptyc, double *f_ptzc, int num_atom2, int num_q2); 
__global__ void force_proj (double *coord, double *Force, double *rot, double *rot_pt, int *bond_pp, int num_pp, int num_atom, int num_atom2);
__global__ void pp_assign (double *coord, double *Force, double *rot, int *bond_pp, int num_pp, int num_atom);
__device__ double dot (double a1, double a2, double a3, double b1, double b2, double b3);
__device__ double cross2 (double a2, double a3, double b2, double b3);
