


#include "env_param.hh"

//int num_q = 667;
//int num_q2 = 1024;
int num_ele = 6;


float k_chi = 1e-7;


int num_raster = 512;
int num_raster2 = 512;

float sol_s = 1.80;
                // H     C     N     O     S    Fe   H2O see Svergun J Appl Crystallogr 1978
float vdW[7] = {1.07, 1.58, 0.84, 1.30, 1.68, 1.24, 1.67};

float c1 = 1.0;

// Choose either FoXS or HyPred
float c2_F = 1.0;  // FoXS style one-for-all c2

// HyPred style every atom type has its own c2
                //  Null         C         N         O         S        Fe       
//float c2[10] = { 0.00000, -1.28015, -1.14564,  1.58676, -0.81264,  0.00000,  
//                 0.11884,  3.29229,  3.59726, -0.24575};
                //    HC        HN        HO        HS  ( up to 8 A )

                //  Null         C         N         O         S        Fe       
float c2_H[10] = { 0.00000, -0.08428, -0.68250,  1.59535,  0.23293,  0.00000,  
                 1.86771,  3.04298,  4.06575,  0.79196};
                //    HC        HN        HO        HS  ( up to R + 3 A )

                //  Null         C         N         O         S        Fe       
//float c2[10] = { 0.00000,  0.02072, -1.01174,  1.50338,  0.50419,  0.00000,  
//                 1.36943,  2.28916,  2.77263,  0.08467};
                //    HC        HN        HO        HS  ( up to our vdW R + 3 A )
float r_m = 1.62;

float offset = 0.2;
