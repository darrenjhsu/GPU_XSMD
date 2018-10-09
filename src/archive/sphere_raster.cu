
#include <stdio.h>
#include <math.h>
#define PI 3.1415926535898

int main () {
    float r = 3.0; 
    int shell = 8;
    float dr = r / shell;
    int count = 0;
    float r_iw, b_ij1, b_ij2, b_ijw, a_ijk1, a_ijk2, a_ijkw, w_x, w_y, w_z;
    // b_ij1 = b_ij, b_ij2 = b_i(j-1); same for a_ijk.
    printf("float raster[%d] = {", shell * shell * shell * 2 * 3);
    for (int ii = 1; ii <= shell; ii++) {
        r_iw = (2 * ii - 1) * PI / 2 / shell;
        for (int jj = 1; jj <= ii; jj++) {
            if (jj == 1) {
                w_x = 0.0; w_y = 0.0; w_z = r_iw;
                //if (ii == 1) printf("ii = %d, jj = %d, (%.7f, %.7f, %.7f), core.", ii, jj, w_x, w_y, w_z);
                if (ii == 1) printf("%.7f, %.7f, %.7f, ", w_x, w_y, w_z);
                //if (ii > 1) printf("ii = %d, jj = %d, (%.7f, %.7f, %.7f), cap. ", ii, jj, w_x, w_y, w_z);
                if (ii > 1) printf("%.7f, %.7f, %.7f, ", w_x, w_y, w_z);
                count++;
            } else {
                b_ij1 = acos((3.0 * (ii - jj) * (ii + jj - 1.0)) / (1.0 + 3.0 * ii * (ii - 1.0)));
                b_ij2 = acos((3.0 * (ii - jj + 1.0) * (ii + jj - 2.0)) / (1.0 + 3.0 * ii * (ii - 1.0)));
                b_ijw = (b_ij1 + b_ij2) / 2.0;
            }
            for (int kk = 1; kk <= (jj-1)*6; kk++) {
                a_ijk1 = (2 * kk * PI) / (6 * (jj - 1));
                a_ijk2 = (2 * (kk - 1) * PI) / (6 * (jj - 1));
                a_ijkw = (a_ijk1 + a_ijk2) / 2.0;
                w_x = r_iw * sin(b_ijw) * cos(a_ijkw);
                w_y = r_iw * sin(b_ijw) * sin(a_ijkw);
                w_z = r_iw * cos(b_ijw);
                //printf("ii = %d, jj = %d, kk = %d, b_ijw = %.7f, (%.7f, %.7f, %.7f), common. ", ii, jj, kk, b_ijw, w_x, w_y, w_z);
                printf("%.7f, %.7f, %.7f, ", w_x, w_y, w_z);
                count++;
            }
        }
        for (int jj = 1; jj <= ii; jj++) {
            if (jj == 1) {
                w_x = 0.0; w_y = 0.0; w_z = -r_iw;
                //if (ii == 1) printf("ii = %d, jj = %d, (%.7f, %.7f, %.7f), core.", ii, jj, w_x, w_y, w_z);
                if (ii == 1) printf("%.7f, %.7f, %.7f, ", w_x, w_y, w_z);
                //if (ii > 1) printf("ii = %d, jj = %d, (%.7f, %.7f, %.7f), cap. ", ii, jj, w_x, w_y, w_z);
                if (ii > 1) printf("%.7f, %.7f, %.7f, ", w_x, w_y, w_z);
                count++;
            } else {
                b_ij1 = acos((3.0 * (ii - jj) * (ii + jj - 1.0)) / (1.0 + 3.0 * ii * (ii - 1.0)));
                b_ij2 = acos((3.0 * (ii - jj + 1.0) * (ii + jj - 2.0)) / (1.0 + 3.0 * ii * (ii - 1.0)));
                b_ijw = (b_ij1 + b_ij2) / 2.0;
            }
            for (int kk = 1; kk <= (jj-1)*6; kk++) {
                a_ijk1 = (2 * kk * PI) / (6 * (jj - 1));
                a_ijk2 = (2 * (kk - 1) * PI) / (6 * (jj - 1));
                a_ijkw = (a_ijk1 + a_ijk2) / 2.0;
                w_x = r_iw * sin(b_ijw) * cos(a_ijkw);
                w_y = r_iw * sin(b_ijw) * sin(a_ijkw);
                w_z = -r_iw * cos(b_ijw);
                //printf("ii = %d, jj = %d, kk = %d, b_ijw = %.7f, (%.7f, %.7f, %.7f), common. ", ii, jj, kk, b_ijw, w_x, w_y, w_z);
                printf("%.7f, %.7f, %.7f, ", w_x, w_y, w_z);
                count++;
            }
        }
    }
    //printf("count: %d ", count);
    printf("}; \n");
}
