
#include <stdio.h>
#include <math.h>
#define PI 3.1415926535898

int main () {
    float r = 3.0; 
    int shell = 10;
    float dr = r / shell;
    int count = 0;
    float r_ii, b_ij1, b_ij2, b_ijw, a_ijk1, a_ijk2, a_ijkw, w_x, w_y, w_z;
    // b_ij1 = b_ij, b_ij2 = b_i(j-1); same for a_ijk.
    for (int ii = 1; ii <= shell; ii++) {
        r_iw = (2 * ii - 1) * PI / 2 / ii;
        for (int jj = 1; jj <= ii; jj++) {
            if (jj == 1) {
                w_x = 0.0; w_y = 0.0; w_z = r_iw;
                if (ii == 1) printf("ii = %d, jj = %d, core.\n", ii, jj);
                if (ii > 1) printf("ii = %d, jj = %d, cap. \n", ii, jj);
                count++;
            }
            for (int kk = 1; kk <= (jj-1)*6; kk++) {
                b_ij1 = acos((3 * (ii - jj) * (ii + jj - 1)) / (1 + 3 * i * (i - 1)));
                b_ij2 = acos((3 * (ii - jj + 1) * (ii + jj - 2)) / (1 + 3 * i * (i - 1)));
                printf("ii = %d, jj = %d, kk = %d, common. \n", ii, jj, kk);
                count++;
            }
        }
        for (int jj = 1; jj <= ii; jj++) {
            if (jj == 1) {
                if (ii == 1) printf("ii = %d, jj = %d, core.\n", ii, -jj);
                if (ii > 1) printf("ii = %d, jj = %d, cap. \n", ii, -jj);
                count++;
            }
            for (int kk = 1; kk <= (jj-1)*6; kk++) {
                printf("ii = %d, jj = %d, kk = %d, common. \n", ii, -jj, kk);
                count++;
            }
        }
    }
    printf("count: %d \n", count);
}
