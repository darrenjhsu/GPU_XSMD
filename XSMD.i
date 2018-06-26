%module XSMD
%{
    #include "XSMD.hh"
    #include "param.hh"
%}

// SWIG helper functions for arrays
%inline %{
/* Create an array */
double *double_array(int size) {
    return (double *) malloc(size*sizeof(double));
}
/* Get a value from an array */
double double_get(double *a, int index) {
    return a[index];
}
/* Set a value in the array */
double double_set(double *a, int index, double value) {
    return (a[index] = value);
}

void double_destroy(double *a) { 
    free(a);
}
%}
%include "XSMD.hh"
%include "param.hh"

