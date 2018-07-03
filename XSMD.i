%module XSMD
%{
    #include "XSMD.hh"
    #include "param.hh"
    #include "WaasKirf.hh"
%}

// SWIG helper functions for arrays
%inline %{
/* Create an array */
float *float_array(int size) {
    return (float *)malloc(size*sizeof(float));
}
/* Get a value from an array */
float float_get(float *a, int index) {
    return a[index];
}
/* Set a value in the array */
float float_set(float *a, int index, float value) {
    return (a[index] = value);
}

void float_destroy(float *a) { 
    free(a);
}
%}
%include "XSMD.hh"
%include "param.hh"
%include "WaasKirf.hh"
