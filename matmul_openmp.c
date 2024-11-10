#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void transpose(double *A, double *B, int n) {
    int i,j;
    for(i=0; i<n; i++) {
        for(j=0; j<n; j++) {
            B[j*n+i] = A[i*n+j];
        }
    }
}

void mm(double *A, double *B, double *C, int n) 
{   
    int i, j, k;
    for (i = 0; i < n; i++) { 
        for (j = 0; j < n; j++) {
            double dot  = 0;
            for (k = 0; k < n; k++) {
                dot += A[i*n+k]*B[k*n+j];
            } 
            C[i*n+j ] = dot;
        }
    }
}

void mm_omp(double *A, double *B, double *C, int n) 
{   
    #pragma omp parallel
    {
        int i, j, k;
        #pragma omp for
        for (i = 0; i < n; i++) { 
            for (j = 0; j < n; j++) {
                double dot  = 0;
                for (k = 0; k < n; k++) {
                    dot += A[i*n+k]*B[k*n+j];
                } 
                C[i*n+j ] = dot;
            }
        }

    }
}

void mmT(double *A, double *B, double *C, int n) 
{   
    int i, j, k;
    double *B2;
    B2 = (double*)malloc(sizeof(double)*n*n);
    transpose(B,B2, n);
    for (i = 0; i < n; i++) { 
        for (j = 0; j < n; j++) {
            double dot  = 0;
            for (k = 0; k < n; k++) {
                dot += A[i*n+k]*B2[j*n+k];
            } 
            C[i*n+j ] = dot;
        }
    }
    free(B2);
}

void mmT_omp(double *A, double *B, double *C, int n) 
{   
    double *B2;
    B2 = (double*)malloc(sizeof(double)*n*n);
    transpose(B,B2, n);
    #pragma omp parallel
    {
        int i, j, k;
        #pragma omp for
        for (i = 0; i < n; i++) { 
            for (j = 0; j < n; j++) {
                double dot  = 0;
                for (k = 0; k < n; k++) {
                    dot += A[i*n+k]*B2[j*n+k];
                } 
                C[i*n+j ] = dot;
            }
        }

    }
    free(B2);
}

int main() {
    int i, n;
    double *A, *B, *C, dtime;
    int sizes[] = {256, 512, 1024}; // Different sizes
    int threads[] = {1, 2, 4, 8};   // Different thread number

    for (int s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++) {
        n = sizes[s];
        A = (double*)malloc(sizeof(double) * n * n);
        B = (double*)malloc(sizeof(double) * n * n);
        C = (double*)malloc(sizeof(double) * n * n);
        for (i = 0; i < n * n; i++) { A[i] = rand() / (double)RAND_MAX; B[i] = rand() / (double)RAND_MAX; }

        printf("Matrix Size: %dx%d\n", n, n);

        for (int t = 0; t < sizeof(threads) / sizeof(threads[0]); t++) {
            omp_set_num_threads(threads[t]);
            printf("Threads: %d\n", threads[t]);

            dtime = omp_get_wtime();
            mm(A, B, C, n);
            dtime = omp_get_wtime() - dtime;
            printf("Original Serial: %f seconds\n", dtime);

            dtime = omp_get_wtime();
            mm_omp(A, B, C, n);
            dtime = omp_get_wtime() - dtime;
            printf("Original Parallel: %f seconds\n", dtime);

            dtime = omp_get_wtime();
            mmT(A, B, C, n);
            dtime = omp_get_wtime() - dtime;
            printf("Transposed Serial: %f seconds\n", dtime);

            dtime = omp_get_wtime();
            mmT_omp(A, B, C, n);
            dtime = omp_get_wtime() - dtime;
            printf("Transposed Parallel: %f seconds\n", dtime);

            printf("\n");
        }
    }

    return 0;
}
