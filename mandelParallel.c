#include <stdio.h>
#include <time.h>
#include <omp.h>  // OpenMP library
#include <stdlib.h>

#define WIDTH 640
#define HEIGHT 480
#define MAX_ITER 255

struct complex {
    double real;
    double imag;
};

int cal_pixel(struct complex c) {
    double z_real = 0;
    double z_imag = 0;
    double z_real2, z_imag2, lengthsq;
    int iter = 0;

    do {
        z_real2 = z_real * z_real;
        z_imag2 = z_imag * z_imag;
        z_imag = 2 * z_real * z_imag + c.imag;
        z_real = z_real2 - z_imag2 + c.real;
        lengthsq = z_real2 + z_imag2;
        iter++;
    } while ((iter < MAX_ITER) && (lengthsq < 4.0));

    return iter;
}

void save_pgm(const char *filename, int image[HEIGHT][WIDTH]) {
    FILE *pgmimg;
    pgmimg = fopen(filename, "wb");
    fprintf(pgmimg, "P2\n%d %d\n255\n", WIDTH, HEIGHT);

    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            fprintf(pgmimg, "%d ", image[i][j]);
        }
        fprintf(pgmimg, "\n");
    }
    fclose(pgmimg);
}

int main() {
    int (*image)[WIDTH] = malloc(HEIGHT * WIDTH * sizeof(int));
    if (image == NULL) {
        perror("Failed to allocate memory");
        return 1;
    }

    double AVG = 0;
    int N = 10;  // Number of trials
    double total_time[N];
    struct complex c;

    // This represents the number of threads
    omp_set_num_threads(8);

    for (int k = 0; k < N; k++) {
        clock_t start_time = clock();  // Start measuring time

        // Parallelizing the main loop using OpenMP
        // Parallelizing row-wise with OpenMP, each thread processes one row at a time
        int i,j;
        #pragma omp parallel for private(c) schedule(dynamic, 1) // Dynamic scheduling allows threads to pick up rows as they complete, balancing the load
        for (i = 0; i < HEIGHT; i++) {
            for (j = 0; j < WIDTH; j++) {
                c.real = (j - WIDTH / 2.0) * 4.0 / WIDTH;
                c.imag = (i - HEIGHT / 2.0) * 4.0 / HEIGHT;
                image[i][j] = cal_pixel(c);
            }
        }

        clock_t end_time = clock();  // End measuring time
        total_time[k] = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
        printf("Execution time of trial [%d]: %f seconds\n", k+1, total_time[k]);
        AVG += total_time[k];
    }

    save_pgm("mandelbrot_parallel.pgm", image);
    printf("The average execution time of 10 trials is: %f ms\n", AVG / N * 1000);

    free(image);
    return 0;
}
//gcc -fopenmp -o mandel mandelParallel.c