#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <string.h>

void generate_matrix(double *A, int n) {
    srand(42);
    for (int i = 0; i < n * n; i++) {
        A[i] = (i % (n + 1) == 0) ? (rand() % 5 + n * 5.0) : (rand() % 5 - 2.0); // Diagonal dominante
    }
}

void lu_decomposition(double *A, double *L, double *U, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            if (j < i) L[j * n + i] = 0;
            else {
                L[j * n + i] = A[j * n + i];
                for (int k = 0; k < i; k++)
                    L[j * n + i] -= L[j * n + k] * U[k * n + i];
            }

            if (j < i) U[i * n + j] = 0;
            else if (j == i) U[i * n + j] = 1;
            else {
                U[i * n + j] = A[i * n + j];
                for (int k = 0; k < i; k++)
                    U[i * n + j] -= L[i * n + k] * U[k * n + j];
                U[i * n + j] /= L[i * n + i];
            }
        }
}

void forward_substitution(double *L, double *B, double *Y, int n) {
    for (int i = 0; i < n; i++) {
        Y[i] = B[i];
        for (int j = 0; j < i; j++)
            Y[i] -= L[i * n + j] * Y[j];
        Y[i] /= L[i * n + i];
    }
}

void backward_substitution(double *U, double *Y, double *X, int n) {
    for (int i = n - 1; i >= 0; i--) {
        X[i] = Y[i];
        for (int j = i + 1; j < n; j++)
            X[i] -= U[i * n + j] * X[j];
    }
}

void multiply(double *A, double *B, double *C, int n) {
    for (int i = 0; i < n * n; i++) C[i] = 0.0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                C[i * n + j] += A[i * n + k] * B[k * n + j];
}

void run_inverse(int N, int rank, int size) {
    MPI_Barrier(MPI_COMM_WORLD);

    char hostname[256];
    gethostname(hostname, 256);

    char info[300];
    snprintf(info, sizeof(info), "Proceso %d de %d ejecut\u00e1ndose en %s\n", rank, size, hostname);

    if (rank != 0) {
        MPI_Send(info, strlen(info) + 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    }

    double *A = NULL, *L = malloc(N * N * sizeof(double)), *U = malloc(N * N * sizeof(double));
    if (rank == 0) {
        A = malloc(N * N * sizeof(double));
        generate_matrix(A, N);
        lu_decomposition(A, L, U, N);
    }

    MPI_Bcast(L, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(U, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int cols_per_proc = N / size;
    int extra = N % size;
    int my_cols = cols_per_proc + (rank < extra ? 1 : 0);
    int start_col = 0;
    for (int i = 0; i < rank; i++)
        start_col += cols_per_proc + (i < extra ? 1 : 0);

    double *Y = malloc(N * sizeof(double));
    double *X = malloc(N * sizeof(double));
    double *A_inv_block = malloc(my_cols * N * sizeof(double));

    double start = MPI_Wtime();

    for (int c = 0; c < my_cols; c++) {
        int col = start_col + c;
        double *e = calloc(N, sizeof(double));
        e[col] = 1.0;

        forward_substitution(L, e, Y, N);
        backward_substitution(U, Y, X, N);

        for (int i = 0; i < N; i++)
            A_inv_block[c * N + i] = X[i];

        free(e);
    }

    int *recvcounts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));
    int offset = 0;
    for (int i = 0; i < size; i++) {
        int cols = cols_per_proc + (i < extra ? 1 : 0);
        recvcounts[i] = cols * N;
        displs[i] = offset;
        offset += recvcounts[i];
    }

    double *A_inv = NULL;
    if (rank == 0) A_inv = malloc(N * N * sizeof(double));

    MPI_Gatherv(A_inv_block, my_cols * N, MPI_DOUBLE,
                A_inv, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double end = MPI_Wtime();

    if (rank == 0) {
        printf("\n--- Inversi\u00f3n de matriz %dx%d con %d procesos ---\n", N, N, size);
        printf("%s", info);
        for (int i = 1; i < size; i++) {
            MPI_Recv(info, 300, MPI_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("%s", info);
        }

        double *product = malloc(N * N * sizeof(double));
        multiply(A, A_inv, product, N);

        printf("\nPrimeros 5x5 elementos de A * A^-1:\n");
        for (int i = 0; i < 5 && i < N; i++) {
            for (int j = 0; j < 5 && j < N; j++) {
                printf("%8.3f ", product[i * N + j]);
            }
            printf("\n");
        }

        double error = 0.0, max_error = 0.0;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                double diff = product[i * N + j] - expected;
                error += diff * diff;
                if (fabs(diff) > max_error) max_error = fabs(diff);
            }
        error = sqrt(error);

        printf("\nError cuadr\u00e1tico respecto a la identidad: %.6e\n", error);
        printf("Error m\u00e1ximo absoluto: %.6e\n", max_error);
        printf("Tiempo de ejecuci\u00f3n: %.6f segundos\n", end - start);

        free(product);
        free(A);
        free(A_inv);
    }

    free(L); free(U); free(A_inv_block); free(Y); free(X); free(recvcounts); free(displs);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Fijo para STRONG scalability: matriz 2000x2000
    int N = 2000;

    run_inverse(N, rank, size);

    MPI_Finalize();
    return 0;
}