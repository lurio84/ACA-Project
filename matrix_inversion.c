#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define N 4  // Tamaño de la matriz cuadrada (puedes cambiar esto)

void print_matrix(const char* label, double* M, int n) {
    printf("%s\n", label);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%8.4f ", M[i * n + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void generate_matrix(double* A, int n) {
    srand(42);
    for (int i = 0; i < n * n; i++) {
        A[i] = (rand() % 5) + 1;
    }
}

// LU Descomposición (Doolittle, sin pivoteo, secuencial en rank 0)
void lu_decomposition(double* A, double* L, double* U, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (j < i)
                L[j * n + i] = 0;
            else {
                L[j * n + i] = A[j * n + i];
                for (int k = 0; k < i; k++) {
                    L[j * n + i] -= L[j * n + k] * U[k * n + i];
                }
            }
        }

        for (int j = 0; j < n; j++) {
            if (j < i)
                U[i * n + j] = 0;
            else if (j == i)
                U[i * n + j] = 1;
            else {
                U[i * n + j] = A[i * n + j] / L[i * n + i];
                for (int k = 0; k < i; k++) {
                    U[i * n + j] -= ((L[i * n + k] * U[k * n + j]) / L[i * n + i]);
                }
            }
        }
    }
}

void forward_substitution(double* L, double* B, double* Y, int n) {
    for (int i = 0; i < n; i++) {
        Y[i] = B[i];
        for (int j = 0; j < i; j++) {
            Y[i] -= L[i * n + j] * Y[j];
        }
        Y[i] /= L[i * n + i];
    }
}

void backward_substitution(double* U, double* Y, double* X, int n) {
    for (int i = n - 1; i >= 0; i--) {
        X[i] = Y[i];
        for (int j = i + 1; j < n; j++) {
            X[i] -= U[i * n + j] * X[j];
        }
    }
}

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double A[N * N], L[N * N], U[N * N], A_inv[N * N];

    if (rank == 0) {
        generate_matrix(A, N);
        lu_decomposition(A, L, U, N);
    }

    MPI_Bcast(L, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(U, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int cols_per_proc = N / size;
    int extra = N % size;
    int my_cols = cols_per_proc + (rank < extra ? 1 : 0);
    int start_col = rank * cols_per_proc + (rank < extra ? rank : extra);

    double local_result[N * my_cols];

    for (int c = 0; c < my_cols; c++) {
        int col_index = start_col + c;
        double B[N], Y[N], X[N];
        for (int i = 0; i < N; i++) B[i] = (i == col_index) ? 1.0 : 0.0;

        forward_substitution(L, B, Y, N);
        backward_substitution(U, Y, X, N);

        for (int i = 0; i < N; i++) {
            local_result[i * my_cols + c] = X[i];
        }
    }

    int* recvcounts = NULL;
    int* displs = NULL;
    if (rank == 0) {
        recvcounts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));
        int offset = 0;
        for (int i = 0; i < size; i++) {
            int cols = cols_per_proc + (i < extra ? 1 : 0);
            recvcounts[i] = N * cols;
            displs[i] = offset;
            offset += recvcounts[i];
        }
    }

    MPI_Gatherv(local_result, N * my_cols, MPI_DOUBLE, A_inv, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        print_matrix("Inversa de A:", A_inv, N);
    }

    if (recvcounts) free(recvcounts);
    if (displs) free(displs);

    MPI_Finalize();
    return 0;
}
