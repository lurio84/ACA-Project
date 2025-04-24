#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#define N 4  // Tamaño de la matriz cuadrada

void print_matrix(const char* label, double* M, int n) {
    printf("\n%s\n", label);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            printf("%8.4f ", M[i * n + j]);
        printf("\n");
    }
    printf("\n");
}

void generate_matrix(double* A, int n) {
    srand(42);  // Semilla fija
    for (int i = 0; i < n * n; i++) {
        A[i] = (rand() % 5) + 1;
    }
}

// LU descomposición (Doolittle sin pivoteo)
void lu_decomposition(double* A, double* L, double* U, int n) {
    for (int i = 0; i < n * n; i++) {
        L[i] = 0.0;
        U[i] = 0.0;
    }

    for (int i = 0; i < n; i++) {
        // L
        for (int j = i; j < n; j++) {
            L[j * n + i] = A[j * n + i];
            for (int k = 0; k < i; k++)
                L[j * n + i] -= L[j * n + k] * U[k * n + i];
        }

        // U
        for (int j = i; j < n; j++) {
            if (i == j)
                U[i * n + j] = 1.0;
            else {
                U[i * n + j] = A[i * n + j];
                for (int k = 0; k < i; k++)
                    U[i * n + j] -= L[i * n + k] * U[k * n + j];
                U[i * n + j] /= L[i * n + i];
            }
        }
    }
}

void forward_substitution(double* L, double* B, double* Y, int n) {
    for (int i = 0; i < n; i++) {
        Y[i] = B[i];
        for (int j = 0; j < i; j++)
            Y[i] -= L[i * n + j] * Y[j];
        Y[i] /= L[i * n + i];
    }
}

void backward_substitution(double* U, double* Y, double* X, int n) {
    for (int i = n - 1; i >= 0; i--) {
        X[i] = Y[i];
        for (int j = i + 1; j < n; j++)
            X[i] -= U[i * n + j] * X[j];
    }
}

int main(int argc, char** argv) {
    int rank, size;
    char hostname[256];
    gethostname(hostname, sizeof(hostname));

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0)
        printf("\n=== Inversión de matriz %dx%d distribuida con LU y MPI ===\n", N, N);

    double A[N * N], L[N * N], U[N * N], A_inv[N * N];
    memset(A_inv, 0, sizeof(A_inv));

    // Cada proceso reporta
    char info[300];
    snprintf(info, sizeof(info), "Proceso %d de %d ejecutándose en %s\n", rank, size, hostname);
    if (rank == 0) {
        printf("%s", info);
        for (int i = 1; i < size; i++) {
            MPI_Recv(info, 300, MPI_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("%s", info);
        }
    } else {
        MPI_Send(info, strlen(info) + 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    }

    double start = MPI_Wtime();

    if (rank == 0) {
        generate_matrix(A, N);
        print_matrix("Matriz A generada:", A, N);
        lu_decomposition(A, L, U, N);
    }

    // Compartimos L y U con todos
    MPI_Bcast(L, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(U, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Distribuir columnas de la inversa entre procesos
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

    // Preparar recopilación en root
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

    double end = MPI_Wtime();

    if (rank == 0) {
        print_matrix("Matriz inversa A⁻¹:", A_inv, N);
        printf("Tiempo de ejecución de inversión: %.6f segundos\n", end - start);
        if (recvcounts) free(recvcounts);
        if (displs) free(displs);
    }

    MPI_Finalize();
    return 0;
}
