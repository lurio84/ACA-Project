#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define M 4  // Número de filas de A
#define N 4  // Número de columnas de A y filas de B
#define P 4  // Número de columnas de B

// Función para inicializar matrices
void initialize_matrices(double A[M][N], double B[N][P], double C[M][P]) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            A[i][j] = i + j;  // Valores de prueba

    for (int i = 0; i < N; i++)
        for (int j = 0; j < P; j++)
            B[i][j] = i - j;  // Valores de prueba

    for (int i = 0; i < M; i++)
        for (int j = 0; j < P; j++)
            C[i][j] = 0;  // Inicializar resultado
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Verificar si el número de procesos divide exactamente las filas de A
    if (M % size != 0) {
        if (rank == 0)
            printf("Número de procesos debe dividir exactamente %d filas.\n", M);
        MPI_Finalize();
        return 1;
    }

    int rows_per_proc = M / size;
    double A[M][N], B[N][P], C[M][P];  // Matrices globales solo en root
    double local_A[rows_per_proc][N], local_C[rows_per_proc][P];

    // Inicializar solo en el proceso raíz
    if (rank == 0) {
        initialize_matrices(A, B, C);
        printf("Matriz A:\n");
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++)
                printf("%5.1f ", A[i][j]);
            printf("\n");
        }
        printf("\nMatriz B:\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < P; j++)
                printf("%5.1f ", B[i][j]);
            printf("\n");
        }
        printf("\n");
    }

    // Distribuir filas de A a cada proceso
    MPI_Scatter(A, rows_per_proc * N, MPI_DOUBLE, local_A, rows_per_proc * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Difundir B a todos los procesos
    MPI_Bcast(B, N * P, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Multiplicación local de matrices
    for (int i = 0; i < rows_per_proc; i++) {
        for (int j = 0; j < P; j++) {
            local_C[i][j] = 0;
            for (int k = 0; k < N; k++) {
                local_C[i][j] += local_A[i][k] * B[k][j];
            }
        }
    }

    // Recolectar resultados en la matriz C del proceso raíz
    MPI_Gather(local_C, rows_per_proc * P, MPI_DOUBLE, C, rows_per_proc * P, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Mostrar la matriz resultado en el proceso raíz
    if (rank == 0) {
        printf("Resultado de A * B:\n");
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < P; j++)
                printf("%5.1f ", C[i][j]);
            printf("\n");
        }
    }

    MPI_Finalize();
    return 0;
}
