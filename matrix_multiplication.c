#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define M 6   // Filas de A y C
#define K_DIM 4  // Columnas de A, Filas de B
#define P 5  // Columnas de B y C

// Función para inicializar matrices
void initialize_matrices(double A[M][K_DIM], double B[K_DIM][P], double C[M][P]) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < K_DIM; j++)
            A[i][j] = i + j;  // Valores de prueba

    for (int i = 0; i < K_DIM; i++)
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

    int rows_per_proc = M / size; // Asignar filas equitativamente
    int extra_rows = M % size; // Filas sobrantes si M no es divisible por size

    double A[M][K_DIM], B[K_DIM][P], C[M][P];  // Matrices globales solo en root
    double local_A[rows_per_proc + (rank < extra_rows ? 1 : 0)][K_DIM];
    double local_C[rows_per_proc + (rank < extra_rows ? 1 : 0)][P];

    // Inicializar solo en el proceso raíz
    if (rank == 0) {
        initialize_matrices(A, B, C);
    }

    // Distribuir filas de A a cada proceso (manejo de carga desigual)
    int sendcounts[size], displs[size];
    int offset = 0;
    for (int i = 0; i < size; i++) {
        sendcounts[i] = (rows_per_proc + (i < extra_rows ? 1 : 0)) * K_DIM;
        displs[i] = offset;
        offset += sendcounts[i];
    }

    MPI_Scatterv(A, sendcounts, displs, MPI_DOUBLE, local_A, sendcounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(B, K_DIM * P, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Multiplicación local de matrices
    int local_rows = sendcounts[rank] / K_DIM;
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < P; j++) {
            local_C[i][j] = 0;
            for (int k = 0; k < K_DIM; k++) {
                local_C[i][j] += local_A[i][k] * B[k][j];
            }
        }
    }

    // Recolectar resultados en la matriz C del proceso raíz
    int recvcounts[size], recvdispls[size];
    offset = 0;
    for (int i = 0; i < size; i++) {
        recvcounts[i] = (rows_per_proc + (i < extra_rows ? 1 : 0)) * P;
        recvdispls[i] = offset;
        offset += recvcounts[i];
    }

    MPI_Gatherv(local_C, recvcounts[rank], MPI_DOUBLE, C, recvcounts, recvdispls, MPI_DOUBLE, 0, MPI_COMM_WORLD);

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