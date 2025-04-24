#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <math.h>

void fill_matrices(double *A, double *B, int M, int K_DIM, int P) {
    srand(42);  // Semilla fija para consistencia entre ejecuciones
    for (int i = 0; i < M * K_DIM; i++)
        A[i] = rand() % 10;

    for (int i = 0; i < K_DIM * P; i++)
        B[i] = rand() % 10;
}

void run_multiplication(int M, int K_DIM, int P, int rank, int size) {
    MPI_Barrier(MPI_COMM_WORLD);

    char hostname[256];
    gethostname(hostname, 256);

    char info[300];
    snprintf(info, sizeof(info), "Proceso %d de %d ejecut\u00e1ndose en %s\n", rank, size, hostname);

    if (rank != 0) {
        MPI_Send(info, strlen(info) + 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    }

    double *A = NULL, *C = NULL;
    double *B = malloc(K_DIM * P * sizeof(double));
    int rows_per_proc = M / size;
    int extra_rows = M % size;
    int my_rows = rows_per_proc + (rank < extra_rows ? 1 : 0);
    double *local_A = malloc(my_rows * K_DIM * sizeof(double));
    double *local_C = malloc(my_rows * P * sizeof(double));

    if (rank == 0) {
        printf("\n--- Multiplicaci\u00f3n para matriz %dx%d * %dx%d ---\n", M, K_DIM, K_DIM, P);
        printf("%s", info);
        for (int i = 1; i < size; i++) {
            MPI_Recv(info, 300, MPI_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("%s", info);
        }
        A = malloc(M * K_DIM * sizeof(double));
        C = malloc(M * P * sizeof(double));
        fill_matrices(A, B, M, K_DIM, P);
    }

    MPI_Bcast(B, K_DIM * P, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int *sendcounts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));
    int *recvcounts = malloc(size * sizeof(int));
    int *recvdispls = malloc(size * sizeof(int));

    int offset = 0;
    for (int i = 0; i < size; i++) {
        int r = rows_per_proc + (i < extra_rows ? 1 : 0);
        sendcounts[i] = r * K_DIM;
        displs[i] = offset;
        recvcounts[i] = r * P;
        recvdispls[i] = i == 0 ? 0 : recvdispls[i - 1] + recvcounts[i - 1];
        offset += sendcounts[i];
    }

    double start = MPI_Wtime();

    MPI_Scatterv(A, sendcounts, displs, MPI_DOUBLE, local_A, my_rows * K_DIM, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < my_rows; i++) {
        for (int j = 0; j < P; j++) {
            local_C[i * P + j] = 0;
            for (int k = 0; k < K_DIM; k++)
                local_C[i * P + j] += local_A[i * K_DIM + k] * B[k * P + j];
        }
    }

    MPI_Gatherv(local_C, my_rows * P, MPI_DOUBLE, C, recvcounts, recvdispls, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double end = MPI_Wtime();

    if (rank == 0) {
        printf("Tiempo de ejecuci\u00f3n: %.6f segundos\n", end - start);
    }

    free(B);
    free(local_A);
    free(local_C);
    free(sendcounts);
    free(displs);
    free(recvcounts);
    free(recvdispls);
    if (rank == 0) {
        free(A);
        free(C);
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int base = 1000; // Carga base por proceso
    int scale = (int) ceil(sqrt((double)size)); // Escalar proporcional al número de procesos
    int N = base * scale; // Tamaño de matriz escalado

    run_multiplication(N, N, N, rank, size);

    MPI_Finalize();
    return 0;
}
