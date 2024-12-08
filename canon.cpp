#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <random>
#include <string>

constexpr float MAX_VAL = 10.0f;
constexpr float MIN_VAL = -10.0f;

constexpr int NOT_ENOUGH_ARGS = 1;
constexpr int WRONG_N = 2;

// Функция для инициализации матриц с использованием генератора случайных чисел
void initialize_matrices(std::vector<float>& A, std::vector<float>& B, int n) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(MIN_VAL, MAX_VAL);

    for (auto& val : A) {
        val = dis(gen);
    }

    for (auto& val : B) {
        val = dis(gen);
    }
}

// Функция для перемножения локальных блоков матриц
void multiply_blocks(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int block_size) {
    for (int i = 0; i < block_size; ++i) {
        for (int j = 0; j < block_size; ++j) {
            for (int k = 0; k < block_size; ++k) {
                C[i * block_size + j] += A[i * block_size + k] * B[k * block_size + j];
            }
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int sqrt_size = static_cast<int>(std::sqrt(static_cast<double>(size)));
    if (sqrt_size * sqrt_size != size) {
        MPI_Finalize();
        return NOT_ENOUGH_ARGS;
    }

    if (argc < 2) {
        MPI_Finalize();
        return NOT_ENOUGH_ARGS;
    }

    int n = std::atoi(argv[1]);

    if (n % sqrt_size != 0) {
        MPI_Finalize();
        return WRONG_N;
    }

    int block_size = n / sqrt_size;
    std::vector<float> local_A(block_size * block_size, 0.0f);
    std::vector<float> local_B(block_size * block_size, 0.0f);
    std::vector<float> local_C(block_size * block_size, 0.0f);

    std::vector<float> A;
    std::vector<float> B;

    if (rank == 0) {
        A.resize(n * n);
        B.resize(n * n);
        initialize_matrices(A, B, n);
    }

    MPI_Barrier(MPI_COMM_WORLD); // Синхронизация процессов перед замером времени
    double start_time = MPI_Wtime();

    MPI_Comm grid_comm;
    int dims[2] = { sqrt_size, sqrt_size };
    int periods[2] = { 1, 1 };
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &grid_comm);

    int coords[2];
    MPI_Cart_coords(grid_comm, rank, 2, coords);

    if (rank == 0) {
        for (int dest_rank = 0; dest_rank < size; ++dest_rank) {
            int dest_coords[2];
            MPI_Cart_coords(grid_comm, dest_rank, 2, dest_coords);
            int row = dest_coords[0];
            int col = dest_coords[1];

            std::vector<float> temp_block_A(block_size * block_size);
            std::vector<float> temp_block_B(block_size * block_size);

            for (int i = 0; i < block_size; ++i) {
                int global_row = row * block_size + i;
                std::copy(
                    A.begin() + global_row * n + col * block_size,
                    A.begin() + global_row * n + col * block_size + block_size,
                    temp_block_A.begin() + i * block_size
                );

                std::copy(
                    B.begin() + global_row * n + col * block_size,
                    B.begin() + global_row * n + col * block_size + block_size,
                    temp_block_B.begin() + i * block_size
                );
            }


            if (dest_rank == 0) {
                local_A = temp_block_A;
                local_B = temp_block_B;
            }
            else {
                MPI_Send(temp_block_A.data(), block_size * block_size, MPI_FLOAT, dest_rank, 0, MPI_COMM_WORLD);
                MPI_Send(temp_block_B.data(), block_size * block_size, MPI_FLOAT, dest_rank, 1, MPI_COMM_WORLD);
            }
        }
    }
    else {
        MPI_Recv(local_A.data(), block_size * block_size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(local_B.data(), block_size * block_size, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    MPI_Status status;
    int src, dest;

    MPI_Cart_shift(grid_comm, 1, -coords[0], &src, &dest);
    MPI_Sendrecv_replace(local_A.data(), block_size * block_size, MPI_FLOAT, dest, 0, src, 0, grid_comm, &status);

    MPI_Cart_shift(grid_comm, 0, -coords[1], &src, &dest);
    MPI_Sendrecv_replace(local_B.data(), block_size * block_size, MPI_FLOAT, dest, 0, src, 0, grid_comm, &status);

    for (int step = 0; step < sqrt_size; ++step) {
        multiply_blocks(local_A, local_B, local_C, block_size);

        MPI_Cart_shift(grid_comm, 1, -1, &src, &dest);
        MPI_Sendrecv_replace(local_A.data(), block_size * block_size, MPI_FLOAT, dest, 0, src, 0, grid_comm, &status);

        MPI_Cart_shift(grid_comm, 0, -1, &src, &dest);
        MPI_Sendrecv_replace(local_B.data(), block_size * block_size, MPI_FLOAT, dest, 0, src, 0, grid_comm, &status);
    }

    double end_time = MPI_Wtime();
    double duration = end_time - start_time;
    double total_time, minimal_time;

    MPI_Reduce(&duration, &total_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&duration, &minimal_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::string filename = "kanon_" + std::to_string(size) + "_" + std::to_string(n) + ".txt";
        std::ofstream output_file(filename, std::ios::app);
        if (output_file.is_open()) {
            output_file << n << "\n" << size << "\n" << total_time / size << "\n" << minimal_time << "\n";
            output_file.close();
        }
    }

    MPI_Finalize();
    return 0;
}
