#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>

namespace patch
{
    template < typename T > std::string to_string(const T& n)
    {
        std::ostringstream stm;
        stm << n;
        return stm.str();
    }
}

// Функция умножения части матрицы (распределённой по строкам) на вектор
void multiply_matrix_vector_row(const double* local_matrix, const double* vector,
    double* local_result, int local_rows, int cols) {
    for (int i = 0; i < local_rows; ++i) {
        local_result[i] = 0.0;
        for (int j = 0; j < cols; ++j) {
            local_result[i] += local_matrix[i * cols + j] * vector[j];
        }
    }
}

// Функция умножения части матрицы (распределённой по столбцам) на вектор
void multiply_matrix_vector_col(const double* local_matrix,
    const double* vector, double* local_result, int rows,
    int local_cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < local_cols; ++j) {
            local_result[i] += local_matrix[i * local_cols + j] * vector[j];
        }
    }
}

// Функция умножения части матрицы, распределённой блоками, на вектор
void multiply_matrix_vector_block(const double* local_matrix,
    const double* vector, double* local_result, int local_rows,
    int local_cols) {
    for (int i = 0; i < local_rows; ++i) {
        local_result[i] = 0.0;
        for (int j = 0; j < local_cols; ++j) {
            local_result[i] += local_matrix[i * local_cols + j] * vector[j];
        }
    }
}

void run_row_partitioning(int rank, int size, int rows, int cols, double* matrix, double* vector);
void run_col_partitioning(int rank, int size, int rows, int cols, double* matrix, double* vector);
void run_block_partitioning(int rank, int size, int rows, int cols, double* matrix, double* vector);

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int rows, cols;
    std::string partition_type;

    // Чтение аргументов командной строки: тип разбиения и размер матрицы (rows = cols)
    if (argc > 2) {
        rows = cols = atoi(argv[2]);  // Считаем, что матрица квадратная
        partition_type = argv[1];     // Строка определяющая тип разбиения ("row", "col", "block")
    }

    double* matrix = nullptr;
    double* vector = nullptr;

    // Инициализация матрицы и вектора в процессе с rank=0
    if (rank == 0) {
        matrix = new double[rows * cols];
        vector = new double[cols];
        srand(unsigned(clock()));

        // Заполнение матрицы случайными числами от 0 до 9
        for (int i = 0; i < rows * cols; ++i) {
            matrix[i] = rand() % 10;
        }
        // Заполнение вектора случайными числами от 0 до 9
        for (int i = 0; i < cols; ++i) {
            vector[i] = rand() % 10;
        }
    }

    // Вызов соответствующей функции разбиения и умножения в зависимости от partition_type
    if (partition_type == "row") {
        run_row_partitioning(rank, num_procs, rows, cols, matrix, vector);
    }
    else if (partition_type == "col") {
        run_col_partitioning(rank, num_procs, rows, cols, matrix, vector);
    }
    else if (partition_type == "block") {
        run_block_partitioning(rank, num_procs, rows, cols, matrix, vector);
    }
    else {
        // Неизвестный тип разбиения
        std::cout << "Unknown type of partition: " << partition_type << "\n";
    }

    // Освобождение памяти в процессе 0
    if (rank == 0) {
        delete[] matrix;
        delete[] vector;
    }

    MPI_Finalize();
    return 0;
}

// Функция для распределения матрицы по строкам между процессами, умножения и сбора результата.
void run_row_partitioning(int rank, int size, int rows, int cols, double* matrix, double* vector) {
    int local_rows = rows / size;

    std::vector<int> sendcounts(size, local_rows * cols);
    std::vector<int> displs(size);

    int offset = 0;
    for (int i = 0; i < size; ++i) {
        displs[i] = offset;
        offset += sendcounts[i];
    }

    // Выделение памяти для локальной части матрицы и локального результата
    std::vector<double> local_matrix(sendcounts[rank]);
    std::vector<double> local_result(local_rows);

    MPI_Scatterv(matrix, sendcounts.data(), displs.data(), MPI_DOUBLE,
        local_matrix.data(), sendcounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<double> local_vector(cols);
    MPI_Bcast(local_vector.data(), cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // Вычисление локального результата умножения
    multiply_matrix_vector_row(local_matrix.data(), local_vector.data(), local_result.data(), local_rows, cols);

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    double elapsed_time = end_time - start_time;

    // Суммируем время со всех процессов и собираем в нулевом процессе для усреднения
    double total_time;
    MPI_Reduce(&elapsed_time, &total_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Собираем результаты обратно в один вектор в процессе 0
    std::vector<int> recvcounts(size, local_rows);
    std::vector<int> recvdispls(size);
    int offset_recv = 0;

    for (int i = 0; i < size; ++i) {
        recvdispls[i] = offset_recv;
        offset_recv += recvcounts[i];
    }

    std::vector<double> result;
    if (rank == 0) {
        result.resize(rows);
    }

    MPI_Gatherv(local_result.data(), local_rows, MPI_DOUBLE, result.data(),
        recvcounts.data(), recvdispls.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Запись результата и времени в файл
    if (rank == 0) {
        std::string filename = "row_" + std::to_string(size) + "_" + std::to_string(rows) + ".txt";
        std::ofstream outfile(filename);
        outfile << total_time / size << std::endl;
        for (const auto& val : result) {
            outfile << val << " ";
        }
        outfile << std::endl;
        outfile.close();
    }
}

// Функция для распределения матрицы по столбцам между процессами, умножения и сбора результата.
void run_col_partitioning(int rank, int size, int rows, int cols, double* matrix, double* vector) {
    // Определяем количество столбцов для каждого процесса
    int local_cols = cols / size;
    int remainder = cols % size;
    if (rank < remainder) {
        local_cols += 1;
    }

    // Формируем sendcounts и displs для Scatterv по столбцам
    std::vector<int> sendcounts(size);
    std::vector<int> displs(size);
    int offset = 0;
    for (int i = 0; i < size; ++i) {
        sendcounts[i] = rows * (cols / size);
        if (i < remainder) {
            sendcounts[i] += rows;
        }
        displs[i] = offset;
        offset += sendcounts[i];
    }

    // Выделяем память под локальную часть матрицы, локальный вектор и локальный результат
    std::vector<double> local_matrix(rows * local_cols);
    std::vector<double> local_vector(local_cols);
    std::vector<double> local_result(rows, 0.0);

    // Рассылка столбцов матрицы
    MPI_Scatterv(matrix, sendcounts.data(), displs.data(), MPI_DOUBLE,
        local_matrix.data(), rows * local_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Теперь рассылаем соответствующие части вектора
    std::vector<int> vec_sendcounts(size);
    std::vector<int> vec_displs(size);
    offset = 0;
    for (int i = 0; i < size; ++i) {
        vec_sendcounts[i] = cols / size;
        if (i < remainder) {
            vec_sendcounts[i] += 1;
        }
        vec_displs[i] = offset;
        offset += vec_sendcounts[i];
    }

    MPI_Scatterv(vector, vec_sendcounts.data(), vec_displs.data(), MPI_DOUBLE,
        local_vector.data(), local_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Синхронизация перед измерением времени
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // Локальное умножение части матрицы на часть вектора
    multiply_matrix_vector_col(local_matrix.data(), local_vector.data(), local_result.data(), rows, local_cols);

    // Синхронизация после вычислений
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    double elapsed_time = end_time - start_time;
    double total_time;

    // Суммирование времен
    MPI_Reduce(&elapsed_time, &total_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    std::vector<double> result;
    if (rank == 0) {
        result.resize(rows);
    }
    MPI_Reduce(local_result.data(), result.data(), rows, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Записываем результат и время в файл
    if (rank == 0) {
        std::string filename = "col_" + patch::to_string(size) + "_" + patch::to_string(rows) + ".txt";
        std::ofstream outfile(filename);
        outfile << total_time / size << std::endl;
        for (const auto& val : result) {
            outfile << val << " ";
        }
        outfile << std::endl;
        outfile.close();
    }
}

// Функция для распределения матрицы по блокам между процессами, умножения и сбора результата.
void run_block_partitioning(int rank, int size, int rows, int cols, double* matrix, double* vector) {
    // Создаем двумерный декартовский коммуникатор
    int dims[2] = { 0, 0 };
    MPI_Dims_create(size, 2, dims);    // Определяем оптимальное разбиение процессов по двум измерениям
    int periods[2] = { 0, 0 };         // Без периодичности
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);

    int coords[2];
    MPI_Cart_coords(cart_comm, rank, 2, coords);

    // Вычисляем размер локального блока матрицы
    int local_rows = rows / dims[0];
    int local_cols = cols / dims[1];

    int row_remainder = rows % dims[0];
    int col_remainder = cols % dims[1];

    if (coords[0] < row_remainder) local_rows++;
    if (coords[1] < col_remainder) local_cols++;

    // Выделяем память под локальный блок матрицы, локальную часть вектора и результат
    std::vector<double> local_matrix(local_rows * local_cols);
    std::vector<double> local_vector(local_cols);
    std::vector<double> local_result(local_rows);

    // Распределение матрицы по блокам
    MPI_Datatype block_type;
    MPI_Type_vector(local_rows, local_cols, cols, MPI_DOUBLE, &block_type);
    MPI_Type_commit(&block_type);

    if (rank == 0) {
        for (int i = 0; i < size; ++i) {
            int dest_coords[2];
            MPI_Cart_coords(cart_comm, i, 2, dest_coords);

            int dest_local_rows = rows / dims[0];
            if (dest_coords[0] < row_remainder) dest_local_rows++;

            int dest_local_cols = cols / dims[1];
            if (dest_coords[1] < col_remainder) dest_local_cols++;

            int dest_row_start = dest_coords[0] * (rows / dims[0]) + std::min(dest_coords[0], row_remainder);
            int dest_col_start = dest_coords[1] * (cols / dims[1]) + std::min(dest_coords[1], col_remainder);

            // Если это процесс 0, копируем данные напрямую.
            if (i == 0) {
                for (int r = 0; r < dest_local_rows; ++r) {
                    for (int c = 0; c < dest_local_cols; ++c) {
                        local_matrix[r * dest_local_cols + c] = matrix[(dest_row_start + r) * cols + dest_col_start + c];
                    }
                }
            }
            else {
                // Создаём тип для отправки конкретного блока
                MPI_Datatype send_block_type;
                MPI_Type_vector(dest_local_rows, dest_local_cols, cols, MPI_DOUBLE, &send_block_type);
                MPI_Type_commit(&send_block_type);
                MPI_Send(&matrix[dest_row_start * cols + dest_col_start], 1, send_block_type, i, 0, cart_comm);
                MPI_Type_free(&send_block_type);
            }
        }
    }
    else {
        // Остальные процессы получают свои блоки матрицы
        MPI_Recv(local_matrix.data(), local_rows * local_cols, MPI_DOUBLE, 0, 0, cart_comm, MPI_STATUS_IGNORE);
    }
    MPI_Type_free(&block_type);

    // Распределение частей вектора между процессами по столбцам
    if (rank == 0) {
        for (int i = 0; i < size; ++i) {
            int dest_coords[2];
            MPI_Cart_coords(cart_comm, i, 2, dest_coords);

            int dest_local_cols = cols / dims[1];
            if (dest_coords[1] < col_remainder) dest_local_cols++;

            int dest_col_start = dest_coords[1] * (cols / dims[1]) + std::min(dest_coords[1], col_remainder);

            if (i == 0) {
                // Заполняем локальный вектор для процесса 0
                for (int c = 0; c < dest_local_cols; ++c) {
                    local_vector[c] = vector[dest_col_start + c];
                }
            }
            else {
                // Отправляем часть вектора другим процессам
                MPI_Send(&vector[dest_col_start], dest_local_cols, MPI_DOUBLE, i, 0, cart_comm);
            }
        }
    }
    else {
        // Получаем локальную часть вектора
        MPI_Recv(local_vector.data(), local_cols, MPI_DOUBLE, 0, 0, cart_comm, MPI_STATUS_IGNORE);
    }

    // Синхронизация перед началом вычислений
    MPI_Barrier(cart_comm);
    double start_time = MPI_Wtime();

    // Локальное умножение блока матрицы на соответствующую часть вектора
    multiply_matrix_vector_block(local_matrix.data(), local_vector.data(), local_result.data(), local_rows, local_cols);

    // Синхронизация после вычислений
    MPI_Barrier(cart_comm);
    double end_time = MPI_Wtime();
    double elapsed_time = end_time - start_time;
    double total_time;

    MPI_Reduce(&elapsed_time, &total_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Сбор результатов обратно на процесс 0
    int* recvcounts = nullptr;
    int* displs = nullptr;
    int local_row_count = local_rows;

    if (rank == 0) {
        recvcounts = new int[size];
        displs = new int[size];
    }

    MPI_Gather(&local_row_count, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, cart_comm);

    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < size; ++i) {
            displs[i] = displs[i - 1] + recvcounts[i - 1];
        }
    }

    std::vector<double> result;
    if (rank == 0) {
        int total_rows = 0;
        for (int i = 0; i < size; ++i) {
            total_rows += recvcounts[i];
        }
        result.resize(total_rows);
    }

    MPI_Gatherv(local_result.data(), local_rows, MPI_DOUBLE, result.data(),
        recvcounts, displs, MPI_DOUBLE, 0, cart_comm);

    // Запись результата в файл
    if (rank == 0) {
        std::string filename = "block_" + patch::to_string(size) + "_" + patch::to_string(rows) + ".txt";
        std::ofstream outfile(filename);
        outfile << total_time / size << std::endl;
        for (const auto& val : result) {
            outfile << val << " ";
        }
        outfile << std::endl;
        outfile.close();
        delete[] recvcounts;
        delete[] displs;
    }

    MPI_Comm_free(&cart_comm);
}
