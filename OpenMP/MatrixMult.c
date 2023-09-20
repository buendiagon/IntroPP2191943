#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Función para generar una matriz aleatoria de enteros entre 0 y 100.
int** generateMatrix(int rows, int cols) {
    // Reserva memoria para las filas de la matriz.
    int** matrix = (int**)malloc(rows * sizeof(int*)); 
    srand(time(NULL));
    for (int i=0; i<rows; i++) {
        // Reserva memoria para las columnas de la matriz.
        matrix[i] = (int*)malloc(cols * sizeof(int));
        for (int j=0; j<cols; j++) {
            // Llena la matriz con valores aleatorios entre 0 y 100.
            matrix[i][j] = rand() % 101;
        }
    }
    return matrix;
}

// Función para mostrar una matriz en la consola.
void displayMatrix(int** matrix) {
    // Calcula el número de filas de la matriz.
    int rows = sizeof(matrix) / sizeof(matrix[0]);
    // Calcula el número de columnas de la matriz.
    int cols = sizeof(matrix[0]) / sizeof(matrix[0][0]);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
}

// Función para multiplicar dos matrices y almacenar el resultado en una tercera matriz.
void multiplyMatrix(int** mat1, int** mat2, int** result) {
    // Calcula el número de filas de la primera matriz.
    int rows1 = sizeof(mat1) / sizeof(mat1[0]);
    // Calcula el número de columnas de la primera matriz.
    int cols1 = sizeof(mat1[0]) / sizeof(mat1[0][0]);
    // Calcula el número de columnas de la segunda matriz.
    int cols2 = sizeof(mat2[0]) / sizeof(mat2[0][0]);

    for (int i = 0; i < rows1; i++) {
        for (int j = 0; j < cols2; j++) {
            result[i][j] = 0;
            for (int k = 0; k < cols1; k++) {
                result[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }
}

// Función para liberar la memoria utilizada por una matriz.
void freeMemory(int** m, int rows) {
    for (int i; i < rows; i++) {
        free(m[i]);
    }
    free(m);
}


int main() {
    int row1, col1, row2, col2;
    double execution_time = 0.0;
    printf("Input number of rows for matrix 1: ");
    scanf("%d", &row1);
    printf("Input number of cols for matrix 1: ");
    scanf("%d", &col1);
    printf("Input number of rows for matrix 2: ");
    scanf("%d", &row2);
    printf("Input number of cols for matrix 2: ");
    scanf("%d", &col2);

    if (col1 != row2) {
        // Si la matriz no tiene el formato adecuado se retorna un error
        printf("The number of columns in Matrix-1 must be "
               "equal to the number of rows in Matrix 2\n");
        return 0;
    }

    // Se generan dos matrices aleatorias con las dimensiones dadas por el usuario.
    int** m1 = generateMatrix(row1, col1);
    int** m2 = generateMatrix(row2, col2);
    // Se separa la memoria que usara la matriz con las respuestas
    int** result = (int**)malloc(row1 * sizeof(int*));
    for (int i = 0; i < row1; i++) {
        result[i] = (int*)malloc(col2 * sizeof(int));
    }

    // Se registra el tiempo de inicio de la multiplicación
    clock_t start = clock();

    multiplyMatrix(m1, m2, result);

    // Se registra el tiempo de finalización de la multiplicación.
    clock_t end = clock();

    // Se libera la memoria de las matrices
    freeMemory(m1, row1);
    freeMemory(m2, row2);

    displayMatrix(result);
    freeMemory(result,row1);

    // Se calcula el tiempo de ejecución y se muestra en consola.
    execution_time += (double)(end-start) / CLOCKS_PER_SEC;

    printf("Matrix multiplication time: %.6lf seconds", execution_time);

    return 0;
}