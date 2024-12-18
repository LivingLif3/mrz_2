const fs = require('fs');
const path = require('path');
const math = require('mathjs');

function stepByStep() {
    const h = 0.8;
    const ethalons = [
        [[1, -1, -1, 1]],
        [[-1, -1, 1, 1]],
        [[1, -1, 1, -1]],
        [[1, -1, -1, -1]],
    ];

    let W = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ];

    const noisy = [1, -1, 1, 1];
    const epsilon = 1e-9;
    const N = noisy.length;
    let change = 0;
    const changes = [];

    do {
        change = 0;

        for (let ethalon of ethalons) {
            const transposedEthalon = transposeMattrix(ethalon);
            const product = multiplyMatrices(W, transposedEthalon);
            const deltaW = multiplyByNumber(
                multiplyMatrices(
                    subtractMatrices(transposedEthalon, product),
                    ethalon
                ),
                h / N
            );

            // Приведение матриц к одинаковому размеру
            deltaW.forEach((row, i) => {
                if (!W[i]) W[i] = [];
                row.forEach((value, j) => {
                    W[i][j] = (W[i][j] || 0) + value; // Складываем элементы или инициализируем
                });
            });

            change += calculateChange(deltaW);
        }

        changes.push(change);
        if (changes.length > 1) {
            change = Math.abs(change - changes[changes.length - 2]);
        }

    } while (change > epsilon);

    return W;
}



function transposeMattrix(matrix) {
    const rows = matrix.length;
    const cols = matrix[0].length;
    const transposed = [];
  
    for (let i = 0; i < cols; i++) {
      transposed[i] = [];
      for (let j = 0; j < rows; j++) {
        transposed[i][j] = matrix[j][i];
      }
    }
  
    return transposed;
  }

function multiplyMatrices(matrixA, matrixB) {
    const rowsA = matrixA.length;
    const colsA = matrixA[0].length;
    const rowsB = matrixB.length;
    const colsB = matrixB[0].length;
  

    if (colsA !== rowsB) {
      throw new Error("Количество столбцов первой матрицы должно быть равно количеству строк второй матрицы.");
    }
  
    const result = new Array(rowsA).fill(null).map(() => new Array(colsB).fill(0));
  
    for (let i = 0; i < rowsA; i++) {
      for (let j = 0; j < colsB; j++) {
        for (let k = 0; k < colsA; k++) {
          result[i][j] += matrixA[i][k] * matrixB[k][j];
        }
      }
    }
  
    return result;
}
  
function subtractMatrices(matrixA, matrixB) {
    const rowsA = matrixA.length;
    const colsA = matrixA[0].length;
    const rowsB = matrixB.length;
    const colsB = matrixB[0].length;
  
    if (rowsA !== rowsB || colsA !== colsB) {
      throw new Error("Размеры матриц должны совпадать для выполнения вычитания.");
    }
  
    const result = new Array(rowsA).fill(null).map(() => new Array(colsA).fill(0));
  
    for (let i = 0; i < rowsA; i++) {
      for (let j = 0; j < colsA; j++) {
        result[i][j] = matrixA[i][j] - matrixB[i][j];
      }
    }
  
    return result;
}

function multiplyByNumber(matrixA, num) {
    return matrixA.map(row => row.map(el => el * num))
}

function addMatrices(matrixA, matrixB) {
    // Проверяем, совпадают ли размеры матриц
    if (matrixA.length !== matrixB.length || matrixA[0].length !== matrixB[0].length) {
        throw new Error('Матрицы должны быть одинакового размера');
    }

    // Создаем новую матрицу для хранения результата сложения
    const result = [];

    // Итерация по строкам
    for (let i = 0; i < matrixA.length; i++) {
        // Итерация по элементам в строке
        const row = [];
        for (let j = 0; j < matrixA[i].length; j++) {
            row.push(matrixA[i][j] + matrixB[i][j]); // Сложение соответствующих элементов
        }
        result.push(row); // Добавление строки в результат
    }

    return result;
}

function calculateChange(deltaW) {
    let change = 0;

    for (let i = 0; i < deltaW.length; i++) {
        for (let j = 0; j < deltaW[i].length; j++) {
            change += Math.abs(deltaW[i][j]);
        }
    }

    return change;
}

console.log(stepByStep())