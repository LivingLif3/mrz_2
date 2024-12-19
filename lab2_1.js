function preprocess_alphabet(alphabet) {
    const processedAlphabet = alphabet.map(item => {
        if (Array.isArray(item) && Array.isArray(item[0])) {
            // Преобразуем двумерный массив в одномерный
            return item.flat();
        } else {
            return item;
        }
    });
    return processedAlphabet;
}

class Hopfield {
    constructor(images, nu = 1) {
        this.size = images[0].length
        this.w = Array.from({ length: this.size }, () => Array(this.size).fill(0));
        this.images = images
        this.neg_images = this._get_neg_images(this.images)
        this.nu = nu
    }

    _get_neg_images(images) {
        return images.map(image => image.map(value => value * -1));
    }
    
    train(e = 1e-6, max_iters = 10000) {
        for(let i = 0; i < max_iters; i++) {
            const old_w = this.w.map(row => [...row]);

            for(let image of this.images) {
                console.log(image, "IMAGES")
                const x_t = transposeMattrix([image])
                console.log(x_t)
                const activation = multiplyMatrices(this.w, x_t).map(value => Math.tanh(value))
                
                this.w += addMatrices(
                    this.w,
                    multiplyByNumber(
                        this.nu / this.size,
                        multiplyMatrices(
                            subtractMatrices(x_t, activation),
                            transposeMattrix(x_t)
                        )
                    ) 
                )
                console.log(this.w)
            }
        }
    }
}

let alphabet = [
    [
        [-1, 1, 1, -1],
        [1, -1, -1, 1],
        [1, 1, 1, 1],
        [1, -1, -1, 1]
    ],
    [
        [1, 1, 1, -1],
        [1, -1, -1, 1],
        [1, -1, -1, 1],
        [1, 1, 1, -1]
    ],
    [
        [-1, 1, 1, -1],
        [1, -1, -1, 1],
        [1, -1, -1, 1],
        [-1, 1, 1, -1]
    ],
    [
        [-1, 1, 1, 1],
        [1, -1, -1, -1],
        [1, -1, -1, -1],
        [-1, 1, 1, 1]
    ]
]

alphabet = preprocess_alphabet(alphabet)

network = new Hopfield(alphabet, 0.7)
network.train()

// console.log(preprocess_alphabet([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]))

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
    console.log(matrixA, matrixB)
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