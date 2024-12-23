// Лабораторная работа выполнена студентом 121702 группы Летко А.Ю
// Вариант 3 (Реализовать сеть Хопфилда работающую в асинхронном режиме и с непрерывным состоянием)
// Источник https://habr.com/ru/articles/561198/
// https://github.com/z0rats/js-neural-networks/tree/master/hopfield_network

function preprocess_alphabet(alphabet) {
    const processedAlphabet = alphabet.map(item => {
        if (Array.isArray(item) && Array.isArray(item[0])) {
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
        this.iters = 0
    }

    _get_neg_images(images) {
        return images.map(image => image.map(value => value * -1));
    }
    
    train(e = 1e-6, max_iters = 10000) {
        for(let i = 0; i < max_iters; i++) {
            this.iters = i
            const old_w = this.w.map(row => [...row]);

            for(let image of this.images) {
                const x_t = transposeMattrix([image])
                const activation = multiplyMatrices(this.w, x_t).map(arr => arr.map(value => Math.tanh(value)))
                this.w = addMatrices(
                    this.w,
                    multiplyByNumber(
                        multiplyMatrices(
                            subtractMatrices(x_t, activation),
                            transposeMattrix(x_t)
                        ),
                        this.nu / this.size,
                    ) 
                )

                for (let i = 0; i < this.w.length; i++) {
                    this.w[i][i] = 0;
                }
            }

            let diffSum = 0;

            for (let i = 0; i < old_w.length; i++) {
                for (let j = 0; j < old_w[i].length; j++) {
                    diffSum += Math.abs(old_w[i][j] - this.w[i][j]);
                }
            }

            // Проверка условия
            if (diffSum < e) {
                // console.log('Weights have converged');
                break;
            }
        }

        for (let i = 0; i < this.w.length; i++) {
            this.w[i][i] = 0;
        }
    }

    findImageNum(x, images) {
        for (let idx = 0; idx < images.length; idx++) {
            const image = images[idx];
            const maxDiff = Math.max(...image.map((val, i) => Math.abs(val - x[i])));
    
            if (maxDiff < 1e-2) {
                return idx; // Возвращает индекс изображения, если найдено совпадение
            }
        }
        return null; // Если не найдено совпадение
    }

    predict(x, max_iters = 1000) {
        let states = Array(4).fill(x.slice());
        let relaxation_iters = 0

        for(let i = 0; i < max_iters; i++) {
            relaxation_iters += 1
            // console.log(states[states.length - 1], transposeMattrix([states[states.length - 1]]), "transposeMattrix(states[states.length - 1])")
            // console.log(multiplyMatrices(this.w, transposeMattrix(states[states.length - 1]), true), "KSFHAKJFSAKFH")
            console.log(states, "STATES")
            let new_state = transposeMattrix(multiplyMatrices(this.w, transposeMattrix([states[states.length - 1]]), true)).map(arr => arr.map(value => Math.tanh(value)))
            // console.log(new_state, "NEW_STATE")
            states.push(...new_state); // Добавляем новый элемент в конец массива
            states.shift();

            if(i >= 3 && this.findAbsMax(0, 2, states) < 1e-8 && this.findAbsMax(1, 3, states) < 1e-8) {
                let image_num = this.findImageNum(new_state, this.images)
                let neg_image_num = this.findImageNum(new_state, this.neg_images)

                let isNegative = neg_image_num !== null;

                return {
                    relaxation_iters,
                    new_state,
                    image_num: image_num !== null ? image_num : neg_image_num,
                    is_negative: isNegative
                };
            }
        }
        
        return {
            relaxation_iters: max_iters,
            new_state,
            image_num: null,
            is_negative: null
        };
    }

    findAbsMax(first, second, states) {
        let difference = states[first].map((val, index) => Math.abs(val - states[second][index]));

        // Нахождение максимального значения
        return Math.max(...difference);
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
    ],
]

alphabet = preprocess_alphabet(alphabet)

network = new Hopfield(alphabet, 0.7)
network.train()

function imageBeautifulPrint(image, rows, cols) {
    image = image.map(value => Math.sign(value));

    image = image.map(value => value === 1 ? ' # ' : ' O ');

    const image2D = [];
    for (let i = 0; i < rows; i++) {
        image2D.push(image.slice(i * cols, (i + 1) * cols));
    }

    image2D.forEach(row => {
        console.log(row.join(''));
    });
}

test_image = [-1, 1, 1, -1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1]

const {relaxation_iters: r_iters, new_state: state, image_num: image_idx, is_negative} = network.predict(test_image, 10000)

let predicted_image = null

if(image_idx) {
    predicted_image = is_negative ? network.neg_images[image_idx] : network.images[image_idx]
} else {
    predicted_image = state
}

console.log("All images")
for(let img of alphabet) {
    imageBeautifulPrint(img.flat(), 4, 4)
    console.log()
}3
console.log("INPUT IMAGE")
imageBeautifulPrint(test_image, 4, 4)
console.log('PREDICTED IMAGE')
imageBeautifulPrint(...predicted_image, 4, 4)

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

function multiplyMatrices(matrixA, matrixB, show=false) {
    // if(show) console.log(matrixB, "MATRIX B")
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