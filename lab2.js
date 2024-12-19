class Hopfield {
    constructor(images, nu = 1) {
        this.size = images[0].length;
        this.w = Array.from({ length: this.size }, () => Array(this.size).fill(0));
        this.images = images;
        this.negImages = this._getNegImages(images);
        this.nu = nu;
    }

    _getNegImages(images) {
        return images.map(image => image.map(value => -value));
    }

    train(e = 1e-6, maxIters = 10000) {
        for (let iter = 0; iter < maxIters; iter++) {
            const oldW = this.w.map(row => [...row]);

            for (const image of this.images) {
                const xT = image.map(val => [val]); // Transpose of x
                const deltaW = this._matrixMult(
                    this._scalarMult(this.nu / this.size, this._matrixSub(xT, this._matrixMult(this.w, xT))),
                    this._transpose(xT)
                );
                this.w = this._matrixAdd(this.w, deltaW);

                // Set diagonal to 0
                for (let i = 0; i < this.size; i++) {
                    this.w[i][i] = 0;
                }
            }

            // Check convergence
            const diff = this._matrixDiff(oldW, this.w);
            if (diff < e) break;
        }

        // Ensure diagonal is 0
        for (let i = 0; i < this.size; i++) {
            this.w[i][i] = 0;
        }
    }

    predict(x, maxIters = 1000) {
        const states = Array(4).fill(x.map(val => [val])); // Store states as column vectors
        let relaxationIters = 0;

        for (let iter = 0; iter < maxIters; iter++) {
            relaxationIters++;
            const newState = this._transpose(this._matrixTanh(this._matrixMult(this.w, this._transpose(states[3]))));
            states.shift();
            states.push(newState);

            if (iter >= 3) {
                const diff1 = this._matrixMaxAbs(this._matrixSub(states[0], states[2]));
                const diff2 = this._matrixMaxAbs(this._matrixSub(states[1], states[3]));

                if (diff1 < 1e-10 && diff2 < 1e-10) {
                    const imageNum = this._findImageNum(newState, this.images);
                    const negImageNum = this._findImageNum(newState, this.negImages);
                    const isNegative = negImageNum !== null;

                    return { relaxationIters, newState, imageNum: imageNum ?? negImageNum, isNegative };
                }
            }
        }

        return { relaxationIters: maxIters, newState: states[3], imageNum: null, isNegative: null };
    }

    _findImageNum(x, images) {
        return images.findIndex(image => this._arraysEqual(image, x)) ?? null;
    }

    _matrixMult(A, B) {
        const rowsA = A.length, colsA = A[0].length, colsB = B[0].length;
        const result = Array.from({ length: rowsA }, () => Array(colsB).fill(0));

        for (let i = 0; i < rowsA; i++) {
            for (let j = 0; j < colsB; j++) {
                for (let k = 0; k < colsA; k++) {
                    result[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return result;
    }

    _scalarMult(scalar, matrix) {
        return matrix.map(row => row.map(value => value * scalar));
    }

    _matrixAdd(A, B) {
        return A.map((row, i) => row.map((val, j) => val + B[i][j]));
    }

    _matrixSub(A, B) {
        return A.map((row, i) => row.map((val, j) => val - B[i][j]));
    }

    _matrixTanh(matrix) {
        return matrix.map(row => row.map(val => Math.tanh(val)));
    }

    _matrixDiff(A, B) {
        return A.reduce((sum, row, i) => sum + row.reduce((rowSum, val, j) => rowSum + Math.abs(val - B[i][j]), 0), 0);
    }

    _matrixMaxAbs(matrix) {
        return Math.max(...matrix.flat().map(Math.abs));
    }

    _transpose(matrix) {
        return matrix[0].map((_, colIndex) => matrix.map(row => row[colIndex]));
    }

    _arraysEqual(arr1, arr2) {
        return arr1.length === arr2.length && arr1.every((val, i) => val === arr2[i]);
    }
}

function preprocessAlphabet(alphabet) {
    return alphabet.map(item => Array.isArray(item[0]) ? item.flat() : item);
}

function imageBeautifulPrint(image, rows, cols) {
    const mapped = image.map(val => (val > 0 ? '⬜' : '⬛')).join('');
    for (let i = 0; i < rows; i++) {
        console.log(mapped.slice(i * cols, (i + 1) * cols));
    }
}

// Example usage
const alphabet = preprocessAlphabet([
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
]);

const network = new Hopfield(alphabet, 0.7);
network.train();

// Пример изображения с шумом для теста
const noisyImage = [1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1];

// Предсказание с использованием сети
const result = network.predict(noisyImage);

// Выводим результат
console.log('Результат работы нейронной сети:');
imageBeautifulPrint(result.newState.flat(), 4, 4);
