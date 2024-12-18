const fs = require('fs');
const path = require('path');
const math = require('mathjs');

class Recognition {
    constructor(directoryPath, noisyPath) {
        this.templates = [];
        this.loadTemplates(directoryPath);

        const inputVector = this.getVectorFromFile(noisyPath);
        this.X = math.matrix(math.reshape(inputVector, [inputVector.length, 1])); // Преобразовать в матрицу
        console.log("Input vector:");
        console.log(this.X.toString());
        console.log("Image to recognize:");
        this.beautifulVisualization(this.X);

        this.N = this.X.size()[0]; // Длина вектора X
        console.log(this.N);
        this.W = math.zeros(this.N, this.N); // Квадратная матрица весов

        this.showImages();
        console.log('here');
        this.calculateWeights();
        this.recognition();
    }

    loadTemplates(directoryPath) {
        const files = fs.readdirSync(directoryPath);
        files.forEach(file => {
            const filePath = path.join(directoryPath, file);
            if (path.extname(file) === '.txt') {
                const vector = this.getVectorFromFile(filePath);
                const reshapedVector = math.matrix(math.reshape(vector, [vector.length, 1])); // Вектор-столбец
                this.templates.push(reshapedVector);
            }
        });
    }

    getVectorFromFile(filePath) {
        const content = fs.readFileSync(filePath, 'utf-8');
        return content.trim().split(/\s+/).map(Number);
    }

    calculateWeights() {
        const epsilon = 1e-9;
        const h = 0.8;
        let change;
        let iterations = 0;
        const changes = [];

        do {
            iterations++;
            change = 0;

            this.templates.forEach(image => {
                const Xi = math.clone(image); // Убедиться, что это вектор-столбец
                console.log('Shape of Xi:', math.size(Xi));       // [N, 1]
                console.log('Shape of W:', math.size(this.W));   // [N, N]
                console.log('Shape of W * Xi:', math.size(math.multiply(this.W, Xi))); // [N, 1]
                console.log('-----------------------------------');

                const deltaW = math.multiply(
                    math.multiply(math.subtract(Xi, math.multiply(this.W, Xi)), math.transpose(Xi)), // (Xi - W * Xi) * Xi^T
                    h / this.N
                );

                this.W = math.add(this.W, deltaW); // Обновить W
                change += math.sum(math.abs(deltaW)); // Сумма абсолютных изменений
            });

            changes.push(change);
            if (changes.length > 1) {
                change = Math.abs(change - changes[changes.length - 2]);
            }
        } while (change > epsilon);

        console.log(`Iteration: ${iterations}`);
    }

    recognition() {
        let relaxation = false;
        let prev;
        this.generateRandomIndexes();
        let recIterations = 0;

        do {
            this.doIteration();
            if (math.deepEqual(prev, this.X)) {
                relaxation = true;
            } else {
                prev = math.clone(this.X);
            }
            recIterations++;
        } while (!relaxation);

        this.showAnswer(recIterations);
    }

    doIteration() {
        let changed = 0;

        while (changed < this.N) {
            const index = this.getRandomIndex(changed + 1);
            let newXi = 0;

            for (let i = 0; i < this.N; i++) {
                newXi += this.X.get([i, 0]) * this.W.get([index, i]); // Работает с матрицей
            }

            this.X.set([index, 0], newXi); // Обновление элемента в матрице X
            this.activationFunction();
            changed++;
        }
    }

    activationFunction() {
        this.X = this.X.map(x => (x > 0 ? 1 : 0)); // Пример функции активации
    }

    beautifulVisualization(matrix) {
        console.log(matrix.toString()); // Визуализация матрицы
    }

    showAnswer(iterations) {
        console.log("You got a static attractor!");
        console.log(`After ${iterations} iteration(s), the recognized image is:`);
        this.beautifulVisualization(this.X);
        console.log("Output vector is:");
        console.log(this.X.toString());
    }

    showImages() {
        console.log("Images-templates are:");
        this.templates.forEach(image => {
            this.beautifulVisualization(image);
            console.log("");
        });
    }

    generateRandomIndexes() {
        this.randomIndexes = Array.from({ length: this.N }, (_, i) => i);
        for (let i = this.randomIndexes.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [this.randomIndexes[i], this.randomIndexes[j]] = [this.randomIndexes[j], this.randomIndexes[i]];
        }
    }

    getRandomIndex(i) {
        return this.randomIndexes[this.randomIndexes.length - i];
    }
}

// Пример вызова
function main() {
    const recognition = new Recognition('./templates', './noisy/noisy4.txt');
}

main();
