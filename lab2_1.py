import numpy as np
# from numbers import Numbers
from pprint import pprint


def preprocess_alphabet(alphabet):
    """
    Преобразует двумерные элементы массива в одномерные.
    """
    processed_alphabet = []
    for item in alphabet:
        if isinstance(item, np.ndarray) and item.ndim == 2:
            processed_alphabet.append(item.ravel())
        else:
            processed_alphabet.append(item)
    return np.array(processed_alphabet)

class Hopfield:
    def __init__(self, images, nu: float = 1.0) -> None:
        self.size = images.shape[1]
        self.w = np.zeros((self.size, self.size))
        self.images = images
        self.neg_images = self._get_neg_images(self.images)
        self.nu = nu

    def _get_neg_images(self, images):
        return images * -1

    def train(self, e=1e-6, max_iters=10000):
        """Алгоритм дельта-проекций с гиперболическим тангенсом."""
        for _ in range(max_iters):
            old_w = self.w.copy()

            for image in self.images:
                x_t = np.matrix(image.copy()).T
                activation = np.tanh(self.w @ x_t)


                self.w += (self.nu / self.size) * (x_t - activation) @ x_t.T
                np.fill_diagonal(self.w, 0)  # Диагональные элементы равны нулю

            # Условие сходимости
            if np.abs(old_w - self.w).sum() < e:
                break

        np.fill_diagonal(self.w, 0)

    def _find_image_num(self, x, images) -> int | None:
        """Ищет, соответствует ли состояние эталону."""
        # mask = (images == x).all(axis=1)
        # search_result = np.where(mask)[0]
        # if len(search_result) > 0:
        #     return search_result.item()
        # return None
        for idx, image in enumerate(images):
            if np.abs(image - x).max() < 1e-2:
                return idx
        return None

    def predict(self, x, max_iters: int = 1000):
        """Рассчитывает новое состояние сети с гиперболическим тангенсом."""
        states = [np.matrix(x.copy())] * 4
        relaxation_iters = 0

        for _ in range(max_iters):
            relaxation_iters += 1

            # Рассчёт нового состояния
            new_state = np.tanh(self.w @ states[-1].T).T
            states.append(new_state)
            states.pop(0)

            # Проверка сходимости
            if _ >= 3 and np.abs(states[0] - states[2]).max() < 1e-8 and np.abs(states[1] - states[3]).max() < 1e-8:
                image_num = self._find_image_num(new_state, self.images)
                neg_image_num = self._find_image_num(new_state, self.neg_images)
                is_negative = neg_image_num is not None

                return (relaxation_iters, new_state,
                        (image_num if image_num is not None else neg_image_num),
                        is_negative)

        return max_iters, new_state, None, None 
    
alphabet = np.array([
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
])

        # [-1, 1, 1, -1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1],
        # [1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1],
        # [-1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1],
        # # [-1, 1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, -1, 1, 1, 1],    

alphabet = preprocess_alphabet(alphabet)
network = Hopfield(alphabet, 0.7)
network.train()

def image_beautiful_print(image, rows, cols):
    image = np.sign(image)
    image = image.astype(np.object_)
    image[image == 1] = '⬜'
    image[image == -1] = '⬛'
    image = image.reshape(rows, cols)
    image_list = image.tolist()
    for row in image_list:
        print(''.join(row))

# test_image = np.array([1, -1, 1, -1, 1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1])
test_image = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1])

r_iters, state, image_idx, is_negative = network.predict(test_image, 10_000)
if image_idx:
    predicted_img = network.neg_images[image_idx] if is_negative else network.images[image_idx]
else:
    predicted_img = state
print(f'Prediction success? - {True if image_idx is not None else False}')
print(f'Image number (index begins from 0): {image_idx}.\n')
print(f'Image: {predicted_img}.\nIters for relax: {r_iters}')
print(f'Is this image a negative? - {is_negative}')
print('ORIGINAL IMAGE')
image_beautiful_print(test_image, 4, 4)
print('PREDICTED IMAGE')
image_beautiful_print(predicted_img, 4, 4)
print('NEGATIVE OF PREDICTED IMAGE')
image_beautiful_print(predicted_img * (-1), 4, 4)