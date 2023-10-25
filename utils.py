import numpy as np


def random_index_generator(count):
    """生成一系列随机的、不重复的索引值,范围是 0 到 count - 1"""
    indices = np.arange(0, count)
    print(indices)
    np.random.shuffle(indices)
    print(indices)

    for idx in indices:
        yield idx



