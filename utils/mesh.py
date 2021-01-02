import numpy as np


def get_mesh(size, mesh_size, start=0):
    """
    :param size: final size [width, height]
    :param mesh_size: # of mesh
    :param start: default 0
    :return:
    """
    w, h = size
    x = np.linspace(start, w, mesh_size)
    y = np.linspace(start, h, mesh_size)

    return np.stack([x, y], axis=0)


def get_vertice(size, mesh_size, offsets):
    """
    :param size: final size [width, height]
    :param mesh_size: # of mesh
    :param offsets: [offset_x, offset_y]
    :return:
    """
    w, h = size
    x = np.linspace(0, w, mesh_size)
    y = np.linspace(0, h, mesh_size)
    next_x = x + w / (mesh_size * 2)
    next_y = y + h / (mesh_size * 2)
    next_x, next_y = np.meshgrid(next_x, next_y)
    vertices = np.stack([next_x, next_y], axis=-1)
    vertices -= np.array(offsets)

    return vertices
