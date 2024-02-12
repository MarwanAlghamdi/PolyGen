import re
import torch
import torch_geometric.utils as tgu
import networkx as nx
from .TruncatedNormal import TruncatedNormal

#Setting the device we will be working on
torch.set_default_device('cuda')


# load 3d object and return vertices and faces
def load_obj(filename, force_triangles=False):
    """
    Read a .obj file.
    :param filename: path to the .obj file
    :param force_triangles: if True, force the faces to be triangles.
    :return: tensors: vertices, faces (could be triangles or quads) NOTE: starting from 0
    """
    vertices = []
    faces = []
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            data = line.split()  # every line is a list of words
            if len(data) == 0:
                continue
            if data[0] == 'v':
                to_numeric = [float(x) for x in data[1:]]
                vertices.append(to_numeric)
            elif data[0] == 'f':
                splitted_data = ([int(re.split('/', x)[0]) - 1 for x in data[1:]])
                if force_triangles:
                    if len(splitted_data) == 3:
                        faces.append(splitted_data)
                    elif len(splitted_data) == 4:
                        faces.append([splitted_data[0], splitted_data[1], splitted_data[2]])
                        faces.append([splitted_data[2], splitted_data[3], splitted_data[0]])
                else:
                    faces.append(splitted_data)
            else:
                continue
    return vertices, faces

def center_vertices(vertices: torch.Tensor) -> torch.Tensor:
    vertices = vertices - vertices.mean(0)
    return vertices


def normalize_vertices(vertices: torch.Tensor):
    """
    Normalize the object to the unit sphere.
    :param vertices: vertices of the object
    :return: normalized vertices, shape (n_vertices, 3)
    """
    vertices = vertices / torch.max(torch.norm(vertices, dim=1))
    return vertices


def quantize_vertices(x, n_bits=8):
    """
    Quantize the values of the object.
    :param x: vector or matrix
    :param n_bits: quantization bits
    :return: tensor of quantized values
    """

    n = float(2 ** n_bits)
    x = (x - x.min()) / (x.max() - x.min())
    delta = 1. / n
    quant_vector = torch.clamp((x / delta).floor().to(torch.int32), min=0, max=n - 1)
    return quant_vector


def flatten_faces(faces: list):
    """
    Prepare the face tokens. which flatten the faces and add seperator, start and end tokens.
        EX: example: [1, 2, 3, 4],[5, 6, 7, 8] -> [3, 4, 5,  6, -1,  7, 8, 9, 10, -1 .... -2]
        note: but the last one doesn't have a minus, we would add an end of sequence token [0]
    :param faces: faces
    :return: tensor of face tokens, flattened and added with seperator, start and end tokens
    """

    if not faces:
        return torch.tensor([0], dtype=torch.int32, device='cuda')

    temp = faces
    l = [f + [-1] for f in temp[:-1]]  # Add a separator token at the end of each face
    l += [temp[-1] + [-2]]  # Add an end of sequence token at the end of the last face
    return (torch.Tensor([item for sublist in l for item in sublist]) + 2).to(torch.int32).to('cuda')


def encode_mesh(vertices, faces, n_bits=8, debug=False) -> (torch.Tensor, torch.Tensor):
    """
    Encode the mesh into a graph.
    :param vertices: vertices of the mesh
    :param faces: faces of the mesh
    :param n_bits: quantization bits
    :return: Vertices and faces of the mesh
    """

    if debug:
        print('Number of loaded vertices:', len(vertices))
        print('Number of loaded faces:', len(faces))

    """ Handling Vertices """
    vertices = torch.Tensor(vertices)
    vertices = center_vertices(vertices)
    vertices = normalize_vertices(vertices)
    vertices = quantize_vertices(vertices, n_bits=8)
    vertices, all_vertices_idx = torch.unique(vertices, return_inverse=True, dim=0)
    vertices = vertices.to('cuda')
    vertices_idx_sorted = tgu.lexsort(
        [vertices[:, 0], vertices[:, 1], vertices[:, 2]])  # returns vertices idx sorted by z,y,then x
    vertices = vertices[vertices_idx_sorted]
    if debug:
        print('Number of unique vertices:', len(vertices))

    """ Handling Faces """
    faces_temp = faces
    # Because the faces is a list, we have to loop through each item and then rearrange it to the new vertices indices
    faces_temp = [torch.argsort(vertices_idx_sorted)[all_vertices_idx[f]].tolist() for f in faces_temp]
    faces = [f for f in faces_temp if len(set(f)) >= len(f)]  # remove cyclic faces
    if debug:
        print('Number of unique faces:', len(faces))

    """ Updating Vertices """
    face_tensors = [torch.tensor(f).to('cuda') for f in faces]  # List of tensors
    # connected_vertices = torch.eq(torch.hstack(face_tensors), vertices_idx_sorted[:, None].to('cuda')).any(dim=-1)  # check if, each vertex is connected to any face
    # if debug:
    #     print('Number of connected vertices:', len(connected_vertices))
    #     print('Number of disconnected vertices:', len(vertices) - connected_vertices.sum().item())
    #
    # vertices = vertices[connected_vertices].to(torch.long)
    faces = flatten_faces(faces)

    return vertices, faces


def random_shift(vertices: torch.Tensor, shift_factor: float = 0.1, quantization_bits: int = 8) -> torch.Tensor:
    """
    FIXME: This function is not functioning properly. needs to be fixed.
    Randomly shift the vertices of the object.
    :param vertices: vertices of the object
    :param shift_factor: factor of the shift
    :return: shifted vertices
    """

    # Calculate the maximum possible positive and negative shifts
    max_positive_shift = ((2 ** quantization_bits - 1) - torch.max(vertices, dim=0)[0]).to(torch.float32)
    max_negative_shift = torch.min(vertices, dim=0)[0].to(torch.float32)

    # Ensure the shifts are within the valid range
    max_positive_shift = torch.where(max_positive_shift > 1e-9, max_positive_shift, torch.Tensor([1e-9, 1e-9, 1e-9]))
    max_negative_shift = torch.where(max_negative_shift < -1e-9, max_negative_shift,torch.Tensor([-1e-9, -1e-9, -1e-9]))

    # Create a truncated normal distribution for the random shift
    normal_dist = TruncatedNormal(
        loc=torch.zeros(1, 3),
        scale=shift_factor * (2 ** quantization_bits - 1),
        a=max_negative_shift,
        b=max_positive_shift,
    )

    # Generate a random shift for each vertex and apply it
    shift = normal_dist.rsample(sample_shape=(len(vertices), 3)).to(torch.float32)
    vertices = vertices + shift
    return vertices
