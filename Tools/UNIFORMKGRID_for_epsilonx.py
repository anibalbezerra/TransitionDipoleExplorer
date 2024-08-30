#! pip install ase

from ase import Atoms
from ase.build import bulk
import ase.spacegroup as sg

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_reciprocal_cell(atoms_list, positions_list, spacegroup, cell_parameters):

    system = sg.crystal(symbols=atoms_list, \
                        basis=positions_list, \
                        spacegroup=spacegroup,\
                        cellpar=cell_parameters)
    print(f'System cell:\n\t {system.cell}')
    system_BZ = system.get_reciprocal_cell()
    system_BZ.bravais = system_BZ.get_bravais_lattice()
    print(f'System reciprocal lattice symmetry:\n\t {system_BZ.bravais}')

    a_star, b_star, c_star = system_BZ.array

    a_star = a_star / np.linalg.norm(a_star)
    b_star = b_star / np.linalg.norm(b_star)
    c_star = c_star / np.linalg.norm(c_star)

    print(f'System reciprocal lattice vectors:\n \t a*={a_star}\n\t b*={b_star}\n\t c*={c_star}')

    return a_star, b_star, c_star

def generate_kpoints(nk, a_star, b_star, c_star, shift=0.0):
    """
    Generates a uniform K-point grid for a lattice given by a_star, b_star, c_star.

    Args:
        nk: The number of K-points along each reciprocal lattice vector.
        shift: A shift applied to the K-point grid.

    Returns:
        A list of K-point coordinates.
    """
    # Reciprocal lattice vectors for FCC
    a_star = np.array(a_star)
    b_star = np.array(b_star)
    c_star = np.array(c_star)

    # K-point grid
    kpoints = []
    for i in range(nk):
        for j in range(nk):
            for k in range(nk):
                kpoint = (i + 0.5) / nk * a_star + (j + 0.5) / nk * b_star + (k + 0.5) / nk * c_star
                kpoints.append(kpoint + shift)

    weigths = 1 / nk**3

    return kpoints, weigths

def plot_k_mesh(kpoints):
    # Extract the x, y, and z coordinates
    x = [point[0] for point in kpoints]
    y = [point[1] for point in kpoints]
    z = [point[2] for point in kpoints]

    # Create the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points
    ax.scatter(x, y, z)

    # Set labels and title
    ax.set_xlabel('kx')
    ax.set_ylabel('ky')
    ax.set_zlabel('kz')
    plt.title('K-points in 3D')

    # Show the plot
    plt.show()

# ZnO example
atoms_list=('Zn', 'O')
positions_list=[(2/3,1/3,0.5),(2/3,1/3,0.87)]
spacegroup=186
cell_parameters=[3.24,3.24,5.22,90,90,120]

a_star, b_star, c_star = get_reciprocal_cell(atoms_list=atoms_list, \
                                             positions_list=positions_list,\
                                             spacegroup=spacegroup, \
                                             cell_parameters=cell_parameters)

# GaAs example
atoms_list=('Ga', 'As')
positions_list=[(0,0,0),(1/4,1/4,3/4)]
spacegroup=216
cell_parameters=[5.75,5.75,5.75,90,90,90]

a_star, b_star, c_star = get_reciprocal_cell(atoms_list=atoms_list, \
                                             positions_list=positions_list,\
                                             spacegroup=spacegroup, \
                                             cell_parameters=cell_parameters)

nk = 10  # Number of K-points along each reciprocal lattice vector

kpoints, weigths = generate_kpoints(nk, a_star, b_star, c_star, shift=0.0)

printPoints = True
if printPoints:
    # Print the K-points
    for ik, kpoint in enumerate(kpoints):
        print(f"{kpoint[0]:.2f} {kpoint[1]:.2f} {kpoint[2]:.2f} {weigths:.3f}")