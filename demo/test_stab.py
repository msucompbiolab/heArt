from dolfin import *
import numpy as np
from scipy.sparse import csr_matrix

# Create an empty mesh
mesh = Mesh()

# Initialize the mesh editor
editor = MeshEditor()
editor.open(mesh, "interval", 1, 1)  # 1D mesh, 1D topology

# Specify the number of vertices and cells
num_vertices = 3
num_cells = 2
editor.init_vertices(num_vertices)
editor.init_cells(num_cells)

# Define vertices, here 0.0 is the start, 1.0 is the end, and 1/3 is the middle point
editor.add_vertex(0, [0.0])  # Vertex 0 at position 0.0
editor.add_vertex(1, [1.0 / 3])  # Vertex 1 at position 1/3
editor.add_vertex(2, [1.0])  # Vertex 2 at position 1.0

# Define cells (intervals), which are lines between consecutive vertices
editor.add_cell(0, [0, 1])  # Cell 0 between vertex 0 and 1
editor.add_cell(1, [1, 2])  # Cell 1 between vertex 1 and 2

# Close the editor to finalize the mesh
editor.close()

cell_vol = CellDiameter(mesh)

F = FunctionSpace(mesh, "CG", 1)
p = TrialFunction(F)
q = TestFunction(F)

# Fs = 1.0 / (CellDiameter(mesh)**(1.0/3.0)) * (p - p / CellDiameter(mesh)) * (q - q / CellDiameter(mesh)) * dx
# Fs = 1.0 / (CellVolume(mesh)**(1.0/3.0)) * (p - p / CellVolume(mesh)) * (q - q / CellDiameter(mesh)) * dx
# Fs = (p - p / CellVolume(mesh)) * (q - q / CellVolume(mesh)) * dx

# Fs = (1/CellVolume(mesh))**2 * p * q * dx
Fs = (1/CellVolume(mesh)) * p * q * dx
# Fs =  p * q * dx

Fs_assem = assemble(Fs)
petsc_a = as_backend_type(Fs_assem).mat()
# Fs_arr = Fs_assem.get_local()
ai, aj, av = petsc_a.getValuesCSR()
Asp = csr_matrix((av, aj, ai), shape=(F.dim(), F.dim()))

# Verify by converting to a dense array
print("Dense matrix representation:\n", Asp.toarray())

Fs_arr = np.array(Fs_assem)

print("Fs arr = ", Fs_arr)


# Plot the mesh
# import matplotlib.pyplot as plt
# plot(mesh)
# plt.show()
# projection_operator = 1.0/CellVolume(mesh) * q * dx

f_p = Function(F)
f_p.vector()[:] = 0.5


# projection_operator = (p - p/CellVolume(mesh)) * (q - q/CellVolume(mesh)) * dx
# projection_operator = (1/CellVolume(mesh))**2 * p * q * dx
projection_operator = (1/CellVolume(mesh)) * p * q * dx
# projection_operator = p * q * dx
cells = [Cell(mesh, i) for i in range(2)]
assem_projection = [assemble_local(projection_operator, cell) for cell in cells]
print("assem_projection = ", assem_projection)

# pi operator is defined element-wise, so
pi_oper = 1/CellVolume(mesh) * p * dx(cell)
qi_oper = 1/CellVolume(mesh) * p * dx(cell)

form_oper = (p - pi_oper) * (q - qi_oper) * dx
assem_form_oper = assemble(form_oper)

