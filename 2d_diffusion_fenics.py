"""
To solve 2D diffusion equation on the rectangular domain with a constant source f using Fenics library.
The goal is to compare the results against PINNs solution for the exact same problem.

Domain: x = [0, 1]
PDE: Diff * Laplacian(u) = f_source -> Laplacian(u) = d2u/dx2 + d2u/dy2
BCs: Dirichlet on left and right / Homogeneous Neumann on top and bottom
- left:   u(x=0) = C_BC1 = 0.2
- right:  u(x=1) = C_BC2 = 1
- bottom: du/dy(y=0) = 0
- top:	  du/dy(y=0.5) = 0

Params: D= 0.1 / f(x,y)= -1

Author: Rojin Anbarafshan
Contact: rojin.anbar@gmail.com
Date: June 2025
"""

# Copyright (C) 2007-2011 Anders Logg
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#
# First added:  2007-08-16
# Last changed: 2012-11-12

from dolfin import *

# Create mesh and define function space
p_bottom_left = Point(0.0, 0.0)
p_top_right   = Point(1, 0.5)
mesh = RectangleMesh(p_bottom_left, p_top_right, 32, 32)

V = FunctionSpace(mesh, "Lagrange", 1)

# Define Dirichlet boundary (x = 0 or x = 1)
def boundary_left(x):
    return x[0] < DOLFIN_EPS

def boundary_right(x):
    return x[0] > 1.0 - DOLFIN_EPS

# Define boundary condition
u_left  = Constant(0.2)
u_right = Constant(1.0)

bc_left  = DirichletBC(V, u_left, boundary_left)
bc_right = DirichletBC(V, u_right, boundary_right)

bc = [bc_left, bc_right]

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Expression("-10.0")
g = Expression("0.0")
a = inner(grad(u), grad(v))*dx
L = f*v*dx + g*v*ds

# Compute solution
u = Function(V)
solve(a == L, u, bc)

# Save solution in VTK format
file = File("poisson.pvd")
file << u

# Plot solution
# plot(u, interactive=True)
