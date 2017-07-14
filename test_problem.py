# Copyright (C) 2007 Kristian B. Oelgaard
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
# Modified by Anders Logg, 2008
# Modified by Johan Hake, 2008
# Modified by Garth N. Wells, 2009
#
# This demo solves the time-dependent convection-diffusion equation by
# a SUPG stabilized method. The velocity field used in the simulation
# is the output from the Stokes (Taylor-Hood) demo.  The sub domains
# for the different boundary conditions are computed by the demo
# program in src/demo/subdomains.


from dolfin import *
from fenics import *
import mshr
# tol = 1E-3
# def boundary_value(n):
#     if n < 5:
#         return 1E-3*float(n)/10.0
#     else:
#        return 1.0*1E-3
# def boundary_value(x, on_boundary):
# return on_boundary and (near(x[0], 0, tol) or near(x[0], 1, tol))


h0 = 0.02
w0 = 0.05
domain = mshr.Rectangle(Point(0,0), Point(w0,h0))
mesh = mshr.generate_mesh(domain, 50)
h = CellSize(mesh)

# Create FunctionSpaces
Q = FunctionSpace(mesh, "CG", 1)
V = VectorFunctionSpace(mesh, "CG", 2)

# Create velocity Function from file
velocity = Function(V);
E = Constant((-300.0, 0))
p = -1
K = 5.9278*1E-12
velocity = project(E*K*p, V)
# print -3*K*p
# velocity.interpolate(Constant((.01, 0.0)))

# Initialise source function and previous solution function
f  = Constant(0.0)


u0 = Function(Q)

# u0.assign(Constant(2000.0))

# u_0 = Constant(2000.)
 # u_0 = Constant(boundary_value(0))
u_0 = Expression('2000*exp(-100*(x[0]))', degree=1)


# u_0 = 2000
# class InitialCondition(Expression):
#     def eval_cell(self, value, x, ufc_cell):
#         if x[0] <= 0.01:
#             value[0] = 2000.
#         else:
#             value[0] = 0.0
#
# u0.interpolate(InitialCondition())
# plot(u0, interactive=True)
u0 = interpolate(u_0, Q)

# u0 = Constant(boundary_value(0))

plot(u0)

# Parameters
T = 3600*50
dt = 360
t = dt
c = 2.0e-9


# boundary1 = CompiledSubDomain("near(x[1], 0) && (x[0] < 0.1*value) && on_boundary", value=w0)
# boundary2 = CompiledSubDomain("near(x[0], 0) && (x[1] < 0.1*value) && on_boundary", value=h0)
boundary1 = CompiledSubDomain("near(x[0], 0) && on_boundary", value=w0)

mf = MeshFunction("size_t", mesh, 1)

mf.set_all(0)
boundary1.mark(mf, 1)
# boundary2.mark(mf, 2)

# plot(mf, interactive=True)

# ds = Measure('ds', domain = mesh, subdomain_data = mf)
# Test and trial functions
u, v = TrialFunction(Q), TestFunction(Q)

# Mid-point solution
u_mid = 0.5*(u0 + u)

# Residual
r = u - u0 + dt*(dot(velocity, grad(u_mid)) - c*div(grad(u_mid)) - f)

# Galerkin variational problem
F = v*(u - u0)*dx + dt*(v*dot(velocity, grad(u_mid))*dx \
                      + c*dot(grad(v), grad(u_mid))*dx)#- c*g*v*ds

# # Add SUPG stabilisation terms
# vnorm = sqrt(dot(velocity, velocity))
# F += (h/(2.0*vnorm))*dot(velocity, grad(v))*r*dx

# Create bilinear and linear forms
a = lhs(F)
L = rhs(F)

# Set up boundary condition
# g = Constant(2)
# bc1 = DirichletBC(Q, g, boundary1)
# bc2 = DirichletBC(Q, g, boundary2)

# Assemble matrix
A = assemble(a)
# bc1.apply(A)
# bc2.apply(A)

# Create linear solver and factorize matrix
solver = LUSolver(A)
solver.parameters["reuse_factorization"] = True

# Output file
# out_file = File("results/solution.pvd")
vtkfile = File('Concentration/solution.pvd')

# Set intial condition
u = Function(Q)
u = u0


# b = assemble(L)
# solver.solve(u.vector(), b)

# plot(u0, mode='color')
# Time-stepping
while t < T:

    # Assemble vector and apply boundary conditions
    b = assemble(L)
    # bc.apply(b)
    # bc1.apply(b)
    # bc2.apply(b)

    # Solve the linear system (re-use the already factorized matrix A)
    solver.solve(u.vector(), b)

    # Copy solution from previous interval
    u0 = u
    # print u

    # Plot solution
    # u0.assign(u)
    plot(u)

    # Save the solution to file
    # out_file << (u, t)

    vtkfile << u



    # Move to next interval and adjust boundary condition
    t += dt


    # vtkfile = File('Concentration/solution.pvd')
    # vtkfile << u
    # out_file << (u, t)



# Hold plot
# plot(u)
interactive()
