# IPC implementation in Taichi

IPC (Incremental Potential Contact: Intersection and Inversion-free Large Deformation Dynamics. https://ipc-sim.github.io/) is a fantastic way to do simulation without violating any constraints. This is a tiny 2D version of IPC in taichi.

1. Most of the code is compatible with 3D. But ccd and the grad and hessians of the barrier energy need further modification.
2. For comparison, I also implement FEM without barrier energy in explicit way and implicit way.
3. Implicit ways include one-step updating method and energy optimization method. (setting iteration for 1 in the latter equals to the former one)
4. I include two kinds of linear solver: direct solver provided by numpy.linalg and conjugate gradient solver using taichi.
5. Note that the triangle structure in the model should be counter-clockwise so that the barrier energy can be calculated correctly.
6. Gravity is applied as the external force instead of energy, so U2 is set to 0, you can remove the external force and use U2 as the gravitational energy

# Todo list

-   TODO: grad and hessian of the barrier energy
-   TODO: accurate ccd calculation (current ccd is not right, but can work in simple cases)
-   TODO: Turing the hessian into semi-definite before conjugate gradient
-   TODO: preconditioned conjugate gradient
-   TODO: add 3D support and cloth simulation

Code structure refers to https://github.com/XunZhan/IPC-2D-Game (note that the algorithm is not correct in that project.)
