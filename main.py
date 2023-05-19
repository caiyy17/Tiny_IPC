import time
import numpy as np
import taichi as ti
from random import choice

ti.init(arch=ti.cuda, kernel_profiler=True, device_memory_GB=4)

bg_color = 0x3A1B19
line_color = 0xC7A085
text_color = 0xFCF0E1
line_radius = 1.5
dt = 1 / 60 / 1
stepDisplayThreshold = 0.2
iterDisplayThreshold = 5

dim = 2
screen_res = (400, 400)
screen_to_world_ratio = 40.0  # 10m * 10m
boundary = [[0, screen_res[0] / screen_to_world_ratio],
            [0, screen_res[1] / screen_to_world_ratio]]

model_dir = "model"


class Object:

    def __init__(self, pos, filename):
        self.pos = pos
        self.dim = dim
        with open(model_dir + "/" + filename + ".node", "r") as file:
            self.vn = int(file.readline().split()[0])
            self.node = np.zeros([self.vn, self.dim])
            for i in range(self.vn):
                tmp = np.array([
                    float(x) for x in file.readline().split()[1:self.dim + 1]
                ])
                self.node[i] = tmp + self.pos
        with open(model_dir + "/" + filename + ".ele", "r") as file:
            self.en = int(file.readline().split()[0])
            self.element = np.zeros([self.en, self.dim + 1]).astype(np.int32)
            for i in range(self.en):
                tmp = np.array([
                    int(ind) for ind in file.readline().split()[1:self.dim + 2]
                ])
                self.element[i] = tmp
        self.begin_point = np.zeros([self.en * (self.dim + 1), self.dim])
        self.end_point = np.zeros([self.en * (self.dim + 1), self.dim])
        for i in range(self.en):
            for j in range(self.dim + 1):
                self.begin_point[i * (self.dim + 1) +
                                 j] = self.node[self.element[i][j]]
                self.end_point[i * (self.dim + 1) +
                               j] = self.node[self.element[i][(j + 1) %
                                                              (self.dim + 1)]]
        self.begin_point = self.begin_point * screen_to_world_ratio / screen_res
        self.end_point = self.end_point * screen_to_world_ratio / screen_res


@ti.data_oriented
class FEM:

    def __init__(self):
        self.dim = ti.static(dim)
        self.epsilon = ti.static(1e-5)

        self.on = ti.static(1000)
        self.vn = ti.static(10000)
        self.en = ti.static(10000)
        self.node = ti.Vector.field(self.dim,
                                    dtype=ti.f32,
                                    shape=self.vn,
                                    needs_grad=True)
        self.prev_node = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.vn)
        self.prev_t_node = ti.Vector.field(self.dim,
                                           dtype=ti.f32,
                                           shape=self.vn)
        self.bar_node = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.vn)
        self.element = ti.Vector.field(self.dim + 1,
                                       dtype=ti.i32,
                                       shape=self.en)

        #  the end index of i's object
        self.vn_object_index = ti.field(dtype=ti.i32, shape=self.on)
        self.en_object_index = ti.field(dtype=ti.i32, shape=self.on)
        self.count = ti.field(dtype=ti.i32, shape=())

        # for simulation
        self.damping = 1
        self.mass = 1
        self.E = 6000  # Young modulus
        self.nu = 0.4  # Poisson's ratio: nu \in [0, 0.5)
        self.mu = self.E / (2 * (1 + self.nu))
        self.la = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        self.dt = dt
        self.bar_d = 0.1
        self.k = 1  # contact stiffness

        self.velocity = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.vn)
        self.node_mass = ti.field(dtype=ti.f32, shape=self.vn)
        self.element_volume = ti.field(dtype=ti.f32, shape=self.en)
        self.energy = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
        self.prev_energy = ti.field(dtype=ti.f32, shape=())
        self.B = ti.Matrix.field(self.dim,
                                 self.dim,
                                 dtype=ti.f32,
                                 shape=self.en)
        # Hessians
        self.A = ti.field(dtype=ti.f32,
                          shape=(self.vn * self.dim, self.vn * self.dim))
        self.b = ti.field(dtype=ti.f32, shape=(self.vn * self.dim))
        self.r = ti.field(dtype=ti.f32, shape=(self.vn * self.dim))
        self.d = ti.field(dtype=ti.f32, shape=(self.vn * self.dim))
        self.flat_v = ti.field(dtype=ti.f32, shape=(self.vn * self.dim))
        self.delta_x = ti.field(dtype=ti.f32, shape=(self.vn * self.dim))
        self.F_mul_ans = ti.field(dtype=ti.f32, shape=(self.vn * self.dim))
        # render
        self.begin_point = ti.Vector.field(self.dim,
                                           ti.f32,
                                           shape=(self.en * 3))
        self.end_point = ti.Vector.field(self.dim, ti.f32, shape=(self.en * 3))
        self.en_flat = ti.field(dtype=ti.i32, shape=(self.en * 6))
        self.ball_pos = ti.Vector([5, 0])
        self.ball_r = 3.2

    @ti.kernel
    def create_lines(self):
        for i in range(self.en_object_index[self.count[None]]):
            for j in ti.static(range(self.dim + 1)):
                self.begin_point[i * (self.dim + 1) + j] = self.node[
                    self.element[i][j]] * screen_to_world_ratio / screen_res
                self.end_point[
                    i * (self.dim + 1) + j] = self.node[self.element[i][
                        (j + 1) %
                        (self.dim + 1)]] * screen_to_world_ratio / screen_res
                # self.en_flat[i * (self.dim + 1) * 2 +
                #              j * 2] = self.element[i][j]
                # self.en_flat[i * (self.dim + 1) * 2 + j * 2 +
                #              1] = self.element[i][(j + 1) % (self.dim + 1)]

    @ti.kernel
    def add_obj(
            self,
            vn: ti.i32,
            en: ti.i32,
            node: ti.types.ndarray(ti.f32, ndim=2),
            element: ti.types.ndarray(ti.f32, ndim=2),
    ):
        # update vn
        for i in range(vn):
            self.node[self.vn_object_index[self.count[None]] +
                      i] = [node[i, 0], node[i, 1]]
            self.prev_node[self.vn_object_index[self.count[None]] +
                           i] = [node[i, 0], node[i, 1]]
            self.prev_t_node[self.vn_object_index[self.count[None]] +
                             i] = [node[i, 0], node[i, 1]]
            self.bar_node[self.vn_object_index[self.count[None]] +
                          i] = [node[i, 0], node[i, 1]]
        # update en
        for i in range(en):
            for j in ti.static(range(self.dim + 1)):
                self.element[self.en_object_index[self.count[None]] +
                             i][j] = self.vn_object_index[self.count[None]] + \
                             element[i, j]
        # update final index
        self.vn_object_index[self.count[None] +
                             1] = self.vn_object_index[self.count[None]] + vn
        self.en_object_index[self.count[None] +
                             1] = self.en_object_index[self.count[None]] + en
        self.count[None] += 1

        volumeScale = 2
        if self.dim == 2:
            volumeScale = 2
        elif self.dim == 3:
            volumeScale = 6

        for i in range(self.en_object_index[self.count[None] - 1],
                       self.en_object_index[self.count[None]]):
            D = self.D(i)
            self.B[i] = D.inverse()
            self.element_volume[i] = ti.abs(D.determinant()) / volumeScale
        for i in range(self.vn_object_index[self.count[None] - 1],
                       self.vn_object_index[self.count[None]]):
            self.node_mass[i] = self.mass

    @ti.func
    def D(self, idx):
        a = self.element[idx][self.dim]
        mat = ti.Matrix.zero(ti.f32, self.dim, self.dim)
        for i in ti.static(range(self.dim)):
            col = self.node[self.element[idx][i]] - self.node[a]
            for j in ti.static(range(self.dim)):
                mat[j, i] = col[j]
        return mat

    @ti.func
    def F(self, i):  # deformation gradient
        return self.D(i) @ self.B[i]

    @ti.func
    def P(self, i):  # first Piola-Kirchhoff stress tensor
        # E = mu / 2 * (F^T * F - I) - mu * log(J) + la / 2 * log(J)^2
        # P = dE/dF = mu * (F - F^-T) + la * log(J) * F^-T
        # F = D * B
        # dE/dxjk = PB^T_kj
        F = self.F(i)
        rF = F.inverse()
        J = ti.max(F.determinant(), 0.01)
        P = self.mu * (F -
                       rF.transpose()) + self.la * ti.log(J) * rF.transpose()
        return P

    @ti.func
    def Psi(self, i):  # (strain) energy density
        # E = mu / 2 * (F^T * F - I) - mu * log(J) + la / 2 * log(J)^2
        F = self.F(i)
        log_J = ti.log(ti.max(F.determinant(), 0.01))
        phi = self.mu / 2 * ((F.transpose() @ F).trace() - self.dim)
        phi -= self.mu * log_J
        phi += self.la / 2 * log_J**2
        return phi

    @ti.func
    def U0(self, i):  # elastic potential energy for element
        return self.element_volume[i] * self.Psi(i)

    @ti.func
    def U1(self, i):  # gravitational potential energy E = mgh for node
        return 0
        return self.node_mass[i] * 10 * self.node[i].y

    @ti.func
    def U2(self, i):  # inertia energy E = 1/2 * (x_bar - x) * M * (x_bar - x)
        return 0.5 * (
            (self.bar_node[i] - self.node[i]).norm_sqr()) * self.node_mass[i]

    @ti.func
    def grad(self, i):
        # grad = wi * dE/dx = wi * dE/dF * dF/dx = wi * P * B^T
        P = self.P(i)
        dE = self.element_volume[i] * P @ self.B[i].transpose()
        return dE

    @ti.func
    def hassian(self, c, u, d):
        B_c = self.B[c]
        F = self.F(c)
        rF = F.inverse()
        J = ti.max(F.determinant(), 0.01)
        dD = ti.Matrix.zero(ti.f32, self.dim, self.dim)
        if u == self.dim:
            for j in ti.static(range(self.dim)):
                dD[d, j] = -1
        else:
            dD[d, u] = 1
        dF = dD @ B_c
        dP = self.mu * dF
        dP += (self.mu - self.la * ti.log(J)) * (
            rF.transpose() @ dF.transpose()) @ rF.transpose()
        dP += self.la * (rF @ dF).trace() * rF.transpose()
        ddE = self.element_volume[c] * dP @ B_c.transpose()
        return ddE

    @ti.kernel
    def reset_energy(self):
        self.energy[None] = 0

    @ti.kernel
    def save_energy(self):
        self.prev_energy[None] = self.energy[None]

    @ti.kernel
    def compute_energy(self):
        for i in range(self.en_object_index[self.count[None]]):
            self.energy[None] += self.U0(i)
        for i in range(self.vn_object_index[self.count[None]]):
            self.energy[None] += self.U1(i)

    @ti.kernel
    def compute_energy_with_inertia(self):
        for i in range(self.en_object_index[self.count[None]]):
            self.energy[None] += self.U0(i) * self.dt * self.dt
        for i in range(self.vn_object_index[self.count[None]]):
            self.energy[None] += self.U1(i) * self.dt * self.dt
            self.energy[None] += self.U2(i)

    @ti.kernel
    def handle_boundary(self):
        for i in range(self.vn_object_index[self.count[None]]):
            # ball boundary condition:
            disp = self.node[i] - self.ball_pos
            disp2 = disp.norm_sqr()
            if disp2 <= self.ball_r**2:
                self.node[i] = self.ball_pos + self.ball_r * disp / disp.norm()
                NoV = self.velocity[i].dot(disp)
                if NoV < 0:
                    self.velocity[i] -= NoV * disp / disp2
            # rect boundary condition:
            for j in ti.static(range(self.dim)):
                if self.node[i][j] < 0:
                    self.node[i][j] = 0
                    if (self.velocity[i][j] < 0):
                        self.velocity[i][j] = 0
                elif self.node[i][j] > screen_res[0] / screen_to_world_ratio:
                    self.node[i][j] = screen_res[0] / screen_to_world_ratio
                    if (self.velocity[i][j] > 0):
                        self.velocity[i][j] = 0

    @ti.kernel
    def update_explicit_auto_grad(self):
        for i in range(self.vn_object_index[self.count[None]]):
            acc = -self.node.grad[i] / self.node_mass[i]
            self.velocity[i] += acc * self.dt
            self.velocity[i] += 9.8 * self.dt * ti.Vector([0, -1])
            self.velocity[i] *= ti.exp(-self.dt * self.damping)
            self.node[i] += self.dt * self.velocity[i]

    def step_explicit_auto_grad(self):
        self.reset_energy()
        with ti.ad.Tape(self.energy):
            self.compute_energy()
        self.update_explicit_auto_grad()
        self.handle_boundary()

    @ti.kernel
    def update_explicit_with_grad(self):
        for i in range(self.en_object_index[self.count[None]]):
            H = -self.grad(i)
            for j in ti.static(range(self.dim)):
                force = ti.Vector(
                    [H[k, j] for k in ti.static(range(self.dim))])
                self.velocity[self.element[i][
                    self.dim]] -= force / self.node_mass[self.element[i][
                        self.dim]] * self.dt
                self.velocity[self.element[i][j]] += force / self.node_mass[
                    self.element[i][j]] * self.dt

        for i in range(self.vn_object_index[self.count[None]]):
            self.velocity[i] += 9.8 * self.dt * ti.Vector([0, -1])
            self.velocity[i] *= ti.exp(-self.dt * self.damping)
            self.node[i] += self.dt * self.velocity[i]

    def step_explicit_with_grad(self):
        self.update_explicit_with_grad()
        self.handle_boundary()

    @ti.kernel
    def buildA(self):
        self.A.fill(0.0)
        # build A
        # A = M - h^2 * K
        # K = -ddE/dxdx
        for c in range(self.en_object_index[self.count[None]]):
            verts = self.element[c]
            for u in range(self.dim + 1):
                for d in ti.static(range(self.dim)):
                    K = -self.hassian(c, u, d)
                    for i in ti.static(range(self.dim)):
                        for j in ti.static(range(self.dim)):
                            self.A[self.dim * verts[u] + d,
                                   self.dim * verts[i] +
                                   j] += -K[j, i] * self.dt**2
                            self.A[self.dim * verts[u] + d,
                                   self.dim * verts[self.dim] +
                                   j] += K[j, i] * self.dt**2
        for i in range(self.vn_object_index[self.count[None]]):
            for j in ti.static(range(self.dim)):
                self.A[self.dim * i + j, self.dim * i + j] += self.node_mass[i]

    @ti.kernel
    def buildB(self):
        self.b.fill(0.0)
        # build b
        # b = h * M * v - h^2 * dE/dx + h^2 * f
        for i in range(self.en_object_index[self.count[None]]):
            H = -self.grad(i)
            for j in ti.static(range(self.dim)):
                for k in ti.static(range(self.dim)):
                    force = H[k, j]
                    self.b[self.element[i][self.dim] * self.dim +
                           k] -= force * self.dt**2
                    self.b[self.element[i][j] * self.dim +
                           k] += force * self.dt**2
        for i in range(self.vn_object_index[self.count[None]]):
            for j in ti.static(range(self.dim)):
                self.b[self.dim * i + j] += (self.node_mass[i] *
                                             self.velocity[i][j] * self.dt)
                self.b[self.dim * i + j] += (self.node_mass[i] * -9.8 * j *
                                             self.dt**2)

    def buildAandB(self):
        # A * delta_x = b
        # x(t + h) = x(t) + delta_x
        # v(t + h) = delta_x / h
        # A = M - h^2 * K
        # b = h * M * v - h^2 * dE/dx + h^2 * f
        self.buildA()
        self.buildB()

    @ti.kernel
    def apply_delta_x_with_vel(self, delta_x: ti.types.ndarray(ti.f32,
                                                               ndim=1)):
        for i in range(self.vn_object_index[self.count[None]]):
            for j in ti.static(range(self.dim)):
                self.velocity[i][j] = delta_x[self.dim * i + j] / self.dt
                self.node[i][j] += delta_x[self.dim * i + j]

    def step_implicit(self):
        self.buildAandB()
        A = self.A.to_numpy()[0:self.dim *
                              self.vn_object_index[self.count[None]],
                              0:self.dim *
                              self.vn_object_index[self.count[None]]]
        b = self.b.to_numpy()[0:self.dim *
                              self.vn_object_index[self.count[None]]]
        delta_x = np.linalg.solve(A, b)
        self.apply_delta_x_with_vel(delta_x)
        self.handle_boundary()

    @ti.kernel
    def calc_x_bar(self):
        for i in range(self.vn_object_index[self.count[None]]):
            self.bar_node[i] = self.node[i] + self.velocity[
                i] * self.dt + ti.Vector([0, -9.8]) * self.dt * self.dt

    @ti.kernel
    def x_to_prev_x(self):
        for i in range(self.vn_object_index[self.count[None]]):
            self.prev_node[i] = self.node[i]

    @ti.kernel
    def x_to_prev_t_x(self):
        for i in range(self.vn_object_index[self.count[None]]):
            self.velocity[i] = (self.node[i] - self.prev_t_node[i]) / self.dt
            self.prev_t_node[i] = self.node[i]

    @ti.kernel
    def buildB_optimize(self):
        self.b.fill(0.0)
        # build b
        # b = M * h * v + h^2 * f - h^2 * dE/dx - M * (x - x_previous_t)
        for i in range(self.en_object_index[self.count[None]]):
            H = -self.grad(i)
            for j in ti.static(range(self.dim)):
                for k in ti.static(range(self.dim)):
                    force = H[k, j]
                    self.b[self.element[i][self.dim] * self.dim +
                           k] -= force * self.dt**2
                    self.b[self.element[i][j] * self.dim +
                           k] += force * self.dt**2
        for i in range(self.vn_object_index[self.count[None]]):
            for j in ti.static(range(self.dim)):
                self.b[self.dim * i + j] += (
                    self.node_mass[i] * self.velocity[i][j] * self.dt +
                    self.node_mass[i] * -9.8 * j * self.dt**2 -
                    self.node_mass[i] *
                    (self.node[i][j] - self.prev_t_node[i][j]))

    def buildAandB_optimize(self):
        # G = 1/2 * (x - x_bar) * M * (x - x_bar) + h^2 * E(x)
        # x_bar = x_previous_t + h * v + h^2 * f * M^-1
        ##############################
        # difference
        ##############################
        # g = dG/dx = M * (x - x_bar) + h^2 * dE/dx
        # g = M * (x - x_previous_t) - M * h * v - h^2 * f + h^2 * dE/dx
        # b = -g = M * h * v + h^2 * f - h^2 * dE/dx - M * (x - x_previous_t)
        # A = ddG/dxdx = M + h^2 * ddE/dxdx
        self.buildA()
        self.buildB_optimize()

    @ti.kernel
    def apply_delta_x(self, delta_x: ti.types.ndarray(ti.f32, ndim=1),
                      step: ti.f32):
        for i in range(self.vn_object_index[self.count[None]]):
            for j in ti.static(range(self.dim)):
                self.node[i][j] = self.prev_node[i][j] + step * delta_x[
                    self.dim * i + j]

    @ti.kernel
    def update_velocity(self):
        for i in range(self.vn_object_index[self.count[None]]):
            self.velocity[i] = (self.node[i] - self.prev_t_node[i]) / self.dt

    def step_implicit_with_optimization(self):
        # G = inertia energy + h^2 * strain energy
        # G = 1/2 * (x - x_bar)^T * M * (x - x_bar) + h^2 * E
        # x_bar = (x_prev + h * v_prev + h^2 * f / M)
        # E = mu / 2 * (F^T * F - dim) - mu * log(J) + la / 2 * log(J)^2
        # dE/dF = mu * (F - F^T) + la * log(J) * F^-T

        # g = dG/dx = M * (x - x_bar) + h^2 * dE/dx
        # b = -g = M * (h * v_prev + h^2 * f / M + (x_prev - x)) - h^2 * dE/dx
        # b = h * M * v_prev - h^2 * dE/dx + h^2 * f - M * (x - x_prev)
        # [b in implicit]: b = h * M * v - h^2 * dE/dx + h^2 * f
        # A = d^2G/dx^2 = M + h^2 * d^2E/dx^2
        # [A in implicit]: A = M + h^2 * d^2E/dx^2

        # loop
        self.x_to_prev_t_x()  # save to x_t_prev
        self.calc_x_bar()
        self.reset_energy()
        self.compute_energy_with_inertia()
        self.save_energy()  # save to prev_energy
        # line search
        # alpha = 0.03
        # beta = 0.5
        # step = 1 / beta
        # delta_x = K^-1 * g
        # while step > epsilon:
        #     step *= beta
        #     x_test = x - step * delta_x
        #     if e(x_test) < e(x) - alpha * step * g^T * delta_x: # e(x) - alpha * step * g^T * K^-1 * g
        #         break
        # x = x_test
        max_iter = 20
        for it in range(max_iter):
            alpha = 0.03
            beta = 0.5
            step = 1.0 / beta
            self.x_to_prev_x()  # save to x_prev as small loop
            self.buildAandB_optimize()
            AA = self.A.to_numpy()[0:self.dim *
                                   self.vn_object_index[self.count[None]],
                                   0:self.dim *
                                   self.vn_object_index[self.count[None]]]
            bb = self.b.to_numpy()[0:self.dim *
                                   self.vn_object_index[self.count[None]]]
            delta_x = np.linalg.solve(AA, bb)
            self.apply_delta_x(delta_x, 1.0)
            if np.max(np.abs(bb)) < 1e-1 * self.dt:
                break
            # line search
            while step > 1e-4:
                step *= beta
                self.apply_delta_x(delta_x, step)
                self.reset_energy()
                self.compute_energy_with_inertia()
                if (self.energy[None] < self.prev_energy[None] - alpha * step *
                    (bb.dot(delta_x))):
                    break
            if step < stepDisplayThreshold:
                print("step: " + f"{step}")
            self.save_energy()
        if it > iterDisplayThreshold:
            print("iter: " + f"{it}")

        self.update_velocity()
        self.handle_boundary()

    @ti.kernel
    def flatten(self, dest: ti.template(), src: ti.template()):
        for i in src:
            for j in ti.static(range(self.dim)):
                dest[self.dim * i + j] = src[i][j]

    @ti.kernel
    def aggragate(self, dest: ti.template(), src: ti.template()):
        for i in dest:
            for j in ti.static(range(self.dim)):
                dest[i][j] = src[self.dim * i + j]

    @ti.kernel
    def copy(self, dest: ti.template(), src: ti.template()):
        for i in dest:
            dest[i] = src[i]

    @ti.kernel
    def add(self, ans: ti.template(), a: ti.template(), k: ti.f32,
            b: ti.template()):
        for i in ans:
            ans[i] = a[i] + k * b[i]

    @ti.kernel
    def dot(self, a: ti.template(), b: ti.template()) -> ti.f32:
        ans = 0.0
        for i in a:
            ans += a[i] * b[i]
        return ans

    @ti.kernel
    def mulA_cell(self, src: ti.template()):
        self.F_mul_ans.fill(0.0)
        for i in range(self.vn_object_index[self.count[None]]):
            for j in ti.static(range(self.dim)):
                self.F_mul_ans[self.dim * i +
                               j] += self.node_mass[i] * src[self.dim * i + j]
        for c in range(self.en_object_index[self.count[None]]):
            verts = self.element[c]
            for u in range(self.dim + 1):
                for d in ti.static(range(self.dim)):
                    K = -self.hassian(c, u, d)
                    for i in ti.static(range(self.dim)):
                        for j in ti.static(range(self.dim)):
                            tmp = src[self.dim * verts[i] + j] -\
                                          src[self.dim * verts[self.dim] + j]
                            self.F_mul_ans[self.dim * verts[u] +
                                           d] += -(self.dt**2) * K[j, i] * tmp
        # for i in range(self.vn_object_index[self.count[None]] * self.dim):
        #     for j in range(self.vn_object_index[self.count[None]] * self.dim):
        #         self.F_mul_ans[i] += self.A[i, j] * src[j]

    def mulA(self, src: ti.template()):
        self.mulA_cell(src)
        return self.F_mul_ans

    def cg(self):
        # conjugate gradient
        # Ax = b
        # here: x = delta_x
        i = 0
        self.delta_x.fill(0.0)
        self.flatten(self.flat_v, self.velocity)
        # initial guess x = v * dt
        # r = b - A * v
        # d = r
        # delta_new = r.dot(r)
        # delta_0 = delta_new
        self.add(self.r, self.b, -1, self.mulA(self.delta_x))
        self.copy(self.d, self.r)
        delta_new = self.dot(self.r, self.r)
        delta_0 = delta_new
        while i < 100 and delta_new > 1e-10 * delta_0:
            q = self.mulA(self.d)
            alpha = delta_new / self.dot(self.d, q)
            self.add(self.delta_x, self.delta_x, alpha, self.d)
            if i % 50 == 0:
                self.add(self.r, self.b, -1, self.mulA(self.delta_x))
            else:
                self.add(self.r, self.r, -alpha, q)
            delta_old = delta_new
            delta_new = self.dot(self.r, self.r)
            beta = delta_new / delta_old
            self.add(self.d, self.r, beta, self.d)
            i += 1
        # print(i)

    def step_implicit_cg_with_optimization(self):
        # loop
        self.x_to_prev_t_x()  # save to x_t_prev
        self.calc_x_bar()
        self.reset_energy()
        self.compute_energy_with_inertia()
        self.save_energy()  # save to prev_energy
        # line search
        # alpha = 0.03
        # beta = 0.5
        # step = 1 / beta
        # delta_x = K^-1 * g
        # while step > epsilon:
        #     step *= beta
        #     x_test = x - step * delta_x
        #     if e(x_test) < e(x) - alpha * step * g^T * delta_x: # e(x) - alpha * step * g^T * K^-1 * g
        #         break
        # x = x_test
        max_iter = 20
        for it in range(max_iter):
            self.x_to_prev_x()  # save to x_prev as small loop
            # self.buildAandB_optimize()
            self.buildB_optimize()
            self.cg()
            bb = self.b.to_numpy()[0:self.dim *
                                   self.vn_object_index[self.count[None]]]
            delta_x = self.delta_x.to_numpy(
            )[0:self.dim * self.vn_object_index[self.count[None]]]
            self.apply_delta_x(delta_x, 1.0)
            if np.max(np.abs(bb)) < 1e-1 * self.dt:
                break
            # line search
            alpha = 0.03
            beta = 0.5
            step = 1.0 / beta
            while step > 1e-4:
                step *= beta
                self.apply_delta_x(delta_x, step)
                self.reset_energy()
                self.compute_energy_with_inertia()
                if (self.energy[None] < self.prev_energy[None] - alpha * step *
                    (bb.dot(delta_x))):
                    break
            if step < stepDisplayThreshold:
                print("step: " + f"{step}")
            self.save_energy()
        if it > iterDisplayThreshold:
            print("iter: " + f"{it}")
        self.update_velocity()
        self.handle_boundary()

    @ti.func
    def U3(self, i):
        # bounding contact potential
        t = 0.0
        dist_x_l = self.node[i].x - boundary[0][0]
        dist_x_u = boundary[0][1] - self.node[i].x
        dist_y_l = self.node[i].y - boundary[1][0]
        dist_y_u = boundary[1][1] - self.node[i].y

        if dist_x_l <= self.bar_d:
            t += -self.k * (
                (dist_x_l - self.bar_d)**2) * ti.log(dist_x_l / self.bar_d)
        if dist_x_u <= self.bar_d:
            t += -self.k * (
                (dist_x_u - self.bar_d)**2) * ti.log(dist_x_u / self.bar_d)
        if dist_y_l <= self.bar_d:
            t += -self.k * (
                (dist_y_l - self.bar_d)**2) * ti.log(dist_y_l / self.bar_d)
        if dist_y_u <= self.bar_d:
            t += -self.k * (
                (dist_y_u - self.bar_d)**2) * ti.log(dist_y_u / self.bar_d)

        return t

    @ti.func
    def barrier_energy(self, p, q):
        rt = 0.0
        d1, d2, d3 = 1.0, 1.0, 1.0
        # points: p
        pt = self.node[p]
        # triangle
        t1 = self.node[self.element[q][0]]
        t2 = self.node[self.element[q][1]]
        t3 = self.node[self.element[q][2]]

        # point - triangle pair, find point - triangle shortest dist
        # 2d problem: point - line segment shortest dist

        ab1 = t2 - t1
        ac1 = pt - t1
        bc1 = pt - t2
        a1 = ab1.dot(ac1)
        abd1 = ab1.norm_sqr()
        if ab1.cross(ac1) < 0:
            if a1 < 0:
                d1 = ac1.norm()
            elif a1 > abd1:
                d1 = bc1.norm()
            else:
                d1 = ti.abs((t2.y - t1.y) * pt.x - (t2.x - t1.x) * pt.y +
                            t2.x * t1.y - t2.y * t1.x) / (t2 - t1).norm()

        ab2 = t3 - t2
        ac2 = pt - t2
        bc2 = pt - t3
        a2 = ab2.dot(ac2)
        abd2 = ab2.norm_sqr()
        if ab2.cross(ac2) < 0:
            if a2 < 0:
                d2 = ac2.norm()
            elif a2 > abd2:
                d2 = bc2.norm()
            else:
                d2 = ti.abs((t3.y - t2.y) * pt.x - (t3.x - t2.x) * pt.y +
                            t3.x * t2.y - t3.y * t2.x) / (t3 - t2).norm()

        ab3 = t1 - t3
        ac3 = pt - t3
        bc3 = pt - t1
        a3 = ab3.dot(ac3)
        abd3 = ab3.norm_sqr()
        if ab3.cross(ac3) < 0:
            if a3 < 0:
                d3 = ac3.norm()
            elif a3 > abd3:
                d3 = bc3.norm()
            else:
                d3 = ti.abs((t1.y - t3.y) * pt.x - (t1.x - t3.x) * pt.y +
                            t1.x * t3.y - t1.y * t3.x) / (t1 - t3).norm()

        dist = ti.min(d1, d2, d3)
        # b_C2 function
        if dist < self.bar_d:
            rt = -self.k * ((dist - self.bar_d)**2) * ti.log(dist / self.bar_d)
        else:
            rt = 0
        return rt

    @ti.kernel
    def compute_energy_with_barrier(self):

        for i in range(self.en_object_index[self.count[None]]):
            self.energy[None] += self.U0(i) * self.dt * self.dt

        for i in range(self.vn_object_index[self.count[None]]):
            self.energy[None] += self.U1(i) * self.dt * self.dt
            self.energy[None] += self.U2(i)
            self.energy[None] += self.U3(i)
        # i, j: object
        # p: node, q:face
        for i in range(1, self.count[None] + 1):
            for j in range(i + 1, self.count[None] + 1):
                for p in range(self.vn_object_index[i - 1],
                               self.vn_object_index[i]):
                    for q in range(self.en_object_index[j - 1],
                                   self.en_object_index[j]):
                        self.energy[None] += self.barrier_energy(p, q)
        for i in range(1, self.count[None] + 1):
            for j in range(i + 1, self.count[None] + 1):
                for p in range(self.vn_object_index[j - 1],
                               self.vn_object_index[j]):
                    for q in range(self.en_object_index[i - 1],
                                   self.en_object_index[i]):
                        self.energy[None] += self.barrier_energy(p, q)

    @ti.kernel
    def compute_energy_only_barrier(self):
        for i in range(self.vn_object_index[self.count[None]]):
            self.energy[None] += self.U3(i)
        # i, j: object
        # p: node, q:face
        for i in range(1, self.count[None] + 1):
            for j in range(i + 1, self.count[None] + 1):
                for p in range(self.vn_object_index[i - 1],
                               self.vn_object_index[i]):
                    for q in range(self.en_object_index[j - 1],
                                   self.en_object_index[j]):
                        self.energy[None] += self.barrier_energy(p, q)
        for i in range(1, self.count[None] + 1):
            for j in range(i + 1, self.count[None] + 1):
                for p in range(self.vn_object_index[j - 1],
                               self.vn_object_index[j]):
                    for q in range(self.en_object_index[i - 1],
                                   self.en_object_index[i]):
                        self.energy[None] += self.barrier_energy(p, q)

    @ti.kernel
    def updateB_barrier(self):
        # b = h * M * v_prev - h^2 * dE/dx + h^2 * f - M * (x - x_prev)
        for i in range(self.vn_object_index[self.count[None]]):
            for j in ti.static(range(self.dim)):
                self.b[i * self.dim + j] += -self.node.grad[i][j]

        # dE_Barrier/dx = -k * Sigma[(2*t1*log(t2) + t1^2/t2/bar_d) * ddist/dx]
        # for i in range(self.vn_object_index[self.count[None]]):
        #     dist_x_l = self.node[i].x - boundary[0][0]
        #     dist_x_u = boundary[0][1] - self.node[i].x
        #     dist_y_l = self.node[i].y - boundary[1][0]
        #     dist_y_u = boundary[1][1] - self.node[i].y

        #     if dist_x_l <= self.bar_d:
        #         self.b[i * self.dim] += self.k * (
        #             2 *
        #             (dist_x_l - self.bar_d) * ti.log(dist_x_l / self.bar_d) +
        #             (dist_x_l - self.bar_d)**2 / dist_x_l)
        #     if dist_x_u <= self.bar_d:
        #         self.b[i * self.dim] += self.k * (
        #             2 *
        #             (dist_x_u - self.bar_d) * ti.log(dist_x_u / self.bar_d) +
        #             (dist_x_u - self.bar_d)**2 / dist_x_u) * (-1)
        #     if dist_y_l <= self.bar_d:
        #         self.b[i * self.dim + 1] += self.k * (
        #             2 *
        #             (dist_y_l - self.bar_d) * ti.log(dist_y_l / self.bar_d) +
        #             (dist_y_l - self.bar_d)**2 / dist_y_l)
        #     if dist_y_u <= self.bar_d:
        #         self.b[i * self.dim + 1] += self.k * (
        #             2 *
        #             (dist_y_u - self.bar_d) * ti.log(dist_y_u / self.bar_d) +
        #             (dist_y_u - self.bar_d)**2 / dist_y_u) * (-1)

        # for i in range(1, self.count[None] + 1):
        #     for j in range(i + 1, self.count[None] + 1):
        #         for p in range(self.vn_object_index[i - 1],
        #                        self.vn_object_index[i]):
        #             for q in range(self.en_object_index[j - 1],
        #                            self.en_object_index[j]):
        #                 self.energy[None] += self.barrier_energy(p, q)
        # for i in range(1, self.count[None] + 1):
        #     for j in range(i + 1, self.count[None] + 1):
        #         for p in range(self.vn_object_index[j - 1],
        #                        self.vn_object_index[j]):
        #             for q in range(self.en_object_index[i - 1],
        #                            self.en_object_index[i]):
        #                 self.energy[None] += self.barrier_energy(p, q)

    @ti.func
    def implicit_prob_node_in_element(self, i, j):
        # node i, element j
        rt = 0
        a, b, c = self.node[self.element[j][0]], self.node[
            self.element[j][1]], self.node[self.element[j][2]]
        p = self.node[i]
        Sabc = abs((b - a).cross(c - a))
        Spbc = abs((b - p).cross(c - p))
        Sapc = abs((p - a).cross(c - a))
        Sabp = abs((b - a).cross(p - a))
        if ti.abs(Sabc - Spbc - Sapc - Sabp) > self.epsilon:
            rt = 0
        else:
            rt = 1
        return rt

    @ti.kernel
    def implicit_prob_all(self) -> ti.i32:
        # i,j object
        # p,q node, face
        # flag 0: not contact, >0: contact
        flag = 0
        for i in range(1, self.count[None] + 1):
            for j in range(1, self.count[None] + 1):
                if i != j:
                    for p in range(self.vn_object_index[i - 1],
                                   self.vn_object_index[i]):
                        for q in range(self.en_object_index[j - 1],
                                       self.en_object_index[j]):
                            flag += self.implicit_prob_node_in_element(p, q)

        # boundary
        for t in range(self.vn_object_index[self.count[None]]):
            if self.node[t][0] <= boundary[0][0] + self.epsilon:
                flag += 1
            if self.node[t][0] > boundary[0][1] - self.epsilon:
                flag += 1
            if self.node[t][1] <= boundary[1][0] + self.epsilon:
                flag += 1
            if self.node[t][1] > boundary[1][1] - self.epsilon:
                flag += 1

        return flag

    def ccd(self, delta_x, l=0, r=1, max_iter=10):
        iter = 0
        while True:
            iter += 1
            mid = (l + r) / 2

            if iter > max_iter or r - l < self.epsilon:
                break

            self.apply_delta_x(delta_x, mid)
            flag = self.implicit_prob_all()
            if flag == 0:
                l = mid
            else:
                r = mid
        return l

    def step_ipc(self):
        # G_star = G + E_Barrier
        # dG_star/dx = dG/dx + dE_Barrier/dx
        # E_Barrier = -k * Sigma((dist - bar_d)**2 * log(dist / bar_d)) = Sigma(t1^2 * log(t2))
        # dE_Barrier/dx = -k * Sigma[(2*t1*log(t2) + t1^2/t2/bar_d) * ddist/dx]
        # ddE_Barrier/dxdx = -k * Sigma[(2*t1*log(t2) + t1^2*1/t2/bar_d) * dddist/dxdx
        #                       + (2*log(t2） + 2*t1/t2/bar_d) * ddist/dx * ddist/dx^T
        #                       + (2*t1/t2/bar_d - t1^2/t2^2/bar_d^2) * ddist/dx * ddist/dx^T]

        # loop
        self.x_to_prev_t_x()  # save to x_t_prev
        self.calc_x_bar()
        self.reset_energy()
        # self.compute_energy_with_inertia()
        self.compute_energy_with_barrier()
        self.save_energy()  # save to prev_energy
        # ccd
        # line search
        max_iter = 20
        for it in range(max_iter):
            self.x_to_prev_x()  # save to x_prev as small loop
            self.reset_energy()
            with ti.ad.Tape(self.energy):
                self.compute_energy_only_barrier()
            self.buildB_optimize()
            self.updateB_barrier()
            ##### update barrier force
            self.cg()
            bb = self.b.to_numpy()[0:self.dim *
                                   self.vn_object_index[self.count[None]]]
            delta_x = self.delta_x.to_numpy(
            )[0:self.dim * self.vn_object_index[self.count[None]]]
            self.apply_delta_x(delta_x, 1.0)
            if np.max(np.abs(bb)) < 1e-1 * self.dt:
                break
            # ccd
            step = self.ccd(delta_x, r=1.0)
            if step < stepDisplayThreshold:
                print("before" + f"{step}")
            alpha = 0.03
            beta = 0.5
            step = step / beta
            while step > 1e-4:
                step *= beta
                self.apply_delta_x(delta_x, step)
                self.reset_energy()
                self.compute_energy_with_barrier()
                if (self.energy[None] < self.prev_energy[None] - alpha * step *
                    (bb.dot(delta_x))):
                    break
            if step < stepDisplayThreshold:
                print("after" + f"{step}")
            self.save_energy()
        if it > iterDisplayThreshold:
            print("iter: " + f"{it}")
        self.update_velocity()


def print_profile(name, display_name):
    result = ti.profiler.query_kernel_profiler_info(name)
    print(display_name, result.counter)


def run():
    # obj = Object([1, 1], "obj")
    objs = []
    for i in range(7):
        objs.append(Object([0, 0], "obj" + str(i)))
    it = 0
    s = FEM()
    obj = objs[0]
    s.add_obj(obj.vn, obj.en, obj.node.astype(np.float32), obj.element)
    gui = ti.GUI('IPC', screen_res)
    canvas = gui.canvas

    ti.profiler.clear_kernel_profiler_info()
    st = time.time()
    while gui.running:
        if it % (int(1 / s.dt / 60)) == 0:
            canvas.clear(bg_color)
            gui.circle(s.ball_pos * screen_to_world_ratio / screen_res,
                       radius=s.ball_r * 40,
                       color=0x666666)
            s.create_lines()
            begins = s.begin_point.to_numpy()[:3 *
                                              s.en_object_index[s.count[None]]]
            ends = s.end_point.to_numpy()[:3 *
                                          s.en_object_index[s.count[None]]]
            gui.lines(begins, ends, color=line_color, radius=line_radius)
            # if it < 1200:
            #     gui.show("img/" + str(it) + "ipc.png")
            gui.show()
        it += 1
        if gui.get_event(ti.GUI.PRESS):
            e = gui.event
            if e.key == ti.GUI.ESCAPE:
                break
            elif e.key == '0':
                s.add_obj(objs[0].vn, objs[0].en,
                          objs[0].node.astype(np.float32), objs[0].element)
            elif e.key == '1':
                s.add_obj(objs[1].vn, objs[1].en,
                          objs[1].node.astype(np.float32), objs[1].element)
            elif e.key == '2':
                s.add_obj(objs[2].vn, objs[2].en,
                          objs[2].node.astype(np.float32), objs[2].element)
            elif e.key == '3':
                s.add_obj(objs[3].vn, objs[3].en,
                          objs[3].node.astype(np.float32), objs[3].element)
            elif e.key == '4':
                s.add_obj(objs[4].vn, objs[4].en,
                          objs[4].node.astype(np.float32), objs[4].element)
            elif e.key == '5':
                s.add_obj(objs[5].vn, objs[5].en,
                          objs[5].node.astype(np.float32), objs[5].element)
            elif e.key == '6':
                s.add_obj(objs[6].vn, objs[6].en,
                          objs[6].node.astype(np.float32), objs[6].element)

        # s.step_explicit_auto_grad()
        # s.step_explicit_with_grad()
        # s.step_implicit()
        # s.step_implicit_with_optimization()
        # s.step_implicit_cg_with_optimization()
        s.step_ipc()

        # TODO: grad and hessian of the barrier energy
        # TODO: accurate ccd calculation
        # TODO: preconditioned conjugate gradient
        # TODO: add 3D support and cloth simulation

    print("time: ", time.time() - st)
    ti.profiler.print_kernel_profiler_info()

    # print_profile(s.mulA_cell.__name__, "mulA_cell")
    # print_profile(s.dot.__name__, "dot")
    # print_profile(s.reset_energy.__name__, "reset_energy")
    # print_profile(s.save_energy.__name__, "save_energy")
    # print_profile(s.buildB_optimize.__name__, "buildB_optimize")


def run_ggui():
    # obj = Object([1, 1], "obj")
    objs = []
    for i in range(7):
        objs.append(Object([0, 0], "obj" + str(i)))
    it = 0
    s = FEM()
    obj = objs[0]
    s.add_obj(obj.vn, obj.en, obj.node.astype(np.float32), obj.element)

    window = ti.ui.Window("Implicit FEM Test", (1000, 1000), vsync=True)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(5.0, 5.0, 5.0)
    camera.lookat(5.0, 5.0, 0)
    camera.fov(90)

    ti.profiler.clear_kernel_profiler_info()
    st = time.time()
    while window.running:
        if it % (int(1 / s.dt / 60)) == 0:
            s.create_lines()
            # begins = s.begin_point.to_numpy()[:3 *
            #                                   s.en_object_index[s.count[None]]]
            # ends = s.end_point.to_numpy()[:3 *
            #                               s.en_object_index[s.count[None]]]
            camera.track_user_inputs(window,
                                     movement_speed=0.03,
                                     hold_key=ti.ui.RMB)
            scene.set_camera(camera)
            scene.ambient_light((1, ) * 3)

            scene.particles(s.node,
                            0.1,
                            color=(0, 1, 1),
                            index_count=s.vn_object_index[s.count[None]])
            scene.lines(s.node,
                        2,
                        s.en_flat,
                        color=(0, 1, 1),
                        vertex_count=s.en_object_index[s.count[None]] * 6)
            print(s.en_object_index[s.count[None]])
            canvas.scene(scene)
            window.show()
        it += 1

        # if gui.get_event(ti.GUI.PRESS):
        #     e = gui.event
        #     if e.key == ti.GUI.ESCAPE:
        #         break
        #     elif e.key == '0':
        #         s.add_obj(objs[0].vn, objs[0].en,
        #                   objs[0].node.astype(np.float32), objs[0].element)
        #     elif e.key == '1':
        #         s.add_obj(objs[1].vn, objs[1].en,
        #                   objs[1].node.astype(np.float32), objs[1].element)
        #     elif e.key == '2':
        #         s.add_obj(objs[2].vn, objs[2].en,
        #                   objs[2].node.astype(np.float32), objs[2].element)
        #     elif e.key == '3':
        #         s.add_obj(objs[3].vn, objs[3].en,
        #                   objs[3].node.astype(np.float32), objs[3].element)
        #     elif e.key == '4':
        #         s.add_obj(objs[4].vn, objs[4].en,
        #                   objs[4].node.astype(np.float32), objs[4].element)
        #     elif e.key == '5':
        #         s.add_obj(objs[5].vn, objs[5].en,
        #                   objs[5].node.astype(np.float32), objs[5].element)
        #     elif e.key == '6':
        #         s.add_obj(objs[6].vn, objs[6].en,
        #                   objs[6].node.astype(np.float32), objs[6].element)

        # s.step_explicit_auto_grad()
        # s.step_explicit_with_grad()
        # s.step_implicit()
        # s.step_implicit_with_optimization()
        # s.step_implicit_cg_with_optimization()
        s.step_ipc()

    print("time: ", time.time() - st)
    # ti.profiler.print_kernel_profiler_info()

    # print_profile(s.mulA_cell.__name__, "mulA_cell")
    # print_profile(s.dot.__name__, "dot")
    # print_profile(s.reset_energy.__name__, "reset_energy")
    # print_profile(s.save_energy.__name__, "save_energy")
    # print_profile(s.buildB_optimize.__name__, "buildB_optimize")


if __name__ == '__main__':
    run()