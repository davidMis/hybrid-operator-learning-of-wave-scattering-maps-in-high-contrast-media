# Overview:
# Reproduce the inherited frequency-domain finite-element Helmholtz solver used
# by the legacy data-generation scripts. The implementation intentionally keeps
# the same low-order triangular FEM discretization, Gaussian source, top
# Dirichlet boundary, and Robin absorbing conditions on the other edges so it can
# be compared directly against the published pressure arrays.
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import numpy as np

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
except ImportError as error:  # pragma: no cover - exercised only without optional deps
    raise ImportError(
        "The FEM Helmholtz solver requires scipy. Install it with "
        "`python -m pip install -e '.[fem]'`."
    ) from error


VelocitySampling = Literal["nearest", "bilinear"]
LinearSolver = Literal["bicgstab", "direct"]

DEFAULT_FEM_DOMAIN_SIZE_M = 10_000.0
DEFAULT_FEM_FREQUENCY_HZ = 4.0
DEFAULT_FEM_SOURCE_ROWS_BELOW_TOP = 3.0
DEFAULT_FEM_SOURCE_SPREAD_GRID_CELLS = 2.0
DEFAULT_FEM_SOURCE_AMPLITUDE = 1.0
DEFAULT_FEM_ABC_VELOCITY_M_S = 1_500.0
DEFAULT_FEM_ABC_SCALE = 1.0
DEFAULT_FEM_SPILU_DROP_TOL = 1e-3
DEFAULT_FEM_SPILU_FILL_FACTOR = 20.0
DEFAULT_FEM_BICGSTAB_RTOL = 1e-7
DEFAULT_FEM_BICGSTAB_MAXITER = 1000


@dataclass(frozen=True)
class FEMHelmholtzSettings:
    """Numerical and physical settings for one FEM Helmholtz solve."""

    frequency_hz: float = DEFAULT_FEM_FREQUENCY_HZ
    domain_size_x_m: float = DEFAULT_FEM_DOMAIN_SIZE_M
    domain_size_y_m: float = DEFAULT_FEM_DOMAIN_SIZE_M
    source_x_m: float | None = None
    source_rows_below_top: float = DEFAULT_FEM_SOURCE_ROWS_BELOW_TOP
    source_spread_grid_cells: float = DEFAULT_FEM_SOURCE_SPREAD_GRID_CELLS
    source_amplitude: float = DEFAULT_FEM_SOURCE_AMPLITUDE
    abc_velocity_m_s: float = DEFAULT_FEM_ABC_VELOCITY_M_S
    abc_scale: float = DEFAULT_FEM_ABC_SCALE
    velocity_sampling: VelocitySampling = "nearest"
    linear_solver: LinearSolver = "bicgstab"
    spilu_drop_tol: float = DEFAULT_FEM_SPILU_DROP_TOL
    spilu_fill_factor: float = DEFAULT_FEM_SPILU_FILL_FACTOR
    bicgstab_rtol: float = DEFAULT_FEM_BICGSTAB_RTOL
    bicgstab_maxiter: int = DEFAULT_FEM_BICGSTAB_MAXITER


@dataclass(frozen=True)
class FEMHelmholtzDiagnostics:
    """Metadata and timings for one completed FEM Helmholtz solve."""

    velocity_shape: tuple[int, int]
    spacing_m: tuple[float, float]
    frequency_hz: float
    omega: float
    source_coordinates_m: tuple[float, float]
    source_sigma_m: float
    matrix_shape: tuple[int, int]
    matrix_nnz: int
    assemble_seconds: float
    solve_seconds: float
    sample_generation_seconds: float
    solve_method: str
    bicgstab_info: int | None
    max_abs_pressure: float


@dataclass(frozen=True)
class FEMHelmholtzSample:
    """Complex FEM pressure field and diagnostics for a single sample."""

    pressure: np.ndarray
    diagnostics: FEMHelmholtzDiagnostics

    @property
    def pressure_channels(self) -> np.ndarray:
        """Return pressure as float32 [2, ny, nx] channels matching paper data."""

        return np.stack([self.pressure.real, self.pressure.imag], axis=0).astype(np.float32, copy=False)


class FEMConfigurationError(RuntimeError):
    """Raised when the requested FEM configuration is invalid."""


class FEMHelmholtzSolver:
    """Finite-element Helmholtz solver matching the recovered legacy generator."""

    def __init__(self, shape: tuple[int, int], settings: FEMHelmholtzSettings | None = None):
        self.settings = settings or FEMHelmholtzSettings()
        self.ny, self.nx = validate_shape(shape)
        self._validate_settings()

        self.x = np.linspace(0.0, self.settings.domain_size_x_m, self.nx, dtype=np.float64)
        self.y = np.linspace(0.0, self.settings.domain_size_y_m, self.ny, dtype=np.float64)
        self.dx = float(self.x[1] - self.x[0])
        self.dy = float(self.y[1] - self.y[0])
        self.omega = float(2.0 * np.pi * self.settings.frequency_hz)
        self.triangles = build_triangles(self.nx, self.ny)

    def solve(self, velocity_m_s: np.ndarray) -> FEMHelmholtzSample:
        """Solve one complex pressure field for a velocity model in m/s."""

        velocity = validate_velocity_m_s(velocity_m_s, expected_shape=(self.ny, self.nx))
        total_start = time.perf_counter()
        assemble_start = time.perf_counter()
        matrix, rhs, source_coordinates_m, source_sigma_m = self._assemble_system(velocity)
        assemble_seconds = time.perf_counter() - assemble_start

        solve_start = time.perf_counter()
        pressure_vector, solve_method, bicgstab_info = self._solve_linear_system(matrix, rhs)
        solve_seconds = time.perf_counter() - solve_start
        pressure = pressure_vector.reshape(self.ny, self.nx).astype(np.complex64, copy=False)
        sample_generation_seconds = time.perf_counter() - total_start

        diagnostics = FEMHelmholtzDiagnostics(
            velocity_shape=(self.ny, self.nx),
            spacing_m=(self.dx, self.dy),
            frequency_hz=float(self.settings.frequency_hz),
            omega=self.omega,
            source_coordinates_m=source_coordinates_m,
            source_sigma_m=source_sigma_m,
            matrix_shape=matrix.shape,
            matrix_nnz=int(matrix.nnz),
            assemble_seconds=assemble_seconds,
            solve_seconds=solve_seconds,
            sample_generation_seconds=sample_generation_seconds,
            solve_method=solve_method,
            bicgstab_info=bicgstab_info,
            max_abs_pressure=float(np.max(np.abs(pressure))),
        )
        return FEMHelmholtzSample(pressure=pressure, diagnostics=diagnostics)

    def _assemble_system(
        self,
        velocity: np.ndarray,
    ) -> tuple["sp.csr_matrix", np.ndarray, tuple[float, float], float]:
        """Assemble the sparse FEM matrix and Gaussian source right-hand side."""

        n_nodes = self.nx * self.ny
        rows: list[int] = []
        cols: list[int] = []
        vals: list[complex] = []
        rhs = np.zeros(n_nodes, dtype=np.complex128)

        source_x_m = self.settings.source_x_m
        if source_x_m is None:
            source_x_m = 0.5 * self.settings.domain_size_x_m
        source_y_m = float(self.settings.source_rows_below_top) * self.dy
        source_sigma_m = float(self.settings.source_spread_grid_cells) * min(self.dx, self.dy)

        k_ref = self.settings.abc_scale * (self.omega / self.settings.abc_velocity_m_s)

        def add(row: int, col: int, value: complex) -> None:
            rows.append(row)
            cols.append(col)
            vals.append(value)

        for n0, n1, n2 in self.triangles:
            j0, i0 = divmod(int(n0), self.nx)
            j1, i1 = divmod(int(n1), self.nx)
            j2, i2 = divmod(int(n2), self.nx)
            xy = np.array(
                [[self.x[i0], self.y[j0]], [self.x[i1], self.y[j1]], [self.x[i2], self.y[j2]]],
                dtype=np.float64,
            )
            x_centroid = float(xy[:, 0].mean())
            y_centroid = float(xy[:, 1].mean())
            v_centroid = sample_velocity(velocity, x_centroid, y_centroid, self.settings, self.nx, self.ny)
            k2_elem = (self.omega / v_centroid) ** 2

            stiffness, mass, area = triangle_local_mats(xy, k2_elem)
            nodes = (int(n0), int(n1), int(n2))
            for local_row, node_row in enumerate(nodes):
                for local_col, node_col in enumerate(nodes):
                    add(node_row, node_col, stiffness[local_row, local_col] - mass[local_row, local_col])

            source_value = gaussian_source(
                x_centroid,
                y_centroid,
                source_x_m=source_x_m,
                source_y_m=source_y_m,
                sigma_m=source_sigma_m,
                amplitude=self.settings.source_amplitude,
            )
            element_load = source_value * (area / 3.0)
            rhs[nodes[0]] += element_load
            rhs[nodes[1]] += element_load
            rhs[nodes[2]] += element_load

        self._add_robin_edges(add, k_ref)
        matrix = sp.csr_matrix(
            (np.asarray(vals, dtype=np.complex128), (np.asarray(rows), np.asarray(cols))),
            shape=(n_nodes, n_nodes),
        )
        matrix = apply_top_dirichlet(matrix, rhs, self.nx)
        return matrix, rhs, (float(source_x_m), float(source_y_m)), float(source_sigma_m)

    def _add_robin_edges(self, add, k_ref: float) -> None:
        """Add Robin absorbing boundary contributions to left, right, and bottom."""

        def add_edge(i_a: int, j_a: int, i_b: int, j_b: int) -> None:
            n_a = node_id(i_a, j_a, self.nx)
            n_b = node_id(i_b, j_b, self.nx)
            edge_length = float(np.hypot(self.x[i_b] - self.x[i_a], self.y[j_b] - self.y[j_a]))
            local = edge_robin_local(edge_length, k_ref)
            add(n_a, n_a, local[0, 0])
            add(n_a, n_b, local[0, 1])
            add(n_b, n_a, local[1, 0])
            add(n_b, n_b, local[1, 1])

        for j in range(self.ny - 1):
            add_edge(0, j, 0, j + 1)
            add_edge(self.nx - 1, j, self.nx - 1, j + 1)
        for i in range(self.nx - 1):
            add_edge(i, self.ny - 1, i + 1, self.ny - 1)

    def _solve_linear_system(self, matrix, rhs: np.ndarray) -> tuple[np.ndarray, str, int | None]:
        """Solve the sparse complex system with the inherited fallback policy."""

        if self.settings.linear_solver == "direct":
            return spla.spsolve(matrix, rhs), "spsolve", None

        try:
            ilu = spla.spilu(
                matrix.tocsc(),
                drop_tol=self.settings.spilu_drop_tol,
                fill_factor=self.settings.spilu_fill_factor,
            )
            preconditioner = spla.LinearOperator(matrix.shape, matvec=ilu.solve)
            pressure, info = bicgstab_compat(
                matrix,
                rhs,
                preconditioner=preconditioner,
                rtol=self.settings.bicgstab_rtol,
                maxiter=self.settings.bicgstab_maxiter,
            )
            if info == 0:
                return pressure, "bicgstab_spilu", int(info)
            return spla.spsolve(matrix, rhs), "spsolve_after_bicgstab", int(info)
        except Exception:
            return spla.spsolve(matrix, rhs), "spsolve_after_spilu_error", None

    def _validate_settings(self) -> None:
        """Validate settings early so CLI errors are actionable."""

        numeric_positive = {
            "frequency_hz": self.settings.frequency_hz,
            "domain_size_x_m": self.settings.domain_size_x_m,
            "domain_size_y_m": self.settings.domain_size_y_m,
            "source_spread_grid_cells": self.settings.source_spread_grid_cells,
            "abc_velocity_m_s": self.settings.abc_velocity_m_s,
            "spilu_fill_factor": self.settings.spilu_fill_factor,
            "bicgstab_rtol": self.settings.bicgstab_rtol,
        }
        for name, value in numeric_positive.items():
            if not np.isfinite(value) or value <= 0.0:
                raise FEMConfigurationError(f"{name} must be positive; got {value}.")
        if self.settings.source_rows_below_top < 0.0:
            raise FEMConfigurationError(
                f"source_rows_below_top must be non-negative; got {self.settings.source_rows_below_top}."
            )
        if (
            self.settings.source_x_m is not None
            and not 0.0 <= self.settings.source_x_m <= self.settings.domain_size_x_m
        ):
            raise FEMConfigurationError(
                "source_x_m must lie inside the x-domain; "
                f"got {self.settings.source_x_m} for Lx={self.settings.domain_size_x_m}."
            )
        if self.settings.velocity_sampling not in ("nearest", "bilinear"):
            raise FEMConfigurationError(
                f"Unknown velocity_sampling '{self.settings.velocity_sampling}'. Expected nearest or bilinear."
            )
        if self.settings.linear_solver not in ("bicgstab", "direct"):
            raise FEMConfigurationError(
                f"Unknown linear_solver '{self.settings.linear_solver}'. Expected bicgstab or direct."
            )
        if self.settings.bicgstab_maxiter <= 0:
            raise FEMConfigurationError(f"bicgstab_maxiter must be positive; got {self.settings.bicgstab_maxiter}.")


def validate_shape(shape: tuple[int, int]) -> tuple[int, int]:
    """Return a validated two-dimensional solver shape."""

    if len(shape) != 2:
        raise FEMConfigurationError(f"Expected a 2D velocity shape, got {shape}.")
    ny, nx = int(shape[0]), int(shape[1])
    if ny < 2 or nx < 2:
        raise FEMConfigurationError(f"Velocity shape must be at least 2x2; got {shape}.")
    return ny, nx


def validate_velocity_m_s(velocity: np.ndarray, *, expected_shape: tuple[int, int]) -> np.ndarray:
    """Return a float64 velocity array in m/s with shape and value validation."""

    array = np.asarray(velocity, dtype=np.float64)
    if array.shape != expected_shape:
        raise ValueError(f"Expected velocity shape {expected_shape}, got {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError("Velocity field contains NaN or Inf values.")
    if np.any(array <= 0.0):
        raise ValueError("Velocity field must be strictly positive everywhere.")
    if float(np.nanmedian(array)) < 100.0:
        raise ValueError(
            "FEM solver expects velocity in m/s, but the median velocity is below 100. "
            "Convert km/s inputs to m/s before solving."
        )
    return array


def build_triangles(nx: int, ny: int) -> np.ndarray:
    """Return the two-right-triangle mesh connectivity used by the legacy code."""

    triangles: list[tuple[int, int, int]] = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n00 = node_id(i, j, nx)
            n10 = node_id(i + 1, j, nx)
            n01 = node_id(i, j + 1, nx)
            n11 = node_id(i + 1, j + 1, nx)
            triangles.append((n00, n10, n01))
            triangles.append((n10, n11, n01))
    return np.asarray(triangles, dtype=np.int32)


def node_id(i: int, j: int, nx: int) -> int:
    """Return the flattened row-major node index."""

    return j * nx + i


def triangle_local_mats(xy: np.ndarray, k2_const: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Return local stiffness, Helmholtz mass, and area for one linear triangle."""

    x1, y1 = xy[0]
    x2, y2 = xy[1]
    x3, y3 = xy[2]
    det_j = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    area = 0.5 * abs(det_j)
    if area <= 0.0:
        raise FEMConfigurationError("Encountered a degenerate triangle while assembling the FEM mesh.")
    b = np.array([y2 - y3, y3 - y1, y1 - y2], dtype=np.float64)
    c = np.array([x3 - x2, x1 - x3, x2 - x1], dtype=np.float64)
    stiffness = (np.outer(b, b) + np.outer(c, c)) / (4.0 * area)
    mass_shape = (area / 12.0) * (np.ones((3, 3), dtype=np.float64) + np.eye(3, dtype=np.float64))
    mass = k2_const * mass_shape
    return stiffness, mass, float(area)


def edge_robin_local(edge_length: float, k_ref: float) -> np.ndarray:
    """Return the inherited local Robin matrix for one boundary edge."""

    return (-1j * k_ref) * (edge_length / 6.0) * np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.complex128)


def apply_top_dirichlet(matrix, rhs: np.ndarray, nx: int):
    """Apply p=0 on the top row by zeroing rows/columns and setting the diagonal."""

    top_nodes = [node_id(i, 0, nx) for i in range(nx)]
    work = matrix.tolil()
    work[top_nodes, :] = 0.0
    work[:, top_nodes] = 0.0
    for node in top_nodes:
        work[node, node] = 1.0
    rhs[top_nodes] = 0.0
    return work.tocsr()


def sample_velocity(
    velocity: np.ndarray,
    x_m: float,
    y_m: float,
    settings: FEMHelmholtzSettings,
    nx: int,
    ny: int,
) -> float:
    """Sample velocity at a physical coordinate using the requested convention."""

    x_index = np.clip(x_m / settings.domain_size_x_m * (nx - 1), 0.0, nx - 1.0)
    y_index = np.clip(y_m / settings.domain_size_y_m * (ny - 1), 0.0, ny - 1.0)
    if settings.velocity_sampling == "nearest":
        i = int(np.rint(x_index))
        j = int(np.rint(y_index))
        return float(velocity[j, i])

    i0 = int(np.floor(x_index))
    j0 = int(np.floor(y_index))
    i1 = min(i0 + 1, nx - 1)
    j1 = min(j0 + 1, ny - 1)
    tx = float(x_index - i0)
    ty = float(y_index - j0)
    top = (1.0 - tx) * velocity[j0, i0] + tx * velocity[j0, i1]
    bottom = (1.0 - tx) * velocity[j1, i0] + tx * velocity[j1, i1]
    return float((1.0 - ty) * top + ty * bottom)


def gaussian_source(
    x_m: float,
    y_m: float,
    *,
    source_x_m: float,
    source_y_m: float,
    sigma_m: float,
    amplitude: float,
) -> float:
    """Evaluate the inherited Gaussian source density."""

    r2 = (x_m - source_x_m) ** 2 + (y_m - source_y_m) ** 2
    return float(amplitude * np.exp(-0.5 * r2 / (sigma_m**2)))


def bicgstab_compat(matrix, rhs: np.ndarray, *, preconditioner, rtol: float, maxiter: int):
    """Call scipy bicgstab across versions that use rtol or tol."""

    try:
        return spla.bicgstab(matrix, rhs, M=preconditioner, rtol=rtol, maxiter=maxiter)
    except TypeError:
        return spla.bicgstab(matrix, rhs, M=preconditioner, tol=rtol, maxiter=maxiter)
