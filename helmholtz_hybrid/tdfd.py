# Overview:
# Build and time a Devito time-domain finite-difference solver that extracts a
# single 40 Hz Helmholtz pressure sample by accumulating a Fourier component of
# the acoustic wavefield. The code keeps Devito imports lazy so the public repo
# remains usable on machines that do not have the optional GPU solver stack.
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Callable, Literal

import numpy as np


Backend = Literal["env", "cpu", "gpu"]
DFTSign = Literal["positive", "negative"]
VelocityUnits = Literal["auto", "km/s", "m/s"]

DEFAULT_FREQUENCY_HZ = 40.0
DEFAULT_BACKGROUND_KM_S = 1.5
DEFAULT_DOMAIN_SIZE_M = 1_000.0
DEFAULT_SPACE_ORDER = 12
DEFAULT_N_WAVELENGTHS = 4.0
DEFAULT_EXTRA_ABSORBING_M = 50.0
DEFAULT_START_TIME_MS = -100.0
DEFAULT_END_TIME_MS = 4_750.0
DEFAULT_DT_MS = 0.2
DEFAULT_CHUNK_STEPS = 512
DEFAULT_MINIMUM_POINTS_PER_WAVELENGTH = 4.0

ProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True)
class DevitoRuntimeSettings:
    """Runtime environment used before importing Devito."""

    backend: Backend = "env"
    devito_arch: str | None = None
    devito_language: str | None = None
    devito_platform: str | None = None
    cuda_visible_devices: str | None = None
    gpu_fit: bool = True


@dataclass(frozen=True)
class TDFDHelmholtzSettings:
    """Numerical and geometric settings for one TDFD Helmholtz sample."""

    frequency_hz: float = DEFAULT_FREQUENCY_HZ
    domain_size_x_m: float = DEFAULT_DOMAIN_SIZE_M
    domain_size_y_m: float = DEFAULT_DOMAIN_SIZE_M
    source_x_m: float | None = None
    source_y_m: float | None = None
    start_time_ms: float = DEFAULT_START_TIME_MS
    end_time_ms: float = DEFAULT_END_TIME_MS
    dt_ms: float = DEFAULT_DT_MS
    space_order: int = DEFAULT_SPACE_ORDER
    nbl: int | None = None
    absorbing_reference_velocity_km_s: float = DEFAULT_BACKGROUND_KM_S
    absorbing_wavelengths: float = DEFAULT_N_WAVELENGTHS
    absorbing_extra_m: float = DEFAULT_EXTRA_ABSORBING_M
    free_surface: bool = True
    dft_sign: DFTSign = "positive"
    normalize_by_source_spectrum: bool = True
    minimum_points_per_wavelength: float = DEFAULT_MINIMUM_POINTS_PER_WAVELENGTH
    dtype: str = "float32"
    chunk_steps: int = DEFAULT_CHUNK_STEPS
    runtime: DevitoRuntimeSettings = DevitoRuntimeSettings()


@dataclass(frozen=True)
class TDFDHelmholtzDiagnostics:
    """Metadata describing one completed numerical sample."""

    velocity_shape: tuple[int, int]
    padded_shape: tuple[int, int]
    spacing_m: tuple[float, float]
    nbl: int
    nt: int
    critical_dt_ms: float
    points_per_min_wavelength: float
    source_coordinates_m: tuple[float, float]
    source_spectrum: complex
    max_abs_pressure: float
    devito_apply_seconds: float


@dataclass(frozen=True)
class TDFDHelmholtzSample:
    """Complex pressure sample and the diagnostics needed to interpret it."""

    pressure: np.ndarray
    diagnostics: TDFDHelmholtzDiagnostics

    @property
    def pressure_channels(self) -> np.ndarray:
        """Return pressure as float32 [2, ny, nx] channels matching paper data."""

        return np.stack([self.pressure.real, self.pressure.imag], axis=0).astype(np.float32, copy=False)


@dataclass(frozen=True)
class TDFDBenchmarkResult:
    """Timing result for one or more repeated TDFD sample-generation runs."""

    warmup_runs: int
    timed_runs: int
    timed_seconds: tuple[float, ...]
    seconds_per_sample: float
    milliseconds_per_sample: float
    last_sample: TDFDHelmholtzSample


class TDFDConfigurationError(RuntimeError):
    """Raised when the requested solver configuration is invalid or incomplete."""


def configure_devito_environment(runtime: DevitoRuntimeSettings) -> None:
    """Set Devito environment variables before any Devito import occurs."""

    backend_defaults: dict[str, str] = {}
    if runtime.backend == "gpu":
        backend_defaults = {
            "DEVITO_ARCH": "nvc",
            "DEVITO_LANGUAGE": "openacc",
            "DEVITO_PLATFORM": "nvidiaX",
        }
    elif runtime.backend == "cpu":
        backend_defaults = {
            "DEVITO_LANGUAGE": "openmp",
        }

    for key, value in backend_defaults.items():
        os.environ.setdefault(key, value)

    overrides = {
        "DEVITO_ARCH": runtime.devito_arch,
        "DEVITO_LANGUAGE": runtime.devito_language,
        "DEVITO_PLATFORM": runtime.devito_platform,
        "CUDA_VISIBLE_DEVICES": runtime.cuda_visible_devices,
    }
    for key, value in overrides.items():
        if value is not None:
            os.environ[key] = value


def load_velocity_sample(
    path: str | Path,
    *,
    sample_index: int | None = None,
    units: VelocityUnits = "auto",
) -> np.ndarray:
    """Load one 2D velocity sample from a paper raw or processed NumPy array."""

    array = np.load(Path(path), mmap_mode="r")
    if array.ndim == 2:
        if sample_index not in (None, 0):
            raise ValueError(
                f"{path} is a single 2D velocity field, but sample index {sample_index} was requested."
            )
        velocity = np.asarray(array, dtype=np.float32)
    elif array.ndim == 3:
        if sample_index is None:
            raise ValueError(f"{path} contains {array.shape[0]} samples; pass --sample-index.")
        if not 0 <= sample_index < array.shape[0]:
            raise ValueError(
                f"Sample index {sample_index} is out of bounds for {path}, which has {array.shape[0]} samples."
            )
        velocity = np.asarray(array[sample_index], dtype=np.float32)
    else:
        raise ValueError(
            f"Expected a velocity array with shape [ny,nx] or [N,ny,nx], got {array.shape} from {path}."
        )

    return normalize_velocity_units(velocity, units=units)


def normalize_velocity_units(velocity: np.ndarray, *, units: VelocityUnits = "auto") -> np.ndarray:
    """Return velocity in km/s with helpful validation for common data mistakes."""

    if velocity.ndim != 2:
        raise ValueError(f"Expected a 2D velocity field, got shape {velocity.shape}.")
    if not np.isfinite(velocity).all():
        raise ValueError("Velocity field contains NaN or Inf values.")
    if np.any(velocity <= 0.0):
        raise ValueError("Velocity field must be strictly positive everywhere.")

    velocity = np.asarray(velocity, dtype=np.float32)
    if units == "m/s" or (units == "auto" and float(np.nanmedian(velocity)) > 100.0):
        velocity = velocity / 1000.0
    elif units not in ("auto", "km/s"):
        raise ValueError(f"Unknown velocity unit mode '{units}'. Expected auto, km/s, or m/s.")

    vmin = float(np.min(velocity))
    vmax = float(np.max(velocity))
    if vmin < 0.1 or vmax > 10.0:
        raise ValueError(
            "Velocity values should be in km/s after unit conversion; "
            f"got min={vmin:.3f}, max={vmax:.3f}. Pass --velocity-units if the input units are different."
        )
    return velocity


def save_pressure_sample(
    pressure: np.ndarray,
    path: str | Path,
    *,
    channels: bool = True,
) -> None:
    """Save a generated pressure sample in processed-channel or complex layout."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if channels:
        data = np.stack([pressure.real, pressure.imag], axis=0).astype(np.float32, copy=False)
    else:
        data = np.asarray(pressure, dtype=np.complex64)
    np.save(output, data)


def benchmark_tdfd_helmholtz_sample(
    velocity_km_s: np.ndarray,
    settings: TDFDHelmholtzSettings,
    *,
    warmup_runs: int = 0,
    timed_runs: int = 1,
    progress_callback: ProgressCallback | None = None,
) -> TDFDBenchmarkResult:
    """Compile once, then time the sample-generation portion of the TDFD solve."""

    if warmup_runs < 0:
        raise ValueError(f"warmup_runs must be non-negative; got {warmup_runs}.")
    if timed_runs < 1:
        raise ValueError(f"timed_runs must be positive; got {timed_runs}.")

    solver = TDFDHelmholtzSolver(velocity_km_s, settings)
    last_sample: TDFDHelmholtzSample | None = None
    for _ in range(warmup_runs):
        last_sample = solver.solve(progress_callback=progress_callback)

    timed_seconds: list[float] = []
    for _ in range(timed_runs):
        start = time.perf_counter()
        last_sample = solver.solve(progress_callback=progress_callback)
        timed_seconds.append(time.perf_counter() - start)

    assert last_sample is not None
    seconds_per_sample = float(mean(timed_seconds))
    return TDFDBenchmarkResult(
        warmup_runs=warmup_runs,
        timed_runs=timed_runs,
        timed_seconds=tuple(timed_seconds),
        seconds_per_sample=seconds_per_sample,
        milliseconds_per_sample=seconds_per_sample * 1_000.0,
        last_sample=last_sample,
    )


class TDFDHelmholtzSolver:
    """Reusable Devito TDFD solver for one velocity shape and source geometry."""

    def __init__(self, velocity_km_s: np.ndarray, settings: TDFDHelmholtzSettings) -> None:
        self.settings = settings
        self.velocity_yx = normalize_velocity_units(velocity_km_s, units="km/s")
        self._validate_settings()
        configure_devito_environment(settings.runtime)
        self.devito = _import_devito_symbols()
        self.examples = _import_devito_examples()
        self.model = self._make_model()
        if float(settings.dt_ms) > float(self.model.critical_dt):
            raise ValueError(
                f"Requested dt={settings.dt_ms:.4f} ms exceeds Devito CFL limit "
                f"{float(self.model.critical_dt):.4f} ms for this model."
            )
        self.geometry = self._make_geometry()
        self.wavefield, self.pressure_real, self.pressure_imag, self.operator = self._make_operator()
        self.source_spectrum = self._source_spectrum()
        self.compile()

    @property
    def nt(self) -> int:
        return int(self.geometry.time_axis.num)

    @property
    def nbl(self) -> int:
        if self.settings.nbl is not None:
            return int(self.settings.nbl)
        spacing_max = max(self.spacing_m)
        absorbing_m = (
            self.settings.absorbing_reference_velocity_km_s
            * 1000.0
            / self.settings.frequency_hz
            * self.settings.absorbing_wavelengths
            + self.settings.absorbing_extra_m
        )
        return int(np.ceil(absorbing_m / spacing_max))

    @property
    def spacing_m(self) -> tuple[float, float]:
        ny, nx = self.velocity_yx.shape
        return (
            self.settings.domain_size_x_m / float(nx - 1),
            self.settings.domain_size_y_m / float(ny - 1),
        )

    def solve(self, progress_callback: ProgressCallback | None = None) -> TDFDHelmholtzSample:
        """Run one full time-domain solve and return its 40 Hz Fourier component."""

        self.wavefield.data.fill(0.0)
        self.pressure_real.data.fill(0.0)
        self.pressure_imag.data.fill(0.0)

        chunk_steps = max(1, int(self.settings.chunk_steps))
        final_time_m = self.nt - 2
        if final_time_m < 0:
            raise RuntimeError("Time axis has fewer than two samples; increase --end-time-ms or reduce --dt-ms.")

        total_steps = final_time_m + 1
        start_apply = time.perf_counter()
        for time_m in range(0, final_time_m + 1, chunk_steps):
            time_M = min(final_time_m, time_m + chunk_steps - 1)
            self.operator.apply(dt=self.settings.dt_ms, time_m=time_m, time_M=time_M)
            if progress_callback is not None:
                progress_callback(time_M - time_m + 1, total_steps)
        apply_seconds = time.perf_counter() - start_apply

        pressure = self._extract_pressure()
        if self.settings.normalize_by_source_spectrum:
            if abs(self.source_spectrum) < 1e-12:
                raise RuntimeError(
                    "Source Fourier coefficient is too small to normalize the pressure field. "
                    "Use --no-normalize-source or adjust the source/time settings."
                )
            pressure = pressure / np.complex64(self.source_spectrum)

        max_abs_pressure = float(np.max(np.abs(pressure)))
        if not np.isfinite(max_abs_pressure):
            raise RuntimeError("Generated pressure contains NaN or Inf values.")
        if max_abs_pressure == 0.0:
            raise RuntimeError("Generated pressure is identically zero; check source and boundary settings.")

        diagnostics = TDFDHelmholtzDiagnostics(
            velocity_shape=tuple(int(v) for v in self.velocity_yx.shape),
            padded_shape=tuple(int(v) for v in self.wavefield.data.shape[1:]),
            spacing_m=tuple(float(v) for v in self.spacing_m),
            nbl=self.nbl,
            nt=self.nt,
            critical_dt_ms=float(self.model.critical_dt),
            points_per_min_wavelength=self.points_per_min_wavelength,
            source_coordinates_m=tuple(float(v) for v in self.geometry.src_positions[0]),
            source_spectrum=complex(self.source_spectrum),
            max_abs_pressure=max_abs_pressure,
            devito_apply_seconds=apply_seconds,
        )
        return TDFDHelmholtzSample(pressure=pressure.astype(np.complex64, copy=False), diagnostics=diagnostics)

    def compile(self) -> None:
        """Force Devito code generation and JIT compilation outside timed runs."""

        _ = self.operator.cfunction

    def _validate_settings(self) -> None:
        ny, nx = self.velocity_yx.shape
        if ny < 3 or nx < 3:
            raise ValueError(f"Velocity grid must be at least 3x3, got {self.velocity_yx.shape}.")
        positive_values = {
            "frequency_hz": self.settings.frequency_hz,
            "domain_size_x_m": self.settings.domain_size_x_m,
            "domain_size_y_m": self.settings.domain_size_y_m,
            "dt_ms": self.settings.dt_ms,
            "space_order": self.settings.space_order,
            "chunk_steps": self.settings.chunk_steps,
        }
        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive; got {value}.")
        if self.settings.end_time_ms <= self.settings.start_time_ms:
            raise ValueError(
                f"end_time_ms must be greater than start_time_ms; got "
                f"{self.settings.start_time_ms} and {self.settings.end_time_ms}."
            )
        if self.settings.nbl is not None and self.settings.nbl < 0:
            raise ValueError(f"nbl must be non-negative; got {self.settings.nbl}.")
        if self.settings.minimum_points_per_wavelength < 0.0:
            raise ValueError(
                "minimum_points_per_wavelength must be non-negative; "
                f"got {self.settings.minimum_points_per_wavelength}."
            )
        ppw = self.points_per_min_wavelength
        if (
            self.settings.minimum_points_per_wavelength > 0.0
            and ppw < self.settings.minimum_points_per_wavelength
        ):
            raise ValueError(
                f"Grid has only {ppw:.2f} points per minimum-velocity wavelength at "
                f"{self.settings.frequency_hz:g} Hz. Increase resolution, reduce --domain-size-m, "
                "use a lower frequency, or set --allow-low-ppw for diagnostic sweeps."
            )

    @property
    def points_per_min_wavelength(self) -> float:
        """Return grid points per wavelength at the minimum velocity."""

        return float(np.min(self.velocity_yx)) * 1000.0 / (self.settings.frequency_hz * max(self.spacing_m))

    def _make_model(self):
        SeismicModel = self.examples.SeismicModel
        if self.settings.dtype not in ("float32", "float64"):
            raise ValueError(f"dtype must be 'float32' or 'float64'; got {self.settings.dtype}.")
        dtype = np.float32 if self.settings.dtype == "float32" else np.float64
        vp_xz = np.ascontiguousarray(self.velocity_yx.T, dtype=dtype)
        nx, nz = vp_xz.shape
        spacing_x_m, spacing_z_m = self.spacing_m
        return SeismicModel(
            space_order=int(self.settings.space_order),
            vp=vp_xz,
            origin=(0.0, 0.0),
            shape=(nx, nz),
            dtype=dtype,
            spacing=(spacing_x_m, spacing_z_m),
            nbl=self.nbl,
            bcs="damp",
            fs=bool(self.settings.free_surface),
        )

    def _make_geometry(self):
        AcquisitionGeometry = self.examples.AcquisitionGeometry
        source_coordinates = np.asarray([self._source_coordinates_m()], dtype=np.float32)
        dummy_receiver = np.asarray([[0.0, 0.0]], dtype=np.float32)
        geometry = AcquisitionGeometry(
            self.model,
            dummy_receiver,
            source_coordinates,
            t0=float(self.settings.start_time_ms),
            tn=float(self.settings.end_time_ms),
            src_type="Ricker",
            f0=float(self.settings.frequency_hz) / 1000.0,
        )
        geometry.resample(float(self.settings.dt_ms))
        return geometry

    def _make_operator(self):
        Eq = self.devito.Eq
        Function = self.devito.Function
        Inc = self.devito.Inc
        Operator = self.devito.Operator
        TimeFunction = self.devito.TimeFunction
        cos = self.devito.cos
        sin = self.devito.sin
        solve = self.devito.solve
        freesurface = self.examples.freesurface

        space_order = int(self.settings.space_order)
        wavefield = TimeFunction(
            name="p",
            grid=self.model.grid,
            space_order=space_order,
            time_order=2,
        )
        pressure_real = Function(name="pressure_real", grid=self.model.grid, space_order=0)
        pressure_imag = Function(name="pressure_imag", grid=self.model.grid, space_order=0)

        pde = self.model.m * wavefield.dt2 - wavefield.laplace + self.model.damp * wavefield.dt
        update = solve(pde, wavefield.forward)
        physdomain = self.model.grid.subdomains["physdomain"]
        stencil = Eq(wavefield.forward, update, subdomain=physdomain)
        equations = [stencil]
        if self.settings.free_surface:
            equations.extend(freesurface(self.model, Eq(wavefield.forward, update)))

        src = self.geometry.src
        dt_symbol = self.model.grid.stepping_dim.spacing
        source = src.inject(field=wavefield.forward, expr=src * dt_symbol**2 / self.model.m)

        sign = 1.0 if self.settings.dft_sign == "positive" else -1.0
        omega = 2.0 * np.pi * float(self.settings.frequency_hz)
        time_dim = self.model.grid.time_dim
        time_s = (time_dim + 1) * dt_symbol / 1000.0 + float(self.settings.start_time_ms) / 1000.0
        phase = sign * omega * time_s
        dft_updates = [
            Inc(pressure_real, wavefield.forward * cos(phase) * dt_symbol / 1000.0, subdomain=physdomain),
            Inc(pressure_imag, wavefield.forward * sin(phase) * dt_symbol / 1000.0, subdomain=physdomain),
        ]

        opt: tuple[str, dict[str, object]] | str
        if self.settings.runtime.backend == "gpu" and self.settings.runtime.gpu_fit:
            opt = ("advanced", {"gpu-fit": [wavefield, src, pressure_real, pressure_imag]})
        else:
            opt = "advanced"

        operator = Operator(
            equations + source + dft_updates,
            subs=self.model.spacing_map,
            name="TDFDHelmholtz40Hz",
            opt=opt,
        )
        return wavefield, pressure_real, pressure_imag, operator

    def _source_coordinates_m(self) -> tuple[float, float]:
        _, spacing_y_m = self.spacing_m
        x_m = self.settings.source_x_m
        y_m = self.settings.source_y_m
        if x_m is None:
            x_m = 0.5 * self.settings.domain_size_x_m
        if y_m is None:
            # A point one grid cell below the free surface avoids canceling a
            # pressure source on a Dirichlet boundary while matching "near
            # center of the free boundary" in the paper setup.
            y_m = spacing_y_m if self.settings.free_surface else self.nbl * spacing_y_m
        if not 0.0 <= x_m <= self.settings.domain_size_x_m:
            raise ValueError(f"Source x-coordinate {x_m} m is outside [0, {self.settings.domain_size_x_m}].")
        if not 0.0 <= y_m <= self.settings.domain_size_y_m:
            raise ValueError(f"Source y-coordinate {y_m} m is outside [0, {self.settings.domain_size_y_m}].")
        return float(x_m), float(y_m)

    def _source_spectrum(self) -> complex:
        src_values = np.asarray(self.geometry.src.data[:, 0], dtype=np.float64)
        times_ms = float(self.settings.start_time_ms) + np.arange(src_values.shape[0]) * float(self.settings.dt_ms)
        sign = 1.0 if self.settings.dft_sign == "positive" else -1.0
        phase = sign * 2.0 * np.pi * float(self.settings.frequency_hz) * times_ms / 1000.0
        return complex(np.sum(src_values * np.exp(1j * phase)) * float(self.settings.dt_ms) / 1000.0)

    def _extract_pressure(self) -> np.ndarray:
        ny, nx = self.velocity_yx.shape
        x_start = self.nbl
        z_start = 0 if self.settings.free_surface else self.nbl
        x_slice = slice(x_start, x_start + nx)
        z_slice = slice(z_start, z_start + ny)
        real_xz = np.asarray(self.pressure_real.data[x_slice, z_slice], dtype=np.float32)
        imag_xz = np.asarray(self.pressure_imag.data[x_slice, z_slice], dtype=np.float32)
        return (real_xz + 1j * imag_xz).T


def _import_devito_symbols():
    try:
        from devito import Eq, Function, Inc, Operator, TimeFunction, cos, sin, solve
    except ImportError as error:
        raise TDFDConfigurationError(
            "Devito is not installed. Activate .venv and install the optional solver dependencies with "
            'python -m pip install -e ".[tdfd]".'
        ) from error

    class Symbols:
        pass

    symbols = Symbols()
    symbols.Eq = Eq
    symbols.Function = Function
    symbols.Inc = Inc
    symbols.Operator = Operator
    symbols.TimeFunction = TimeFunction
    symbols.cos = cos
    symbols.sin = sin
    symbols.solve = solve
    return symbols


def _import_devito_examples():
    try:
        from examples.seismic import AcquisitionGeometry
        from examples.seismic.acoustic.operators import freesurface
        from examples.seismic.model import SeismicModel
    except ImportError as error:
        raise TDFDConfigurationError(
            "Devito seismic examples are not importable. Install the optional solver dependencies with "
            'python -m pip install -e ".[tdfd]"; if using a source Devito checkout, ensure its examples/ '
            "package is on PYTHONPATH."
        ) from error

    class Examples:
        pass

    examples = Examples()
    examples.AcquisitionGeometry = AcquisitionGeometry
    examples.SeismicModel = SeismicModel
    examples.freesurface = freesurface
    return examples
