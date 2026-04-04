"""
CFDWind — RotorPy wind object backed by van der Laan LES NetCDF data.

Data source
-----------
van der Laan & Andersen (2018): hub-height horizontal plane of a single NREL-5MW
wake at U0 = 8 m/s.  Two atmospheric conditions (low TI / high TI), each with
6 random-seed bins stored as NetCDF files.

File structure  (per .nc file, ~2 GB):
    Dimensions : t (2500) × x (286) × y (121)
    Variables  : U, V, W  [m/s]   shape = (t, x, y)
    Time       : 0.24 → 600 s,  dt = 0.24 s
    Spatial    : x ∈ [-126, 1071] m  (dx=4.2 m)
                 y ∈ [-252,  252] m  (dy=4.2 m)
    Turbine    : rotor diameter D = 126 m, hub at x=0, y=0

Coordinate convention
---------------------
LES frame  : x = streamwise,  y = lateral,  z = vertical (W)
RotorPy    : x = world East,  y = world North,  z = up
  → Caller sets ``origin_les`` to map UAV world-origin to an LES position.

Usage
-----
>>> from rotorpy.wind.cfd_wind import CFDWind
>>> wind = CFDWind("case6_highTi_UWV_10min_bin1.nc",
...                origin_les=(3*126, 0))   # hover 3D downstream
>>> # Inside simulate():
>>> w = wind.update(t, uav_position)        # returns np.array([wx, wy, wz])
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator


class CFDWind:
    """
    Wind profile backed by a 2-D (horizontal hub-height) LES snapshot stack.

    Parameters
    ----------
    nc_file : str
        Path to van der Laan LES NetCDF file
        (e.g. ``case6_highTi_UWV_10min_bin1.nc``).
    origin_les : tuple of float, optional
        (x0, y0) in the LES coordinate frame [m] that corresponds to the
        UAV's world-frame origin (0, 0, *).  Default: (378, 0) = 3D downstream.
    t_offset : float, optional
        Shift the simulation clock by this many seconds into the LES record
        before querying.  Default: 0 (start at the first LES snapshot).
    loop_time : bool, optional
        If True, wrap the query time modulo the LES record length when the
        simulation exceeds 600 s.  Default: False (clamp at boundaries).
    mean_only : bool, optional
        If True, return the time-averaged mean field (no turbulent
        fluctuations).  Useful for isolating mean wake deficit effects.
        Default: False.
    """

    # Physical constants (van der Laan / NREL-5MW dataset)
    D   = 126.0         # rotor diameter [m]
    R   = 63.0          # rotor radius   [m]  = D / 2
    U0  = 8.0           # hub-height freestream velocity [m/s]

    def __init__(self, nc_file, origin_les=(3*126, 0),
                 t_offset=0.0, loop_time=False, mean_only=False):

        import xarray as xr

        print(f"[CFDWind] Loading {nc_file} …", flush=True)
        ds = xr.open_dataset(nc_file)

        t_arr = ds['t'].values.astype(np.float64)   # (2500,)
        x_arr = ds['x'].values.astype(np.float64)   # (286,)
        y_arr = ds['y'].values.astype(np.float64)   # (121,)

        # Load all three velocity components — shape (t, x, y)
        U = ds['U'].values.astype(np.float32)
        V = ds['V'].values.astype(np.float32)
        W = ds['W'].values.astype(np.float32)
        ds.close()

        self.t_arr = t_arr
        self.x_arr = x_arr
        self.y_arr = y_arr
        self.origin_les = np.asarray(origin_les, dtype=np.float64)
        self.t_offset   = float(t_offset)
        self.loop_time  = bool(loop_time)
        self.mean_only  = bool(mean_only)

        self.t_min = t_arr[0]
        self.t_max = t_arr[-1]

        pts = (t_arr, x_arr, y_arr)
        kwargs = dict(method='linear', bounds_error=False, fill_value=None)

        if mean_only:
            # Pre-compute time-mean and build a 2-D interpolator
            from scipy.interpolate import RegularGridInterpolator as RGI
            U_mean = U.mean(axis=0)   # (286, 121)
            V_mean = V.mean(axis=0)
            W_mean = W.mean(axis=0)
            pts2d = (x_arr, y_arr)
            self._iU = RGI(pts2d, U_mean, **kwargs)
            self._iV = RGI(pts2d, V_mean, **kwargs)
            self._iW = RGI(pts2d, W_mean, **kwargs)
            print("[CFDWind] Mean-field mode — 2-D interpolators built.")
        else:
            self._iU = RegularGridInterpolator(pts, U, **kwargs)
            self._iV = RegularGridInterpolator(pts, V, **kwargs)
            self._iW = RegularGridInterpolator(pts, W, **kwargs)
            mem_mb = (U.nbytes + V.nbytes + W.nbytes) / 1e6
            print(f"[CFDWind] 3-D interpolators built  ({mem_mb:.0f} MB loaded).")

    # ------------------------------------------------------------------
    def update(self, t, position):
        """
        Return the LES wind vector at simulation time ``t`` and UAV
        world-frame ``position``.

        Parameters
        ----------
        t : float
            Simulation time [s].
        position : array-like, shape (3,)
            UAV position in world frame [m]: (x_world, y_world, z_world).
            Only (x, y) are used; the LES is a hub-height horizontal plane.

        Returns
        -------
        wind : np.ndarray, shape (3,)
            Wind velocity [wx, wy, wz] in world frame [m/s].
            wx ≡ U (streamwise = world-x), wy ≡ V (lateral = world-y),
            wz ≡ W (vertical = world-z).
        """
        # Map UAV world position → LES coordinates
        x_les = float(position[0]) + self.origin_les[0]
        y_les = float(position[1]) + self.origin_les[1]

        if self.mean_only:
            pt = np.array([[x_les, y_les]])
        else:
            t_les = t + self.t_offset
            if self.loop_time:
                span = self.t_max - self.t_min
                t_les = self.t_min + (t_les - self.t_min) % span
            else:
                t_les = np.clip(t_les, self.t_min, self.t_max)
            pt = np.array([[t_les, x_les, y_les]])

        wx = float(self._iU(pt)[0])
        wy = float(self._iV(pt)[0])
        wz = float(self._iW(pt)[0])

        # Guard against NaN (domain edge) — fall back to freestream
        if not np.isfinite(wx): wx = self.U0
        if not np.isfinite(wy): wy = 0.0
        if not np.isfinite(wz): wz = 0.0

        return np.array([wx, wy, wz])

    # ------------------------------------------------------------------
    def mean_wind_at(self, x_world, y_world):
        """
        Convenience: return the time-averaged LES wind at a fixed world
        position (no UAV position offset applied).  Useful for checking
        the mean wake deficit at a candidate hover point.
        """
        x_les = float(x_world) + self.origin_les[0]
        y_les = float(y_world) + self.origin_les[1]

        if self.mean_only:
            pt = np.array([[x_les, y_les]])
            return np.array([float(self._iU(pt)[0]),
                             float(self._iV(pt)[0]),
                             float(self._iW(pt)[0])])
        else:
            # Query at mid-point of time record
            t_mid = 0.5 * (self.t_min + self.t_max)
            # Use full time dimension
            from scipy.interpolate import RegularGridInterpolator as RGI
            pts2d = (self.x_arr, self.y_arr)
            # This is slow — only for diagnostics
            return self.update(t_mid - self.t_offset, np.array([x_world, y_world, 0.0]))

    # ------------------------------------------------------------------
    @classmethod
    def from_params(cls, nc_file, wake_position=5.0, unit='R',
                    lateral_offset=0.0, **kwargs):
        """
        Convenience constructor: place the UAV world-frame origin at a
        specified downstream position behind the wind turbine.

        Parameters
        ----------
        nc_file : str
            Path to van der Laan LES NetCDF file.
        wake_position : float
            Downstream distance in units of ``unit``.  Default: 5.0 R.
        unit : {'R', 'D', 'm'}
            Length unit for ``wake_position``:
              'R'  — rotor radius  (R = 63 m, NREL-5MW)
              'D'  — rotor diameter (D = 126 m)
              'm'  — metres
            Default: 'R'.
        lateral_offset : float
            Lateral (y) offset from wake centreline in the same ``unit``.
            Positive = port side.  Default: 0 (centreline).
        """
        scale = {'R': cls.R, 'D': cls.D, 'm': 1.0}.get(unit)
        if scale is None:
            raise ValueError(f"unit must be 'R', 'D', or 'm', got {unit!r}")
        x0 = wake_position  * scale
        y0 = lateral_offset * scale
        print(f"[CFDWind] origin_les = ({x0:.1f}, {y0:.1f}) m  "
              f"= ({wake_position:.1f}{unit}, {lateral_offset:.1f}{unit})")
        return cls(nc_file, origin_les=(x0, y0), **kwargs)
