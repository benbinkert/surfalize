import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

# ACHTUNG: Filter hier von mir nicht gefixed worden, gibt falsche dinge auss
class Profile:
    def __init__(self, height_data, step, length_um, axis_data=None, axis_label=None, title=None):
        self.data = np.asarray(height_data, dtype=float)
        self.step = float(step)
        self.length_um = float(length_um)

        self.axis_data = None if axis_data is None else np.asarray(axis_data, dtype=float)
        self.axis_label = axis_label
        self.title = title

    def __repr__(self):
        return f"{self.__class__.__name__}({self.length_um:.2f} µm)"

    def _repr_png_(self):
        self.show_real()

    def copy(self):
        return Profile(
            height_data=self.data.copy(),
            step=self.step,
            length_um=self.length_um,
            axis_data=None if self.axis_data is None else self.axis_data.copy(),
            axis_label=self.axis_label,
            title=self.title,
        )

    def period(self):
        fft = np.abs(np.fft.fft(self.data))
        freq = np.fft.fftfreq(self.data.shape[0], d=self.step)
        peaks, properties = find_peaks(fft.flatten(), distance=10, prominence=10)
        prominences = properties["prominences"]
        sorted_indices = np.argsort(prominences)[::-1]
        peaks_sorted = peaks[sorted_indices]
        period = 1 / np.abs(freq[peaks_sorted[0]])
        return period

    def Ra(self):
        z = self.data - np.mean(self.data)
        return np.mean(np.abs(z))

    def Rq(self):
        z = self.data - np.mean(self.data)
        return np.sqrt(np.mean(z ** 2))

    def Rp(self):
        z = self.data - np.mean(self.data)
        return np.max(z)

    def Rv(self):
        z = self.data - np.mean(self.data)
        return np.abs(np.min(z))

    def Rz(self):
        return self.Rp() + self.Rv()

    def Rsk(self):
        rq = self.Rq()
        if rq == 0:
            return np.nan
        z = self.data - np.mean(self.data)
        return np.mean(z ** 3) / (rq ** 3)

    def Rku(self):
        rq = self.Rq()
        if rq == 0:
            return np.nan
        z = self.data - np.mean(self.data)
        return np.mean(z ** 4) / (rq ** 4)

    def Wa(self):
        return self.Ra()

    def Wq(self):
        return self.Rq()

    def Wp(self):
        return self.Rp()

    def Wv(self):
        return self.Rv()

    def Wz(self):
        return self.Rz()

    def Wsk(self):
        return self.Rsk()

    def Wku(self):
        return self.Rku()

    def plot_2d(self, ax=None, **plot_kwargs):
        if ax is None:
            fig, ax = plt.subplots(dpi=150, figsize=(10, 3))
        else:
            fig = ax.figure

        x = np.linspace(0, self.length_um, self.data.size)
        ax.plot(x, self.data, **({"c": "k", "lw": 0.8} | plot_kwargs))
        ax.set_xlim(x[0], x[-1])
        ax.set_xlabel("Profilweg [µm]")
        ax.set_ylabel("z [µm]")

        if self.title is not None:
            ax.set_title(self.title)

        return fig, ax

    def show(self, block=True):
        self.plot_2d()
        plt.show(block=block)

    def plot_real(self, ax=None, **plot_kwargs):
        if self.axis_data is None:
            return self.plot_2d(ax=ax, **plot_kwargs)

        if ax is None:
            fig, ax = plt.subplots(dpi=150, figsize=(10, 3))
        else:
            fig = ax.figure

        ax.plot(self.axis_data, self.data, **({"c": "k", "lw": 0.8} | plot_kwargs))
        ax.set_xlim(self.axis_data[0], self.axis_data[-1])
        ax.set_xlabel(self.axis_label if self.axis_label is not None else "Koordinate [µm]")
        ax.set_ylabel("z [µm]")

        if self.title is not None:
            ax.set_title(self.title)

        return fig, ax

    def show_real(self, block=True):
        self.plot_real()
        plt.show(block=block)

    def detrend_polynomial(self, degree=1, inplace=False, return_trend=False):
        z = np.asarray(self.data, dtype=float)

        if self.axis_data is not None:
            x = np.asarray(self.axis_data, dtype=float)
        else:
            x = np.linspace(0.0, float(self.length_um), z.size)

        if x.size != z.size:
            raise ValueError("axis_data und data müssen gleiche Länge haben.")

        mask = np.isfinite(z)
        if np.count_nonzero(mask) < degree + 1:
            raise ValueError("Zu wenige gültige Punkte für den gewählten Polynomgrad.")

        x_valid = x[mask]
        z_valid = z[mask]

        x0 = x_valid.mean()
        x_norm = x_valid - x0
        denom = np.max(np.abs(x_norm))
        if denom == 0:
            raise ValueError("x-Koordinate ist konstant.")
        x_norm /= denom

        A = np.vander(x_norm, N=degree + 1, increasing=True)
        coeffs, _, _, _ = np.linalg.lstsq(A, z_valid, rcond=None)

        x_all = (x - x0) / denom
        A_all = np.vander(x_all, N=degree + 1, increasing=True)
        trend = A_all @ coeffs

        detrended = np.where(np.isfinite(z), z - trend, np.nan)

        if inplace:
            self.data = detrended
            out_profile = self
        else:
            out_profile = Profile(
                height_data=detrended,
                step=self.step,
                length_um=self.length_um,
                axis_data=self.axis_data,
                axis_label=self.axis_label,
                title=self.title,
            )

        if return_trend:
            trend_profile = Profile(
                height_data=trend,
                step=self.step,
                length_um=self.length_um,
                axis_data=self.axis_data,
                axis_label=self.axis_label,
                title="Trend" if self.title is None else f"Trend – {self.title}",
            )
            return out_profile, trend_profile

        return out_profile

    def level(self, return_trend=False, inplace=False):
        return self.detrend_polynomial(degree=0, inplace=inplace, return_trend=return_trend)

    def threshold_percentile(self, upper=0.25, lower=0.25, inplace=False):
        z = np.asarray(self.data, dtype=float)
        mask = np.isfinite(z)
        z_valid = z[mask]

        if z_valid.size == 0:
            raise ValueError("Keine gültigen Punkte für threshold_percentile vorhanden.")

        low_val = np.percentile(z_valid, lower)
        high_val = np.percentile(z_valid, 100.0 - upper)

        z_new = z.copy()
        z_new[z_new < low_val] = np.nan
        z_new[z_new > high_val] = np.nan

        if inplace:
            self.data = z_new
            return self

        return Profile(
            height_data=z_new,
            step=self.step,
            length_um=self.length_um,
            axis_data=self.axis_data,
            axis_label=self.axis_label,
            title=self.title,
        )

    def fill_nonmeasured_linear(self, inplace=False):
        z = np.asarray(self.data, dtype=float).copy()

        if self.axis_data is not None:
            x = np.asarray(self.axis_data, dtype=float)
        else:
            x = np.arange(z.size, dtype=float) * self.step

        mask = np.isfinite(z)
        if np.count_nonzero(mask) < 2:
            raise ValueError("Zu wenige gültige Punkte für lineare Interpolation.")

        z_filled = z.copy()
        z_filled[~mask] = np.interp(x[~mask], x[mask], z[mask])

        if inplace:
            self.data = z_filled
            return self

        return Profile(
            height_data=z_filled,
            step=self.step,
            length_um=self.length_um,
            axis_data=self.axis_data,
            axis_label=self.axis_label,
            title=self.title,
        )


def preprocess_profile(profile: Profile,
                       do_level=True,
                       do_detrend=True,
                       degree=2,
                       do_threshold=True,
                       threshold_upper=0.25,
                       threshold_lower=0.25,
                       do_fill=True):
    out = profile.copy()

    if do_level:
        out = out.level()

    if do_detrend:
        out = out.detrend_polynomial(degree=degree)

    if do_threshold:
        out = out.threshold_percentile(
            upper=threshold_upper,
            lower=threshold_lower
        )

    if do_fill:
        out = out.fill_nonmeasured_linear()

    return out


def gaussian_lowpass_profile(profile: Profile, cutoff_um: float):
    sigma_px = cutoff_um / profile.step / (2.0 * np.pi)
    sigma_px = max(sigma_px, 0.5)

    z_lp = gaussian_filter1d(profile.data, sigma=sigma_px, mode="nearest")

    return Profile(
        height_data=z_lp,
        step=profile.step,
        length_um=profile.length_um,
        axis_data=profile.axis_data,
        axis_label=profile.axis_label,
        title=profile.title,
    )


def gaussian_highpass_profile(profile: Profile, cutoff_um: float):
    low = gaussian_lowpass_profile(profile, cutoff_um)
    z_hp = profile.data - low.data

    return Profile(
        height_data=z_hp,
        step=profile.step,
        length_um=profile.length_um,
        axis_data=profile.axis_data,
        axis_label=profile.axis_label,
        title=profile.title,
    )


def split_roughness_waviness(profile: Profile, nis_um=2.5, nic_um=800.0):
    """
    S-Filter: Nis = 2.5 µm
    Trennung Rauheit/Welligkeit: Nic = 800 µm = 0.8 mm

    Nach dieser Logik:
    - Welligkeit = Anteile über 0.8 mm
    - Rauheit = Anteile unter 0.8 mm
    """
    profile_s = gaussian_highpass_profile(profile, cutoff_um=nis_um)
    waviness = gaussian_lowpass_profile(profile_s, cutoff_um=nic_um)

    roughness_data = profile_s.data - waviness.data
    roughness = Profile(
        height_data=roughness_data,
        step=profile.step,
        length_um=profile.length_um,
        axis_data=profile.axis_data,
        axis_label=profile.axis_label,
        title=profile.title,
    )

    return profile_s, roughness, waviness


def profile_metrics_dict(profile: Profile, prefix="R"):
    if prefix.upper() == "R":
        return {
            "Ra_um": profile.Ra(),
            "Rq_um": profile.Rq(),
            "Rp_um": profile.Rp(),
            "Rv_um": profile.Rv(),
            "Rz_um": profile.Rz(),
            "Rsk": profile.Rsk(),
            "Rku": profile.Rku(),
        }
    elif prefix.upper() == "W":
        return {
            "Wa_um": profile.Wa(),
            "Wq_um": profile.Wq(),
            "Wp_um": profile.Wp(),
            "Wv_um": profile.Wv(),
            "Wz_um": profile.Wz(),
            "Wsk": profile.Wsk(),
            "Wku": profile.Wku(),
        }
    else:
        raise ValueError("prefix muss 'R' oder 'W' sein.")