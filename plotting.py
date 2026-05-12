import logging
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.linewidth": 1.2,
    "lines.linewidth": 1.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "legend.frameon": False,
})

class stats_plots:
    def __init__(self, logger: logging.Logger, corr_masked: np.ndarray, names: list[str], band_index: int,
                q_vals: np.ndarray, i_vals: np.ndarray, delta_chi2: np.ndarray, q_best: float,
                i_best: float, delta_q: np.ndarray, rv_q_mean: float, rv_q_sigma: float,
                gaia_id: str) -> None:
        self.logger = logger
        self.corr_masked = corr_masked
        self.names = names                  
        self.band_index = band_index
        self.q_vals = q_vals
        self.i_vals = i_vals
        self.delta_chi2 = delta_chi2
        self.q_best = q_best
        self.i_best = i_best
        self.delta_q = delta_q
        self.rv_q_mean = rv_q_mean
        self.rv_q_sigma = rv_q_sigma
        self.gaia_id = gaia_id

    def correlation(self) -> None:
        fig1, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(self.corr_masked, cmap='coolwarm', vmin=-1, vmax=1, interpolation='none')
        cbar = fig1.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Correlation', fontsize=9)
        cbar.ax.tick_params(labelsize=8)
        ax.set_xticks(np.arange(len(self.names)))
        ax.set_yticks(np.arange(len(self.names)))
        ax.set_xticklabels(self.names, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(self.names, fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        fig1.savefig(f"{self.gaia_id}_correlation_{self.band_index}.png", dpi=300)
        fig1.savefig(f"{self.gaia_id}_correlation_{self.band_index}.pdf", dpi=300)
        self.logger.info(f"saved - {self.gaia_id}_correlation_{self.band_index}.png/.pdf")
        plt.close(fig1)

    def chi2_2d_map(self) -> None:
        levels_2d = [2.30, 6.17, 11.8]
        fig2, ax1 = plt.subplots(figsize=(3.5, 6.0))
        cf = ax1.contourf(self.q_vals, self.i_vals, self.delta_chi2.T, levels=50, cmap='viridis')
        cs = ax1.contour(self.q_vals, self.i_vals, self.delta_chi2.T, levels=levels_2d, colors='black')
        plt.clabel(cs, fmt='%.2f', fontsize=8)
        plt.colorbar(cf, ax=ax1, label=r"$\Delta \chi^2$")
        ax1.plot(self.q_best, self.i_best, 'ro', markersize=3)
        ax1.set_xlabel(r"$q$")
        ax1.set_ylabel(r"$i\ (^\circ)$")
        plt.tight_layout()
        fig2.savefig(f"{self.gaia_id}_chi2_2d_map.png", dpi=300)
        fig2.savefig(f"{self.gaia_id}_chi2_2d_map.pdf", dpi=300)
        self.logger.info(f"saved - {self.gaia_id}_chi2_2d_map.png/.pdf")
        plt.close(fig2)
    
    def marginalised_profile(self) -> None:
        fig3, ax2 = plt.subplots(figsize=(3.5, 6.0))
        ax2.plot(self.q_vals, self.delta_q, 'k-', lw=1.5)
        q_fine = np.linspace(self.q_vals[0], self.q_vals[-1], 500)
        rv_gaussian = np.exp(-0.5 * ((q_fine - self.rv_q_mean) / self.rv_q_sigma)**2)
        ax2_twin = ax2.twinx()
        ax2_twin.fill_between(q_fine, rv_gaussian, alpha=0.2, color='steelblue')
        ax2_twin.plot(q_fine, rv_gaussian, color='steelblue', lw=1.0)
        ax2_twin.set_ylim(0, 3)
        ax2_twin.set_yticks([])
        ax2_twin.spines['right'].set_visible(False) 
        ax2.axvline(self.rv_q_mean, color='red', lw=0.8, ls='--', alpha=0.6)
        ax2.set_xlabel(r"$q$")
        ax2.set_ylabel(r"$\Delta \chi^2$ (marginalised over $i$)")
        ax2.set_ylim(0, 30)
        plt.tight_layout()
        fig3.savefig(f"{self.gaia_id}_1d_marginalised_profile.png")
        fig3.savefig(f"{self.gaia_id}_1d_marginalised_profile.pdf")
        self.logger.info(f"saved - {self.gaia_id}_1d_marginalised_profile.png/.pdf")
        plt.close(fig3)

class lcurve_model_plot:
    def __init__(self, logger: logging.Logger, fig_name: str, results: dict) -> None:
        self.logger = logger
        self.fig_name = fig_name
        self.results = results

    def lc_with_model(self, use_phase: bool = True) -> None:
        fig = plt.figure(figsize=(10, 12))
        gs = fig.add_gridspec(6, 1, height_ratios=[3, 1, 3, 1, 3, 1], hspace=0.05)
        plot_order = [1, 2, 3]
        band_colours = {1: 'blue', 2: 'green', 3: 'red'}
        for plot_idx, band_idx in enumerate(plot_order):
            ax_main = fig.add_subplot(gs[plot_idx*2, 0])
            ax_res  = fig.add_subplot(gs[plot_idx*2 + 1, 0], sharex=ax_main)
            res = self.results[band_idx]
            colour = band_colours[band_idx]
            if use_phase:
                x_data = res['phase']
                x_model = res['phase_model']
                xlim = (0, 1)
                xlabel = r"Phase ($\sigma$)"
            else:
                x_data = res['time'] - res['t0_nearby']
                x_model = res['time_model']
                xlim = (0, max(x_data))
                xlabel= f"BMJD (TDB) - {res['t0_nearby']:.2f}"
            ax_main.plot(x_model, res['flux_model'],
                        color='black', lw=2, zorder=5)
            ax_main.errorbar(x_data, res['flux'],
                            yerr=res['flux_err'],
                            color=colour, fmt='o', markersize=3, alpha=0.7,
                            zorder=3)
            ax_main.set_ylabel("Normalized Flux")
            plt.setp(ax_main.get_xticklabels(), visible=False)
            ax_main.set_ylim(-0.1, 1.3)
            ax_main.set_xlim(*xlim)
            oot_mask = res['flux'] > 0.85
            rse = np.sqrt(np.sum((res['flux'][oot_mask] - res['flux_model'][oot_mask])**2) /
                            (len(res['flux_model'][oot_mask]) - 2))
            residuals_sigma = (res['flux'] - res['flux_model']) / rse
            ax_res.errorbar(x_data, residuals_sigma,
                        color=colour, fmt='o', markersize=3, alpha=0.7)
            ax_res.axhline(0, color='black', linestyle='--', alpha=0.3)
            ax_res.axhline(2.5, color='black', linestyle='--', alpha=0.3)
            ax_res.axhline(-2.5, color='black', linestyle='--', alpha=0.3)
            y_med = np.median(residuals_sigma)
            y_mad = np.median(np.abs(residuals_sigma - y_med))
            ax_res.set_ylim(y_med - 8 * y_mad, y_med + 8 * y_mad)
            ax_res.set_ylabel(r"Residuals ($\sigma$)")
            if plot_idx < 2:
                plt.setp(ax_res.get_xticklabels(), visible=False)
            ax_res.set_xlabel(xlabel, fontsize=16)
        plt.savefig(f"{self.fig_name}.png", bbox_inches='tight', dpi=150)
        plt.savefig(f"{self.fig_name}.pdf", bbox_inches='tight', dpi=150)
        self.logger.info(f"Saved: {self.fig_name}.png/.pdf")
        plt.close(fig)