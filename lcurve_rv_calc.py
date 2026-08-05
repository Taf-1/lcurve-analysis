import logging
import numpy as np

class xshooter_params:
    def __init__(self, logger, rv_config, inclination: float | None = 90,
             inclination_err: float | None = 0.5) -> None:
        self.logger = logger
        self.rv_config = rv_config
        self.inclination = inclination
        self.inclination_err = inclination_err

    def load_rv_parameters(self) -> dict[str, tuple[float, float]]:
        params = []
        names = []
        steps = []
        self.logger.debug(f"Reading {self.rv_config} to extract RV params")
        with open(self.rv_config, "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 4 or parts[1] != "=":
                    continue
                names.append(parts[0])
                params.append(float(parts[2]))
                steps.append(float(parts[3]))
        rvs = {name: (param, step) for name, param, step in zip(names, params, steps)}
        return rvs

    def q_n_velocityscale(self) -> tuple[float, float, float, float]:
        rvs = self.load_rv_parameters()
        k1_dict, k2_dict = rvs["K1"], rvs["K2"]
        q = k1_dict[0] / k2_dict[0]
        q_err = q * np.sqrt((k1_dict[1]/k1_dict[0])**2 + (k2_dict[1]/k2_dict[0])**2)
        velocity_scale = (k1_dict[0] + k2_dict[0]) / np.sin(np.radians(self.inclination))
        p1_err = np.sqrt(k1_dict[1]**2 + k2_dict[1]**2)
        sigma_i = np.radians(self.inclination_err)
        p2_err = np.abs(np.cos(np.radians(self.inclination))) * sigma_i 
        v_err = velocity_scale * np.sqrt((p1_err/(k1_dict[0] + k2_dict[0]))**2 + (p2_err/np.sin(np.radians(self.inclination)))**2)
        self.logger.debug(f"Calculated mass ratio and velocity scale \n - q = {q} ± {q_err} \n - velocity scale = {velocity_scale} ± {v_err}")
        return q, q_err, velocity_scale, v_err