import logging
import numpy as np

class xshooter_params:
    def __init__(self, logger: logging.Logger, rv_config: str, phase: list[float]) -> None:
        self.logger = logger
        self.rv_config = rv_config
        self.phase = phase

    def load_rv_parameters(self) -> dict[str, tuple[float, float]]:
        params = []
        names = []
        steps = []
        self.logger.info(f"Reading {self.rv_config} to extract RV params")
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
        velocity_scale = k1_dict[0] / (q/(1+q))
        p2_err = q_err / (1+q)**2
        v_err = velocity_scale * np.sqrt((k1_dict[1]/k1_dict[0])**2 + (p2_err/(q/(1+q)))**2)
        self.logger.info(f"Calculated mass ratio and velocity scale \n - q = {q} ± {q_err} \n - velocity scale = {velocity_scale} ± {v_err}")
        return q, q_err, velocity_scale, v_err

    def velocity(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        q, q_err, vs, vs_error = self.q_n_velocityscale()
        phase = np.asarray(self.phase)
        K1 = (q/(1+q)) * vs
        K2 = (1/(1+q)) * vs
        k1p1_err = q_err / (1+q)**2
        k1_error = K1 * np.sqrt((k1p1_err/(q/(1+q)))**2 + (vs_error/vs)**2)
        k2p1_err = q_err / (1+q)**2
        k2_error = K2 * np.sqrt((k2p1_err/(1/(1+q)))**2 + (vs_error/vs)**2)
        primary_v = K1 * np.sin(2*np.pi*phase)
        companion_v = -K2 * np.sin(2*np.pi*phase)
        prim_v_err = k1_error * np.abs(np.sin(2*np.pi*phase))
        comp_v_err = k2_error * np.abs(np.sin(2*np.pi*phase))
        self.logger.info("Calculated velocity components from mass ratio and velocity scale")
        return primary_v, prim_v_err, companion_v, comp_v_err
