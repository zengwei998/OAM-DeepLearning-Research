import yaml
import torch
from pathlib import Path


def load_config():
    config_path = Path(__file__).parents[1] / "config" / "frozen_v1.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config



if __name__ == "__main__":

    print("=" * 50)
    print("OAM MC-VWLS Environment Check")
    print("=" * 50)


    # 读取配置
    cfg = load_config()

    print("\n[Config]")
    print("Wavelength:",
          cfg["optics"]["wavelength"])

    print("OAM modes:",
          cfg["oam"]["modes"])

    print("Phase bins:",
          cfg["oam"]["phase_bins"])

    print("Cn2 levels:",
          len(cfg["turbulence"]["Cn2"]))

    print("Mask levels:",
          len(cfg["mask"]["energy_ratio"]))


    # GPU检查
    print("\n[GPU]")

    print("PyTorch:",
          torch.__version__)

    print("CUDA available:",
          torch.cuda.is_available())


    if torch.cuda.is_available():

        print("GPU:",
              torch.cuda.get_device_name(0))

        print("CUDA:",
              torch.version.cuda)


    print("\nEnvironment OK.")