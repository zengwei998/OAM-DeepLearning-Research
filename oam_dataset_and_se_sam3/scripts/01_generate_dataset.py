import os
import csv
import yaml
import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)

GLOBAL_CACHE = {}


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def build_seed(base_seed, l, cn2, distance, snr, idx):
    cn2_code = int(round(float(cn2) * 1e15 * 10))
    return base_seed + l * 100000000 + cn2_code * 1000000 + distance * 1000 + (snr + 100) * 100 + idx


def build_filename(l, cn2, distance, snr, idx):
    cn2_text = f"{cn2:.1e}".replace("+", "")
    return f"l{l:02d}_cn2_{cn2_text}_z{distance:04d}_snr{snr:+03d}_idx{idx:03d}.png"


def init_worker(cfg_core):
    global GLOBAL_CACHE

    grid_size = cfg_core["grid_size"]
    aperture_size = cfg_core["aperture_size"]
    wavelength = cfg_core["wavelength"]
    beam_waist = cfg_core["beam_waist"]
    phase_screen_num = cfg_core["phase_screen_num"]

    x = np.linspace(-aperture_size / 2, aperture_size / 2, grid_size, endpoint=False)
    y = np.linspace(-aperture_size / 2, aperture_size / 2, grid_size, endpoint=False)
    dx = aperture_size / grid_size

    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx ** 2 + yy ** 2)
    theta = np.arctan2(yy, xx)

    fx = np.fft.fftfreq(grid_size, d=dx)
    fy = np.fft.fftfreq(grid_size, d=dx)
    fxx, fyy = np.meshgrid(fx, fy)

    k = 2.0 * np.pi / wavelength

    lg_fields = {}
    for l in cfg_core["topological_charges"]:
        abs_l = abs(l)
        amplitude = (np.sqrt(2.0) * r / beam_waist) ** abs_l
        amplitude *= np.exp(-(r ** 2) / (beam_waist ** 2))
        field = amplitude * np.exp(1j * l * theta)

        norm = np.sqrt(np.sum(np.abs(field) ** 2))
        if norm > 0:
            field = field / norm

        lg_fields[int(l)] = field.astype(np.complex128)

    transfer_functions = {}
    for distance in cfg_core["propagation_distances"]:
        dz = float(distance) / phase_screen_num
        root = 1.0 - (wavelength * fxx) ** 2 - (wavelength * fyy) ** 2
        root = np.where(root >= 0, root, 0.0)
        h = np.exp(1j * k * dz * np.sqrt(root))
        transfer_functions[int(distance)] = h.astype(np.complex128)

    df = 1.0 / aperture_size
    kappa = 2.0 * np.pi * np.sqrt(fxx ** 2 + fyy ** 2)
    kappa[0, 0] = np.inf

    phase_amp_cache = {}

    for cn2 in cfg_core["cn2_values"]:
        for distance in cfg_core["propagation_distances"]:
            dz = float(distance) / phase_screen_num
            phi_n = 0.033 * float(cn2) * (kappa ** (-11.0 / 3.0))
            phi_phi = 2.0 * np.pi * (k ** 2) * dz * phi_n
            phase_amp = np.sqrt(phi_phi) * df
            phase_amp[0, 0] = 0.0
            phase_amp *= cfg_core["turbulence_strength_scale"]
            phase_amp_cache[(float(cn2), int(distance))] = phase_amp.astype(np.float64)

    GLOBAL_CACHE = {
        "cfg_core": cfg_core,
        "lg_fields": lg_fields,
        "transfer_functions": transfer_functions,
        "phase_amp_cache": phase_amp_cache,
        "xx": xx,
        "yy": yy,
        "r": r,
        "fxx": fxx,
        "fyy": fyy,
    }


def angular_spectrum_propagate(field, h):
    return np.fft.ifft2(np.fft.fft2(field) * h).astype(np.complex128)


def kolmogorov_phase_screen(phase_amp, rng, grid_size):
    white_noise = rng.normal(size=(grid_size, grid_size)) + 1j * rng.normal(size=(grid_size, grid_size))
    phase_spectrum = white_noise * phase_amp
    phase = np.real(np.fft.ifft2(np.fft.ifftshift(phase_spectrum))) * (grid_size ** 2)
    phase -= np.mean(phase)
    return phase.astype(np.float64)


def apply_frequency_tilt(field, rng, strength):
    cfg_core = GLOBAL_CACHE["cfg_core"]
    xx = GLOBAL_CACHE["xx"]
    yy = GLOBAL_CACHE["yy"]

    ax = rng.uniform(-strength, strength)
    ay = rng.uniform(-strength, strength)

    phase = ax * xx + ay * yy
    return field * np.exp(1j * phase), ax, ay


def normalize_intensity(intensity):
    intensity = np.asarray(intensity, dtype=np.float64)
    intensity -= np.min(intensity)
    max_value = np.max(intensity)
    if max_value > 0:
        intensity /= max_value
    return intensity.astype(np.float32)


def apply_receiver_shift(image, max_shift_pixels, rng):
    shift_x = int(rng.integers(-max_shift_pixels, max_shift_pixels + 1))
    shift_y = int(rng.integers(-max_shift_pixels, max_shift_pixels + 1))

    shifted = np.roll(image, shift=(shift_y, shift_x), axis=(0, 1))

    if shift_y > 0:
        shifted[:shift_y, :] = 0
    elif shift_y < 0:
        shifted[shift_y:, :] = 0

    if shift_x > 0:
        shifted[:, :shift_x] = 0
    elif shift_x < 0:
        shifted[:, shift_x:] = 0

    return shifted.astype(np.float32), shift_x, shift_y


def apply_aperture_clipping(image, rng, ratio_min, ratio_max):
    h, w = image.shape
    cy, cx = h / 2, w / 2
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    ratio = float(rng.uniform(ratio_min, ratio_max))
    radius = min(h, w) * 0.5 * ratio

    mask = (rr <= radius).astype(np.float32)
    clipped = image * mask

    return clipped.astype(np.float32), ratio


def apply_nonuniform_background(image, rng, strength):
    h, w = image.shape
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    xx, yy = np.meshgrid(x, y)

    a = rng.uniform(-strength, strength)
    b = rng.uniform(-strength, strength)
    c = rng.uniform(0, strength)

    background = a * xx + b * yy + c
    image = image + background
    image = np.clip(image, 0, 1)

    return image.astype(np.float32)


def apply_speckle_noise(image, rng, strength):
    speckle = rng.normal(0.0, strength, size=image.shape)
    noisy = image + image * speckle
    noisy = np.clip(noisy, 0, 1)
    return noisy.astype(np.float32)


def apply_poisson_noise(image, rng, peak):
    scaled = np.clip(image, 0, 1) * peak
    noisy = rng.poisson(scaled) / float(peak)
    noisy = np.clip(noisy, 0, 1)
    return noisy.astype(np.float32)


def add_awgn(image, snr_db, rng):
    image = image.astype(np.float64)
    signal_power = np.mean(image ** 2)
    snr_linear = 10.0 ** (float(snr_db) / 10.0)
    noise_power = signal_power / snr_linear
    noise_std = np.sqrt(noise_power)
    noise = rng.normal(0.0, noise_std, size=image.shape)
    noisy = np.clip(image + noise, 0.0, 1.0)
    return noisy.astype(np.float32)


def apply_blur(image, rng, probability):
    if rng.random() > probability:
        return image.astype(np.float32), False

    radius = float(rng.uniform(0.3, 1.1))
    image_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    pil = Image.fromarray(image_uint8, mode="L")
    pil = pil.filter(ImageFilter.GaussianBlur(radius=radius))
    out = np.asarray(pil).astype(np.float32) / 255.0
    return out.astype(np.float32), True


def apply_intensity_gain(image, rng, gain_min, gain_max):
    gain = float(rng.uniform(gain_min, gain_max))
    image = np.clip(image * gain, 0, 1)
    return image.astype(np.float32), gain


def apply_stray_light(image, rng, max_value):
    stray = float(rng.uniform(0, max_value))
    image = np.clip(image + stray, 0, 1)
    return image.astype(np.float32), stray


def save_gray_png(image, path):
    image_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(image_uint8, mode="L").save(path)


def generate_one_sample(task):
    global GLOBAL_CACHE

    l, cn2, distance, snr, idx = task
    cfg = GLOBAL_CACHE["cfg_core"]

    seed = build_seed(cfg["seed"], l, cn2, distance, snr, idx)
    rng = np.random.default_rng(seed)

    grid_size = cfg["grid_size"]
    raw_image_dir = Path(cfg["raw_image_dir"])

    field = GLOBAL_CACHE["lg_fields"][int(l)].copy()
    h = GLOBAL_CACHE["transfer_functions"][int(distance)]
    phase_amp = GLOBAL_CACHE["phase_amp_cache"][(float(cn2), int(distance))]

    tilt_ax = 0.0
    tilt_ay = 0.0

    if cfg["enable_frequency_tilt"]:
        field, tilt_ax, tilt_ay = apply_frequency_tilt(
            field, rng, cfg["frequency_tilt_strength"]
        )

    for _ in range(cfg["phase_screen_num"]):
        phase = kolmogorov_phase_screen(phase_amp, rng, grid_size)
        field *= np.exp(1j * phase)
        field = angular_spectrum_propagate(field, h)

    intensity = np.abs(field) ** 2
    intensity = normalize_intensity(intensity)

    clip_ratio = 1.0
    shift_x = 0
    shift_y = 0
    gain = 1.0
    stray_light = 0.0
    blurred = False

    if cfg["enable_aperture_clipping"]:
        intensity, clip_ratio = apply_aperture_clipping(
            intensity,
            rng,
            cfg["aperture_clip_ratio_min"],
            cfg["aperture_clip_ratio_max"],
        )

    if cfg["enable_receiver_shift"]:
        intensity, shift_x, shift_y = apply_receiver_shift(
            intensity,
            cfg["max_shift_pixels"],
            rng,
        )

    if cfg["enable_random_intensity_gain"]:
        intensity, gain = apply_intensity_gain(
            intensity,
            rng,
            cfg["gain_min"],
            cfg["gain_max"],
        )

    if cfg["enable_background_stray_light"]:
        intensity, stray_light = apply_stray_light(
            intensity,
            rng,
            cfg["stray_light_max"],
        )

    if cfg["enable_nonuniform_background"]:
        intensity = apply_nonuniform_background(
            intensity,
            rng,
            cfg["background_strength"],
        )

    if cfg["enable_speckle_noise"]:
        intensity = apply_speckle_noise(
            intensity,
            rng,
            cfg["speckle_strength"],
        )

    if cfg["enable_poisson_noise"]:
        intensity = apply_poisson_noise(
            intensity,
            rng,
            cfg["poisson_peak"],
        )

    intensity = add_awgn(intensity, snr, rng)

    if cfg["enable_blur"]:
        intensity, blurred = apply_blur(
            intensity,
            rng,
            cfg["blur_probability"],
        )

    class_dir = raw_image_dir / f"class_{l:02d}"
    class_dir.mkdir(parents=True, exist_ok=True)

    filename = build_filename(l, cn2, distance, snr, idx)
    save_path = class_dir / filename
    save_gray_png(intensity, save_path)

    return {
        "image_path": str(save_path).replace("\\", "/"),
        "label": int(l),
        "topological_charge_abs": int(l),
        "radial_index": 0,
        "cn2": f"{float(cn2):.6e}",
        "propagation_distance_m": int(distance),
        "snr_db": int(snr),
        "sample_index": int(idx),
        "seed": int(seed),
        "wavelength_m": f"{cfg['wavelength']:.6e}",
        "beam_waist_m": f"{cfg['beam_waist']:.6e}",
        "aperture_size_m": f"{cfg['aperture_size']:.6e}",
        "grid_size": int(cfg["grid_size"]),
        "phase_screen_num": int(cfg["phase_screen_num"]),
        "turbulence_strength_scale": float(cfg["turbulence_strength_scale"]),
        "shift_x": int(shift_x),
        "shift_y": int(shift_y),
        "intensity_gain": f"{gain:.6f}",
        "stray_light": f"{stray_light:.6f}",
        "clip_ratio": f"{clip_ratio:.6f}",
        "frequency_tilt_ax": f"{tilt_ax:.6f}",
        "frequency_tilt_ay": f"{tilt_ay:.6f}",
        "blurred": bool(blurred),
        "propagation_method": "angular_spectrum",
        "turbulence_spectrum": "kolmogorov",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/simulation.yaml")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    topological_charges = [int(v) for v in cfg["oam"]["topological_charges"]]
    cn2_values = [float(v) for v in cfg["channel"]["cn2_values"]]
    distances = [int(v) for v in cfg["channel"]["propagation_distances"]]
    snrs = [int(v) for v in cfg["channel"]["snr_values"]]
    samples_per_condition = int(cfg["data"]["samples_per_condition"])

    raw_image_dir = Path(cfg["output"]["raw_image_dir"])
    metadata_dir = Path(cfg["output"]["metadata_dir"])
    manifest_dir = Path(cfg["output"]["manifest_dir"])

    mkdir(raw_image_dir)
    mkdir(metadata_dir)
    mkdir(manifest_dir)

    num_workers = args.workers
    if num_workers is None:
        num_workers = int(cfg.get("performance", {}).get("num_workers", 16))

    cfg_core = {
        "seed": int(cfg["seed"]),
        "grid_size": int(cfg["physics"]["grid_size"]),
        "aperture_size": float(cfg["physics"]["aperture_size"]),
        "wavelength": float(cfg["physics"]["wavelength"]),
        "beam_waist": float(cfg["physics"]["beam_waist"]),
        "phase_screen_num": int(cfg["physics"]["phase_screen_num"]),
        "turbulence_strength_scale": float(cfg["physics"]["turbulence_strength_scale"]),
        "raw_image_dir": str(raw_image_dir),
        "topological_charges": topological_charges,
        "cn2_values": cn2_values,
        "propagation_distances": distances,
        "enable_receiver_shift": bool(cfg["complexity"]["enable_receiver_shift"]),
        "max_shift_pixels": int(cfg["complexity"]["max_shift_pixels"]),
        "enable_random_intensity_gain": bool(cfg["complexity"]["enable_random_intensity_gain"]),
        "gain_min": float(cfg["complexity"]["gain_min"]),
        "gain_max": float(cfg["complexity"]["gain_max"]),
        "enable_background_stray_light": bool(cfg["complexity"]["enable_background_stray_light"]),
        "stray_light_max": float(cfg["complexity"]["stray_light_max"]),
        "enable_speckle_noise": bool(cfg["complexity"]["enable_speckle_noise"]),
        "speckle_strength": float(cfg["complexity"]["speckle_strength"]),
        "enable_poisson_noise": bool(cfg["complexity"]["enable_poisson_noise"]),
        "poisson_peak": float(cfg["complexity"]["poisson_peak"]),
        "enable_nonuniform_background": bool(cfg["complexity"]["enable_nonuniform_background"]),
        "background_strength": float(cfg["complexity"]["background_strength"]),
        "enable_frequency_tilt": bool(cfg["complexity"]["enable_frequency_tilt"]),
        "frequency_tilt_strength": float(cfg["complexity"]["frequency_tilt_strength"]),
        "enable_aperture_clipping": bool(cfg["complexity"]["enable_aperture_clipping"]),
        "aperture_clip_ratio_min": float(cfg["complexity"]["aperture_clip_ratio_min"]),
        "aperture_clip_ratio_max": float(cfg["complexity"]["aperture_clip_ratio_max"]),
        "enable_blur": bool(cfg["complexity"]["enable_blur"]),
        "blur_probability": float(cfg["complexity"]["blur_probability"]),
    }

    tasks = [
        (l, cn2, distance, snr, idx)
        for l in topological_charges
        for cn2 in cn2_values
        for distance in distances
        for snr in snrs
        for idx in range(samples_per_condition)
    ]

    print("=" * 80)
    print("OAM-SE-SAM3-PPI 强退化复杂数据集生成")
    print("=" * 80)
    print(f"总样本数: {len(tasks)}")
    print(f"相位屏数量: {cfg_core['phase_screen_num']}")
    print(f"湍流增强倍数: {cfg_core['turbulence_strength_scale']}")
    print(f"最大接收端偏移: {cfg_core['max_shift_pixels']} px")
    print(f"散斑噪声: {cfg_core['enable_speckle_noise']}")
    print(f"泊松噪声: {cfg_core['enable_poisson_noise']}")
    print(f"非均匀背景: {cfg_core['enable_nonuniform_background']}")
    print(f"频域倾斜: {cfg_core['enable_frequency_tilt']}")
    print(f"接收端孔径裁剪: {cfg_core['enable_aperture_clipping']}")
    print(f"模糊扰动: {cfg_core['enable_blur']}")
    print(f"进程数: {num_workers}")
    print("=" * 80)

    rows = []

    with mp.Pool(
        processes=num_workers,
        initializer=init_worker,
        initargs=(cfg_core,),
    ) as pool:
        for row in tqdm(
            pool.imap_unordered(generate_one_sample, tasks, chunksize=8),
            total=len(tasks),
            desc="Generating",
        ):
            rows.append(row)

    rows.sort(
        key=lambda x: (
            int(x["label"]),
            float(x["cn2"]),
            int(x["propagation_distance_m"]),
            int(x["snr_db"]),
            int(x["sample_index"]),
        )
    )

    all_csv = manifest_dir / "all.csv"
    fieldnames = list(rows[0].keys())

    with open(all_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_file = metadata_dir / "dataset_summary.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("OAM-SE-SAM3-PPI 强退化复杂数据集生成摘要\n")
        f.write("=" * 80 + "\n")
        f.write(f"总样本数: {len(rows)}\n")
        f.write(f"拓扑荷阶数: {topological_charges}\n")
        f.write(f"Cn2: {cn2_values}\n")
        f.write(f"传播距离: {distances}\n")
        f.write(f"SNR: {snrs}\n")
        f.write(f"每组样本数: {samples_per_condition}\n")
        f.write(f"相位屏数量: {cfg_core['phase_screen_num']}\n")
        f.write(f"湍流增强倍数: {cfg_core['turbulence_strength_scale']}\n")
        f.write(f"最大偏移: {cfg_core['max_shift_pixels']} px\n")
        f.write(f"散斑噪声强度: {cfg_core['speckle_strength']}\n")
        f.write(f"泊松噪声峰值: {cfg_core['poisson_peak']}\n")
        f.write(f"非均匀背景强度: {cfg_core['background_strength']}\n")
        f.write(f"频域倾斜强度: {cfg_core['frequency_tilt_strength']}\n")
        f.write(f"孔径裁剪比例: {cfg_core['aperture_clip_ratio_min']} - {cfg_core['aperture_clip_ratio_max']}\n")
        f.write(f"模糊概率: {cfg_core['blur_probability']}\n")

    print("=" * 80)
    print("数据集生成完成")
    print(f"all.csv: {all_csv}")
    print(f"summary: {summary_file}")
    print("=" * 80)


if __name__ == "__main__":
    mp.freeze_support()
    main()