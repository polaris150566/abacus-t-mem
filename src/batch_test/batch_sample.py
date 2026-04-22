#!/usr/bin/env python3
import argparse
import logging
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from multiprocessing import Process
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PLM_PARAM_DIR = "/database/lyf_database/pretrain_lm/esm/param"
SELFCOND_PLM = "650M"
MASK_MODE = "aatype_nll"
MAX_LIG_NUM = 10
AUGMENT_EPS = 0.2


def read_lines_as_list(txt_path):
    with open(txt_path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def normalize_data_names(raw_names):
    data_names = []
    for item in raw_names:
        name = Path(item.strip()).name
        if not name:
            continue
        if name.endswith(".npy"):
            name = Path(name).stem
        data_names.append(name)

    deduped = []
    seen = set()
    for name in data_names:
        if name in seen:
            continue
        seen.add(name)
        deduped.append(name)
    return deduped


def parse_device_list(raw_device_list):
    devices = [int(item.strip()) for item in raw_device_list.split(",") if item.strip()]
    if not devices:
        raise ValueError("device_list must contain at least one GPU id")
    return devices


def build_output_root(design_seq_dir, temperature, iter_num, tm_raw, checkpoint):
    ckpt_name = Path(checkpoint).stem
    return Path(design_seq_dir) / f"T_{temperature}_R_{iter_num}_esm_refined_{tm_raw}" / ckpt_name


def log_cmd_to_file(cmd, log_path, extra_env=None):
    header = "# " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    env_lines = []
    if extra_env:
        env_lines = [f"export {k}={shlex.quote(str(v))}" for k, v in extra_env.items()]
    cmd_str = " ".join(shlex.quote(part) for part in cmd)
    block = "\n".join([header] + env_lines + [cmd_str])
    logger.info("[cmd]\n%s", block)
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(block + "\n\n")

def read_data_names(txt_path):
    data_names = normalize_data_names(read_lines_as_list(txt_path))
    if not data_names:
        raise SystemExit(f"No targets found in {txt_path}")
    return data_names


def run_design_on_device(*, data_items, order_suffix, device, common_args, run_spec):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device)
    env["PYTHONPATH"] = "/home/chenty/public_data/tmdet_data/data_utils:/home/chenty/public_utils:/home/chenty/abacust_mem/src:" + env.get("PYTHONPATH", "")

    cfg_w = run_spec.get("cfg_guidance_scale", 1.0)
    w_tag = "" if cfg_w == 1.0 else f"_w{cfg_w}"
    mbg_w = run_spec.get("mem_bg_weight", 0.0)
    mbg_tag = "" if mbg_w == 0.0 else f"_mbg{mbg_w}"
    suffix = "esm_refined_{}{}{}/{}".format(run_spec["tm_raw"], w_tag, mbg_tag, Path(run_spec["checkpoint"]).stem)
    cmd_template = f"""
        python batch_seq_design_with_lig_raw_pdb.py
        --data_list {shlex.quote(":".join(data_items))}
        --order_suffix {shlex.quote(str(order_suffix))}
        --iter_num {shlex.quote(str(common_args.iter_num))}
        --temperature {shlex.quote(str(common_args.temperature))}
        --suffix {shlex.quote(suffix)}
        --npy_dir {shlex.quote(str(Path(common_args.npy_dir) / "all_npy"))}
        --checkpoint {shlex.quote(run_spec["checkpoint"])}
        --save_dir {shlex.quote(common_args.design_seq_dir)}
        --root_dir {shlex.quote(common_args.input_path)}
        --batchsize {shlex.quote(str(common_args.batchsize))}
        --designed_positions ""
        --PLM_selfcond 1
        --PLM_param_dir {shlex.quote(PLM_PARAM_DIR)}
        --selfcondPLM {shlex.quote(SELFCOND_PLM)}
        --augment_eps {shlex.quote(str(AUGMENT_EPS))}
        --mask_mode {shlex.quote(MASK_MODE)}
        --tm_raw {shlex.quote(run_spec["tm_raw"])}
        --device cuda:0
        --max_lig_num {shlex.quote(str(MAX_LIG_NUM))}
        --consider_lig
        --embed_unimol_reprs
        --nar
        --esm_refinement
        --write_allatom_model
        {"--mem_config " + shlex.quote(run_spec["mem_config"]) if run_spec.get("mem_config", "") else ""}
        {"--cfg_guidance_scale " + str(run_spec["cfg_guidance_scale"]) if run_spec.get("cfg_guidance_scale", 1.0) != 1.0 else ""}
        {"--mem_bg_weight " + str(run_spec["mem_bg_weight"]) if run_spec.get("mem_bg_weight", 0.0) != 0.0 else ""}
        """.strip()
    cmd = shlex.split(cmd_template)

    log_path = build_output_root(
        common_args.design_seq_dir,
        common_args.temperature,
        common_args.iter_num,
        run_spec["tm_raw"],
        run_spec["checkpoint"],
    ) / f"run_cmd_device{order_suffix}.log"
    log_cmd_to_file(cmd, log_path, {"CUDA_VISIBLE_DEVICES": str(device)})
    subprocess.run(cmd, env=env, cwd=str(SCRIPT_DIR), check=True)


def run_one_spec(common_args, data_names, run_spec):
    Path(common_args.design_seq_dir).mkdir(parents=True, exist_ok=True)
    device_list = parse_device_list(run_spec["device_list"])
    data_list = [f"{data_name}.npy" for data_name in data_names]
    data_chunks = [data_list[index:: len(device_list)] for index in range(len(device_list))]

    logger.info("=" * 60)
    logger.info("tm_raw:        %s", run_spec["tm_raw"])
    logger.info("checkpoint:    %s", run_spec["checkpoint"])
    logger.info("device_list:   %s", run_spec["device_list"])
    logger.info("batchsize:     %s", common_args.batchsize)
    logger.info("temperature:   %s", common_args.temperature)
    logger.info("iter_num:      %s", common_args.iter_num)
    logger.info("num_targets:   %s", len(data_names))
    logger.info("input_path:    %s", common_args.input_path)
    logger.info("npy_dir:       %s", common_args.npy_dir)
    logger.info(
        "output_root:   %s",
        build_output_root(
            common_args.design_seq_dir,
            common_args.temperature,
            common_args.iter_num,
            run_spec["tm_raw"],
            run_spec["checkpoint"],
        ),
    )
    logger.info("=" * 60)

    processes = []
    for order_suffix, data_items in enumerate(data_chunks):
        if not data_items:
            continue
        process = Process(
            target=run_design_on_device,
            kwargs={
                "data_items": data_items,
                "order_suffix": order_suffix,
                "device": device_list[order_suffix],
                "common_args": common_args,
                "run_spec": run_spec,
            },
        )
        processes.append(process)
        process.start()

    failed = []
    for process in processes:
        process.join()
        if process.exitcode != 0:
            failed.append(process.exitcode)

    if failed:
        raise SystemExit(f"sample failed with child exit codes: {failed}")


def build_run_specs(args):
    if len(args.checkpoint) != len(args.tm_raw) or len(args.checkpoint) != len(args.device_list):
        raise SystemExit("checkpoint, tm_raw and device_list must have the same number of entries")

    mem_configs = args.mem_config if args.mem_config else [""] * len(args.checkpoint)
    cfg_scales = args.cfg_guidance_scale if args.cfg_guidance_scale else [1.0] * len(args.checkpoint)
    mem_bg_weights = args.mem_bg_weight if args.mem_bg_weight else [0.0] * len(args.checkpoint)
    if len(mem_configs) == 1:
        mem_configs = mem_configs * len(args.checkpoint)
    if len(cfg_scales) == 1:
        cfg_scales = cfg_scales * len(args.checkpoint)
    if len(mem_bg_weights) == 1:
        mem_bg_weights = mem_bg_weights * len(args.checkpoint)

    specs = []
    for idx in range(len(args.checkpoint)):
        specs.append(
            {
                "checkpoint": args.checkpoint[idx],
                "tm_raw": args.tm_raw[idx],
                "device_list": args.device_list[idx],
                "mem_config": mem_configs[idx],
                "cfg_guidance_scale": cfg_scales[idx],
                "mem_bg_weight": mem_bg_weights[idx],
                "name": f"{Path(args.checkpoint[idx]).stem}_{args.tm_raw[idx]}",
            }
        )
    return specs


def parse_args():
    parser = argparse.ArgumentParser(description="Run abacust batch sample.")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--design-seq-dir", required=True)
    parser.add_argument("--npy-dir", required=True)
    parser.add_argument("--data-names-file", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--tm-raw", action="append", required=True)
    parser.add_argument("--device-list", action="append", required=True)
    parser.add_argument("--batchsize", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--iter-num", type=int, default=20)
    parser.add_argument("--mem-config", action="append", default=None,
                        help="Path to mem_config YAML (one per checkpoint, or one for all)")
    parser.add_argument("--cfg-guidance-scale", action="append", type=float, default=None,
                        help="CFG guidance scale (one per checkpoint, or one for all)")
    parser.add_argument("--mem-bg-weight", action="append", type=float, default=None,
                        help="Membrane depth background weight (one per checkpoint, or one for all)")
    return parser.parse_args()


def main():
    args = parse_args()
    data_names = read_data_names(args.data_names_file)
    run_specs = build_run_specs(args)

    failed = []
    with ThreadPoolExecutor(max_workers=len(run_specs)) as executor:
        future_map = {
            executor.submit(run_one_spec, args, data_names, run_spec): run_spec["name"]
            for run_spec in run_specs
        }
        for future in as_completed(future_map):
            name = future_map[future]
            try:
                future.result()
                logger.info("[%s] finished", name)
            except Exception as exc:
                logger.error("[%s] failed: %s", name, exc)
                failed.append(name)

    if failed:
        raise SystemExit("failed runs: " + ", ".join(failed))


if __name__ == "__main__":
    main()
