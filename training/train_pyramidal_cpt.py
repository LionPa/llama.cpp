import argparse
import json
import numpy as np
import os
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F

class HubAdapter(nn.Module):
    def __init__(self, dim, num_forks):
        super().__init__()
        self.w_merge = nn.Linear(dim, dim, bias=False)
        self.gate_proj = nn.Linear(dim, 1, bias=False)
        self.sin_scale = nn.Parameter(torch.full((num_forks,), 0.05))
        nn.init.normal_(self.w_merge.weight, std=0.01)
        nn.init.normal_(self.gate_proj.weight, std=0.01)

    def forward(self, forks_tensor):
        k_scale = 1.0 / (forks_tensor.shape[0] ** 0.5)
        weights = torch.abs(self.sin_scale).view(-1, 1, 1)
        x = (forks_tensor * weights).sum(dim=0) * k_scale
        delta = self.w_merge(x)
        gate = F.gelu(self.gate_proj(x))
        return delta * gate

def save_weights(adapters, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    all_scales = []
    for idx, adapter in enumerate(adapters):
        hub_id = idx + 1
        w_path = os.path.join(output_dir, f"w_merge_{hub_id}.bin")
        g_path = os.path.join(output_dir, f"gate_{hub_id}.bin")
        w_fp16 = adapter.w_merge.weight.detach().cpu().to(torch.float16).contiguous().numpy()
        g_fp16 = adapter.gate_proj.weight.detach().cpu().to(torch.float16).contiguous().numpy()
        w_fp16.tofile(w_path)
        g_fp16.tofile(g_path)
        scales = adapter.sin_scale.detach().cpu().to(torch.float32).numpy().tolist()
        all_scales.extend(scales)
        print(f"Saved Hub {hub_id}: {w_path} ({os.path.getsize(w_path)} bytes), {g_path} ({os.path.getsize(g_path)} bytes), scales: {np.round(np.abs(scales), 4)}", flush=True)

    sin_np = np.array(all_scales, dtype=np.float32)
    sin_path = os.path.join(output_dir, "sin_scale.bin")
    sin_np.tofile(sin_path)
    print(f"Saved sin_scale.bin ({len(all_scales)} fork scales, {os.path.getsize(sin_path)} bytes): {np.round(np.abs(sin_np), 4)}", flush=True)

def recv_exact(conn, num_bytes):
    buf = bytearray(num_bytes)
    view = memoryview(buf)
    pos = 0
    try:
        while pos < num_bytes:
            nbytes = conn.recv_into(view[pos:])
            if nbytes == 0:
                return None
            pos += nbytes
        return buf
    except (ConnectionResetError, OSError):
        return None

class TrainerServer(threading.Thread):
    def __init__(self, port, dim, num_hubs, lr, output_dir, device):
        super().__init__(daemon=True)
        self.port = port
        self.dim = dim
        self.num_hubs = num_hubs
        self.lr = lr
        self.output_dir = output_dir
        self.device = device
        self.fork_counts = [2, 3, 3, 3, 3]
        self.adapters = [HubAdapter(dim, self.fork_counts[i]).to(device) for i in range(num_hubs)]
        self.optimizers = [torch.optim.AdamW(a.parameters(), lr=lr, weight_decay=1e-4) for a in self.adapters]
        self.step_counts = [0] * num_hubs
        self.loss_acc = [0.0] * num_hubs
        self.is_running = True
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind(("127.0.0.1", port))
        self.server_sock.listen(1)
        self.server_sock.settimeout(1.0)
        print(f"CPT streaming trainer listening on 127.0.0.1:{port} (Device: {device}, Dim: {dim}, Hubs: {num_hubs})", flush=True)

    def run(self):
        while self.is_running:
            try:
                conn, addr = self.server_sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            print(f"Connected to inference engine at {addr}", flush=True)
            try:
                while self.is_running:
                    hdr_data = recv_exact(conn, 20)
                    if not hdr_data:
                        break

                    magic, hub_idx, n_tokens, n_embd, num_forks = struct.unpack("<5i", hdr_data)
                    if magic != 0x43505433:
                        print(f"Invalid magic received: {hex(magic)}", flush=True)
                        break

                    if hub_idx < 0:
                        break

                    if hub_idx >= self.num_hubs or n_embd != self.dim:
                        print(f"Invalid packet: hub_idx={hub_idx}, n_tokens={n_tokens}, n_embd={n_embd}", flush=True)
                        break

                    x_payload_size = num_forks * n_tokens * n_embd * 4
                    y_payload_size = n_tokens * n_embd * 4
                    x_bytes = recv_exact(conn, x_payload_size)
                    y_bytes = recv_exact(conn, y_payload_size)
                    if not x_bytes or not y_bytes:
                        break

                    x_tensor = torch.frombuffer(x_bytes, dtype=torch.float32).reshape(num_forks, n_tokens, n_embd).to(self.device)
                    y_tensor = torch.frombuffer(y_bytes, dtype=torch.float32).reshape(n_tokens, n_embd).to(self.device)

                    adapter = self.adapters[hub_idx]
                    optimizer = self.optimizers[hub_idx]

                    optimizer.zero_grad()
                    pred = adapter(x_tensor)
                    loss = F.smooth_l1_loss(pred, y_tensor, beta=1.0)
                    loss.backward()
                    optimizer.step()

                    self.step_counts[hub_idx] += 1
                    self.loss_acc[hub_idx] += loss.item()

                    if self.step_counts[hub_idx] == 1 or self.step_counts[hub_idx] % 10 == 0:
                        divisor = 1.0 if self.step_counts[hub_idx] == 1 else 10.0
                        avg_loss = self.loss_acc[hub_idx] / divisor
                        cur_scales = [round(float(s), 4) for s in adapter.sin_scale.detach().cpu().abs()]
                        print(f"Hub {hub_idx + 1} | Step {self.step_counts[hub_idx]:5d} | Tokens: {n_tokens:4d} | Smooth L1 Loss: {avg_loss:.6f} | sin_scales: {cur_scales}", flush=True)
                        self.loss_acc[hub_idx] = 0.0
            finally:
                conn.close()

    def stop(self):
        self.is_running = False
        try:
            self.server_sock.close()
        except OSError:
            pass

    def export(self):
        print(f"Saving final adapter weights. Total steps per hub: {self.step_counts}", flush=True)
        save_weights(self.adapters, self.output_dir)

def split_dataset(file_path, max_chars=32000):
    if not os.path.exists(file_path):
        print(f"Dataset file not found: {file_path}", flush=True)
        return []

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    raw_chunks = []
    if "<|turn>user" in text:
        parts = text.split("<|turn>user")
        for p in parts:
            item = p.strip()
            if len(item) > 0:
                raw_chunks.append(item)
    elif "\n---\n" in text:
        parts = text.split("\n---\n")
        for p in parts:
            item = p.strip()
            if len(item) > 0:
                raw_chunks.append(item)
    else:
        paras = text.split("\n\n")
        curr = ""
        for pa in paras:
            if len(curr) + len(pa) > max_chars:
                if len(curr) > 0:
                    raw_chunks.append(curr.strip())
                curr = pa
            else:
                curr += "\n\n" + pa
        if len(curr) > 0:
            raw_chunks.append(curr.strip())

    final_prompts = []
    for chunk in raw_chunks:
        if len(chunk) <= max_chars:
            final_prompts.append(chunk)
        else:
            for start in range(0, len(chunk), max_chars):
                sub = chunk[start:start + max_chars].strip()
                if len(sub) > 0:
                    final_prompts.append(sub)

    return final_prompts

def wait_server_ready(url, timeout=90):
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(f"{url}/health", method="GET")
            with urllib.request.urlopen(req, timeout=2) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        time.sleep(1.0)
    return False

def main():
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description="CPT Pyramidal Hub Trainer")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--http-port", type=int, default=8085)
    parser.add_argument("--dim", type=int, default=3840)
    parser.add_argument("--num-hubs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=str, default="./weights/cpt")
    parser.add_argument("--dataset", type=str, default="./data/rich_train_dataset.txt")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--server-bin", type=str, default="./build/bin/Release/llama-server")
    parser.add_argument("--ngl", type=int, default=99)
    parser.add_argument("--ctx-size", type=int, default=8192)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-launch", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu") if args.cpu or not torch.cuda.is_available() else torch.device("cuda")
    trainer = TrainerServer(args.port, args.dim, args.num_hubs, args.lr, args.output_dir, device)
    trainer.start()

    proc = None
    if not args.no_launch:
        if not os.path.exists(args.server_bin):
            print(f"Error: Server binary not found at {args.server_bin}", flush=True)
            trainer.stop()
            sys.exit(1)

        cmd = [
            args.server_bin,
            "-m", args.model,
            "-ngl", str(args.ngl),
            "-c", str(args.ctx_size),
            "--cascade-train-port", str(args.port),
            "--port", str(args.http_port)
        ]
        print(f"Launching inference engine: {' '.join(cmd)}", flush=True)
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

        server_url = f"http://127.0.0.1:{args.http_port}"
        print(f"Waiting for engine to load model at {server_url}...", flush=True)
        if not wait_server_ready(server_url):
            print("Engine failed to become ready within timeout.", flush=True)
            if proc:
                proc.terminate()
            trainer.stop()
            sys.exit(1)
        print("Engine is ready and listening. Starting dataset streaming...", flush=True)

    prompts = split_dataset(args.dataset)
    print(f"Loaded {len(prompts)} prompt chunks from {args.dataset}", flush=True)

    if args.max_samples > 0:
        prompts = prompts[:args.max_samples]
        print(f"Limiting training to first {len(prompts)} samples.", flush=True)

    server_url = f"http://127.0.0.1:{args.http_port}"
    total = len(prompts)
    try:
        for idx, p in enumerate(prompts):
            print(f"Processing sample [{idx + 1}/{total}] ({len(p)} chars)...", flush=True)
            body = json.dumps({"prompt": p, "n_predict": 1, "temperature": 0.0}).encode("utf-8")
            req = urllib.request.Request(
                f"{server_url}/completion",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    resp.read()
            except Exception as e:
                print(f"Sample [{idx + 1}/{total}] warning: {e}", flush=True)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Finalizing...", flush=True)
    finally:
        if proc:
            print("Terminating engine process...", flush=True)
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        time.sleep(1.0)
        trainer.stop()
        trainer.export()
        print("All done. Weights exported successfully.", flush=True)

if __name__ == "__main__":
    main()
