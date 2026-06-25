import torch
import csv
import os
import numpy as np

class GeometricLogger:
    def __init__(self, filename="geometry_analysis.csv"):
        self.filename = filename
        # 自动创建目录（如果不存在）
        os.makedirs(os.path.dirname(os.path.abspath(filename)) or ".", exist_ok=True)

        if not os.path.exists(self.filename):
            with open(self.filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # 记录：方法名，层数，CaseID，X(信号强度), Y(噪音强度), 总模长
                writer.writerow(["Method", "Layer", "Case_ID", "X_Signal", "Y_Noise", "Norm_Delta"])



    def log_update(self, method_name, layer, case_id, delta_W, u_manifold):

        if not isinstance(delta_W, torch.Tensor):
            delta_W = torch.tensor(delta_W)
        if not isinstance(u_manifold, torch.Tensor):
            u_manifold = torch.tensor(u_manifold)

        with torch.no_grad():

            W = delta_W.detach().float()
            u = u_manifold.detach().float()

            if W.device != u.device:
                u = u.to(W.device)

            if u.dim() == 1:
                u = u.view(-1, 1)

            # W shape: [Out, In]
            # u shape: [D, 1]

            out_dim, in_dim = W.shape
            u_dim = u.shape[0]

            signal_vec = None

            # ( up_proj, gate_proj, attention)
            # [Out, In] @ [In, 1] -> [Out, 1]
            if in_dim == u_dim:

                if u.shape[1] == 1:
                    u = u / (torch.norm(u) + 1e-8)
                signal_vec = torch.matmul(W, u)

            # [Out, In].T @ [Out, 1] -> [In, 1]
            elif out_dim == u_dim:

                if u.shape[1] == 1:
                    u = u / (torch.norm(u) + 1e-8)

                signal_vec = torch.matmul(W.t(), u)

            else:
                print(f"[GeoLog Error] Dim mismatch irrecoverable: W{W.shape} vs u{u.shape}. Skipping.")
                return

            x_signal = torch.norm(signal_vec).item()
            total_norm = torch.norm(W).item()

            y_noise = np.sqrt(max(0, total_norm ** 2 - x_signal ** 2))

        try:
            with open(self.filename, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([method_name, layer, case_id, x_signal, y_noise, total_norm])

            print(f"[GeoLog] Saved: {method_name} L{layer} | Signal={x_signal:.4f} | Noise={y_noise:.4f}")
        except Exception as e:
            print(f"[GeoLog Error] Write failed: {e}")


def format_tensor(val):
    if isinstance(val, torch.Tensor):
        if val.numel() == 1:
            return f"{val.item():.4e}"
        return f"Tensor shape={list(val.shape)} norm={val.norm().item():.4e}"
    if isinstance(val, (float, np.float32, np.float64)):
        return f"{val:.4e}"
    return str(val)

def log_edit_stats(case_id, method, step, **kwargs):
    print(f"\n[Logger] Case {case_id} | {method} | Step: {step}")
    print("-" * 50)
    for k, v in kwargs.items():
        if "eigen" in k.lower() and isinstance(v, (torch.Tensor, np.ndarray)):
            vals = v.flatten()

            max_eig = vals[-1].item() if isinstance(v, torch.Tensor) else vals[-1]
            min_eig = vals[0].item() if isinstance(v, torch.Tensor) else vals[0]
            print(f"  > {k} Range: [{min_eig:.2e}, {max_eig:.2e}]")

            head = vals[:10].tolist()
            tail = vals[-10:].tolist()
            print(f"  > {k} Head: {[f'{x:.2e}' for x in head]}")
            print(f"  > {k} Tail: {[f'{x:.2e}' for x in tail]}")
        else:
            print(f"  > {k}: {format_tensor(v)}")
    print("-" * 50 + "\n")



# 全局实例
geo_logger = GeometricLogger("experiment_data_for_plot.csv")


