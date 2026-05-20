import torch
torch.manual_seed(42)

B, T, H, N = 1, 2, 12, 64
device = 'mps'
C = H * N

# ── Forward ──
def diff_wkv7(rr, ww, kk, vv, aa, bb, state):
    ww_decay = torch.exp(-torch.exp(ww))
    outs = []
    for t in range(T):
        ww_t = ww_decay[:, t].unsqueeze(-2)
        kk_t = kk[:, t].unsqueeze(-2)
        vv_t = vv[:, t].unsqueeze(-1)
        aa_t = aa[:, t].unsqueeze(-1)
        bb_t = bb[:, t].unsqueeze(-2)
        rr_t = rr[:, t].unsqueeze(-1)
        state = state * ww_t + (state @ aa_t) @ bb_t + vv_t @ kk_t
        outs.append((state @ rr_t).squeeze(-1))
    return torch.stack(outs, dim=1).view(B, T, C), state

rr = torch.randn(B, T, H, N, device=device, dtype=torch.float32, requires_grad=True)
ww = torch.randn(B, T, H, N, device=device, dtype=torch.float32, requires_grad=True)
kk = torch.randn(B, T, H, N, device=device, dtype=torch.float32, requires_grad=True)
vv = torch.randn(B, T, H, N, device=device, dtype=torch.float32, requires_grad=True)
aa = torch.randn(B, T, H, N, device=device, dtype=torch.float32, requires_grad=True)
bb = torch.randn(B, T, H, N, device=device, dtype=torch.float32, requires_grad=True)
state = torch.randn(B, H, N, N, device=device, dtype=torch.float32, requires_grad=True)

out, s_out = diff_wkv7(rr, ww, kk, vv, aa, bb, state)
loss = out.pow(2).sum() + s_out.pow(2).sum()
loss.backward()

auto_r = rr.grad.clone()
auto_w = ww.grad.clone()
auto_k = kk.grad.clone()
auto_v = vv.grad.clone()
auto_a = aa.grad.clone()
auto_b = bb.grad.clone()
auto_state = state.grad.clone()

# ── Manual backward ──
w_decay = torch.exp(-torch.exp(ww))
all_states = [state.clone().detach()]
s = state.clone().detach()
for t in range(T):
    state_t = all_states[-1]
    ww_t = w_decay[:, t].unsqueeze(-2)
    kk_t = kk[:, t].unsqueeze(-2)
    vv_t = vv[:, t].unsqueeze(-1)
    aa_t = aa[:, t].unsqueeze(-1)
    bb_t = bb[:, t].unsqueeze(-2)
    s = s * ww_t + (s @ aa_t) @ bb_t + vv_t @ kk_t
    all_states.append(s.clone().detach())

go = (2.0 * out).detach()
gs = (2.0 * s_out).detach()
go_h = go.view(B, T, H, N)

man_r = torch.zeros_like(rr)
man_w = torch.zeros_like(ww)
man_k = torch.zeros_like(kk)
man_v = torch.zeros_like(vv)
man_a = torch.zeros_like(aa)
man_b = torch.zeros_like(bb)

for t in range(T - 1, -1, -1):
    s_t = all_states[t]
    s_t1 = all_states[t + 1]
    go_t = go_h[:, t]

    r_t = rr[:, t]
    w_decay_t = w_decay[:, t]
    a_t = aa[:, t]
    b_t = bb[:, t]
    v_t = vv[:, t]
    k_t = kk[:, t]

    gs_input = gs + (go_t.unsqueeze(-1) @ r_t.unsqueeze(-2))

    # grad_r = s_{t+1}^T @ go_t
    man_r[:, t] = (s_t1.transpose(-2, -1) @ go_t.unsqueeze(-1)).squeeze(-1)

    # grad_w: decay_grad * d(decay)/d(w_log)
    decay_grad = (gs_input * s_t).sum(dim=-2)
    ww_log_grad = decay_grad * (-w_decay_t * torch.exp(ww[:, t]))
    man_w[:, t] = ww_log_grad

    # grad_a
    man_a[:, t] = (s_t.transpose(-2, -1) @ (gs_input @ b_t.unsqueeze(-1))).squeeze(-1)

    # grad_b
    s_aa = (s_t @ a_t.unsqueeze(-1))
    man_b[:, t] = (s_aa.transpose(-2, -1) @ gs_input).squeeze(-2)

    # grad_v
    man_v[:, t] = (gs_input @ k_t.unsqueeze(-1)).squeeze(-1)

    # grad_k
    man_k[:, t] = (v_t.unsqueeze(-1).transpose(-2, -1) @ gs_input).squeeze(-2)

    # propagate gs
    gs = gs_input * w_decay_t.unsqueeze(-2) + (gs_input @ b_t.unsqueeze(-1)) @ a_t.unsqueeze(-2)

# Compare
for name, auto, man in [
    ('r', auto_r, man_r), ('w', auto_w, man_w), ('k', auto_k, man_k),
    ('v', auto_v, man_v), ('a', auto_a, man_a), ('b', auto_b, man_b),
]:
    d = (auto - man).abs().max().item()
    status = 'PASS' if d < 1e-3 else 'FAIL'
    print(f'{name}: max_diff={d:.6f} {status}')

d_state = (auto_state - gs).abs().max().item()
print(f'state: max_diff={d_state:.6f} {"PASS" if d_state < 1e-3 else "FAIL"}')
