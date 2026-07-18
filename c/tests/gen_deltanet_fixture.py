#!/usr/bin/env python3
"""Reference implementation of the Qwen3.5 Gated DeltaNet recurrence
(transformers: torch_recurrent_gated_delta_rule + modular_qwen3_5), in pure
Python double precision. Emits a C header with inputs and expected outputs.

Dims: D=4, Hv=2, Hk=1 (R=2), dk=4, dv=4, K=3 (conv kernel), T=3 tokens.
"""
import math

D, Hv, Hk, dk, dv, K, T = 4, 2, 1, 4, 4, 3, 3
kd, vd = Hk*dk, Hv*dv
cd = 2*kd + vd
EPS = 1e-6

state = 123456789
def rnd():
    global state
    state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
    return ((state >> 33) / float(1 << 31)) - 0.5

def silu(x): return x / (1 + math.exp(-x))
def softplus(x): return x if x > 20 else math.log1p(math.exp(x))
def sigmoid(x): return 1 / (1 + math.exp(-x))

W_qkv  = [[rnd() for _ in range(D)] for _ in range(cd)]
W_z    = [[rnd() for _ in range(D)] for _ in range(vd)]
W_b    = [[rnd() for _ in range(D)] for _ in range(Hv)]
W_a    = [[rnd() for _ in range(D)] for _ in range(Hv)]
conv_w = [[rnd() for _ in range(K)] for _ in range(cd)]
conv_b = [rnd()*0.1 for _ in range(cd)]
dt_bias= [rnd() for _ in range(Hv)]
A_log  = [rnd() for _ in range(Hv)]
norm_w = [1 + rnd()*0.1 for _ in range(dv)]
W_out  = [[rnd() for _ in range(vd)] for _ in range(D)]
X      = [[rnd() for _ in range(D)] for _ in range(T)]

conv_state = [[0.0]*K for _ in range(cd)]
S = [[[0.0]*dv for _ in range(dk)] for _ in range(Hv)]
outs = []
for t in range(T):
    x = X[t]
    qkv = [sum(W_qkv[o][i]*x[i] for i in range(D)) for o in range(cd)]
    z   = [sum(W_z[o][i]*x[i] for i in range(D)) for o in range(vd)]
    b   = [sum(W_b[o][i]*x[i] for i in range(D)) for o in range(Hv)]
    a   = [sum(W_a[o][i]*x[i] for i in range(D)) for o in range(Hv)]
    # causal depthwise conv1d on qkv only, then silu
    for ch in range(cd):
        cs = conv_state[ch]
        cs.pop(0); cs.append(qkv[ch])
        v = sum(cs[j]*conv_w[ch][j] for j in range(K)) + conv_b[ch]
        qkv[ch] = silu(v)
    q = qkv[:kd]; k = qkv[kd:2*kd]; v = qkv[2*kd:]
    # l2norm per head + q scale
    for h in range(Hk):
        for vec in (q, k):
            seg = vec[h*dk:(h+1)*dk]
            r = 1/math.sqrt(sum(s*s for s in seg) + EPS)
            for i in range(dk): vec[h*dk+i] *= r
    qs = 1/math.sqrt(dk)
    q = [qi*qs for qi in q]
    o_all = []
    R = Hv // Hk
    for hv in range(Hv):
        hk = hv // R
        qh = q[hk*dk:(hk+1)*dk]; kh = k[hk*dk:(hk+1)*dk]; vh = v[hv*dv:(hv+1)*dv]
        g    = -math.exp(A_log[hv]) * softplus(a[hv] + dt_bias[hv])
        beta = sigmoid(b[hv])
        dec  = math.exp(g)
        Sh = S[hv]
        for i in range(dk):
            for j in range(dv): Sh[i][j] *= dec
        kv = [sum(Sh[i][j]*kh[i] for i in range(dk)) for j in range(dv)]
        delta = [(vh[j]-kv[j])*beta for j in range(dv)]
        for i in range(dk):
            for j in range(dv): Sh[i][j] += kh[i]*delta[j]
        oh = [sum(Sh[i][j]*qh[i] for i in range(dk)) for j in range(dv)]
        # gated rmsnorm per head: norm(o)*w * silu(z)
        ms = sum(x_*x_ for x_ in oh)/dv
        r = 1/math.sqrt(ms + EPS)
        zh = z[hv*dv:(hv+1)*dv]
        oh = [oh[j]*r*norm_w[j]*silu(zh[j]) for j in range(dv)]
        o_all += oh
    out = [sum(W_out[d][j]*o_all[j] for j in range(vd)) for d in range(D)]
    outs.append(out)

def carr(name, flat):
    vals = ", ".join(f"{v:.9g}f" for v in flat)
    return f"static const float {name}[] = {{ {vals} }};"

flat = lambda m: [x for row in m for x in row]
print("/* GENERATO da gen_deltanet_fixture.py: ricorrenza gated delta rule di")
print(" * riferimento in Python double (formule transformers qwen3_5). NON editare. */")
print(f"#define FX_D {D}\n#define FX_HV {Hv}\n#define FX_HK {Hk}")
print(f"#define FX_DK {dk}\n#define FX_DV {dv}\n#define FX_K {K}\n#define FX_T {T}")
print(carr("FX_W_QKV", flat(W_qkv)))
print(carr("FX_W_Z", flat(W_z)))
print(carr("FX_W_B", flat(W_b)))
print(carr("FX_W_A", flat(W_a)))
print(carr("FX_CONV_W", flat(conv_w)))
print(carr("FX_CONV_B", conv_b))
print(carr("FX_DT_BIAS", dt_bias))
print(carr("FX_A_LOG", A_log))
print(carr("FX_NORM_W", norm_w))
print(carr("FX_W_OUT", flat(W_out)))
print(carr("FX_X", flat(X)))
print(carr("FX_EXPECT", flat(outs)))
