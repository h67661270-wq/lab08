# -*- coding: utf-8 -*-
"""
实验作业八：自制语言模型的知识蒸馏
对应教材：第15章《模型量化与压缩技术》
完整版：含所有报告表所需的字段记录，以及拓展 D（动态量化）、拓展 E（教师规模扫描）支持。

常用命令：
    python distill_lm.py --mode quick                     # 30 秒自检
    python distill_lm.py --mode teacher --retrain         # 训练教师（约 5 分钟）
    python distill_lm.py --mode all                       # 从零 + 蒸馏对比（约 3 分钟）
    python distill_lm.py --mode ablate                    # 16 组 T×α 消融（约 6 分钟）
    python distill_lm.py --mode all --feat 0.1            # 特征蒸馏（选做 A）
    python distill_lm.py --mode all --teacher_C 256       # 教师规模扫描（选做 E）
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import json
import math
import random
import time
import copy
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(max(1, min(8, os.cpu_count() or 4)))
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs_distill")

# ==========================================
# 1. 语料生成：可复现的合成「仿古诗句」
# ==========================================
SUBJ = ["小明", "小红", "阿远", "书生", "渔夫", "牧童", "旅人", "诗人", "少年", "故人", "归客", "山僧", "孤客", "游子"]
PLACE = ["江边", "山寺", "长亭", "渡口", "竹林", "荒村", "城楼", "野渡", "溪桥", "松间", "篱下", "窗前", "陌上", "舟中"]
VERB = ["望见", "听见", "想起", "送去", "寻到", "梦见", "独对", "遥看", "静听", "又逢", "恰遇", "空忆"]
OBJ = ["明月", "落花", "孤舟", "归雁", "残灯", "细雨", "青山", "晚钟", "白雪", "寒江", "疏星", "暮云", "秋草", "远帆"]
MOOD = ["心中惆怅", "不觉天明", "欣然归去", "泪湿衣襟", "长叹一声", "默然无语", "忽忆旧游", "欲语还休", "此意难平", "一笑而别"]

POEMS = [
    "床前明月光,疑是地上霜。举头望明月,低头思故乡。", "春眠不觉晓,处处闻啼鸟。夜来风雨声,花落知多少。",
    "白日依山尽,黄河入海流。欲穷千里目,更上一层楼。", "千山鸟飞绝,万径人踪灭。孤舟蓑笠翁,独钓寒江雪。",
    "空山不见人,但闻人语响。返景入深林,复照青苔上。", "红豆生南国,春来发几枝。愿君多采撷,此物最相思。",
    "独坐幽篁里,弹琴复长啸。深林人不知,明月来相照。", "月落乌啼霜满天,江枫渔火对愁眠。姑苏城外寒山寺,夜半钟声到客船。",
    "两个黄鹂鸣翠柳,一行白鹭上青天。窗含西岭千秋雪,门泊东吴万里船。", "朝辞白帝彩云间,千里江陵一日还。两岸猿声啼不住,轻舟已过万重山。",
    "日照香炉生紫烟,遥看瀑布挂前川。飞流直下三千尺,疑是银河落九天。", "渭城朝雨浥轻尘,客舍青青柳色新。劝君更尽一杯酒,西出阳关无故人。",
    "寒雨连江夜入吴,平明送客楚山孤。洛阳亲友如相问,一片冰心在玉壶。", "独在异乡为异客,每逢佳节倍思亲。遥知兄弟登高处,遍插茱萸少一人。",
    "清明时节雨纷纷,路上行人欲断魂。借问酒家何处有,牧童遥指杏花村。", "朱雀桥边野草花,乌衣巷口夕阳斜。旧时王谢堂前燕,飞入寻常百姓家。"
]

def build_corpus(n_lines=3000, seed=0):
    rng = random.Random(seed)
    out = []
    for i in range(n_lines):
        s, p, v, o, m = (rng.choice(x) for x in (SUBJ, PLACE, VERB, OBJ, MOOD))
        if i % 3 == 0:
            out.append("%s%s%s%s, %s。" % (s, p, v, o, m))
        elif i % 3 == 1:
            out.append("%s在%s%s了一夜%s, %s。" % (s, p, o, rng.choice(["看", "听", "守", "忆"]), m))
        else:
            out.append("%s%s道:%s%s。”%s。" % (s, v, o, rng.choice(["正好", "可怜", "可惜", "依旧"]), m))
        out.append("\n")
    out.extend(POEMS)
    return "".join(out)

# ==========================================
# 2. 字符级分词器
# ==========================================
class Tokenizer:
    def __init__(self, text):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = chars
        self.vocab_size = len(chars)
    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]
    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

# ==========================================
# 3. 迷你 GPT 模型架构 (Pre-Norm)
# ==========================================
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        C = cfg["n_embed"]
        assert C % cfg["n_head"] == 0
        self.n_head = cfg["n_head"]
        self.c_attn = nn.Linear(C, 3 * C)
        self.c_proj = nn.Linear(C, C)
        self.drop = nn.Dropout(cfg["dropout"])
        self.register_buffer("mask", torch.tril(torch.ones(cfg["block_size"], cfg["block_size"])).view(1, 1, cfg["block_size"], cfg["block_size"]))
    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = self.drop(F.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.c_proj(y))

class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        C = cfg["n_embed"]
        self.fc = nn.Linear(C, 4 * C)
        self.proj = nn.Linear(4 * C, C)
        self.drop = nn.Dropout(cfg["dropout"])
    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg["n_embed"])
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg["n_embed"])
        self.mlp = MLP(cfg)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))

class GPT(nn.Module):
    def __init__(self, vocab_size, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(vocab_size, cfg["n_embed"])
        self.pos_emb = nn.Embedding(cfg["block_size"], cfg["n_embed"])
        self.drop = nn.Dropout(cfg["dropout"])
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg["n_layer"])])
        self.ln_f = nn.LayerNorm(cfg["n_embed"])
        self.head = nn.Linear(cfg["n_embed"], vocab_size, bias=False)
        self.apply(self._init)
        self.n_param = sum(p.numel() for p in self.parameters())
    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    def forward(self, idx, targets=None, need_hidden=False):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        if need_hidden:
            return logits, loss, x
        return logits, loss
    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg["block_size"]:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx

# ==========================================
# 4. 数据采样与评估
# ==========================================
def get_batch(data, bs, block, gen):
    ix = torch.randint(len(data) - block - 1, (bs,), generator=gen)
    x = torch.stack([data[i:i + block] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block] for i in ix])
    return x, y

@torch.no_grad()
def evaluate(model, data, bs, block, iters=20, seed=1234):
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    tot = 0.0
    for _ in range(iters):
        x, y = get_batch(data, bs, block, gen)
        _, loss = model(x, y)
        tot += loss.item()
    model.train()
    return tot / iters

@torch.no_grad()
def agreement(student, teacher, data, bs, block, iters=10, seed=4321):
    student.eval(); teacher.eval()
    gen = torch.Generator().manual_seed(seed)
    hit = tot = 0
    for _ in range(iters):
        x, _ = get_batch(data, bs, block, gen)
        a = student(x)[0].argmax(-1)
        b = teacher(x)[0].argmax(-1)
        hit += (a == b).sum().item()
        tot += b.numel()
    student.train(); teacher.train()
    return hit / tot

# ==========================================
# 5. 核心训练引擎
# ==========================================
def train(model, data, iters, bs, block, lr=3e-3, seed=0, teacher=None,
          T=4.0, alpha=0.5, feat_weight=0.0, log_every=50, tag=""):
    proj = None
    if teacher is not None and feat_weight > 0:
        proj = nn.Linear(model.cfg["n_embed"], teacher.cfg["n_embed"])
    params = list(model.parameters()) + (list(proj.parameters()) if proj is not None else [])
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    gen = torch.Generator().manual_seed(seed + 1)
    hist = []
    if teacher is not None:
        teacher.eval()
    model.train()
    t0 = time.time()
    V = model.head.out_features
    for it in range(iters):
        x, y = get_batch(data, bs, block, gen)
        logits = model(x)[0]
        ce = F.cross_entropy(logits.view(-1, V), y.reshape(-1))
        loss = ce
        kd_val = 0.0
        if teacher is not None and alpha > 0:
            with torch.no_grad():
                t_logits = teacher(x)[0]
            s_lp = F.log_softmax(logits.reshape(-1, V) / T, dim=-1)
            t_p = F.softmax(t_logits.reshape(-1, V) / T, dim=-1)
            kd = F.kl_div(s_lp, t_p, reduction="batchmean") * (T * T)
            kd_val = kd.item()
            loss = (1 - alpha) * ce + alpha * kd
        if proj is not None:
            _, _, hs = model(x, need_hidden=True)
            with torch.no_grad():
                _, _, ht = teacher(x, need_hidden=True)
            loss = loss + feat_weight * F.mse_loss(proj(hs), ht)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        hist.append(loss.item())
        if (it + 1) % log_every == 0 or it == 0:
            extra = ""
            if teacher is not None and alpha > 0:
                extra = "ce %.4f kd(T^2) %.4f" % (ce.item(), kd_val)
            print(" [%s] iter %4d/%d loss %.4f %s (%.1fs)" %
                  (tag or "train", it + 1, iters, loss.item(), extra, time.time() - t0))
    return hist, time.time() - t0

# ==========================================
# 6. 绘图与保存
# ==========================================
def save(res, model=None, tok=None, tag="result"):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"results_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    if model is not None:
        torch.save(model.state_dict(), os.path.join(OUT_DIR, f"model_{tag}.pt"))

def plot_curves(res, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        print(f"[plot] 跳过绘图（{e}）")
        return
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for k, lab in (("teacher", "Teacher"), ("scratch", "Student-scratch"), ("kd", "Student-KD")):
        if k in res and "curve" in res[k]:
            c = res[k]["curve"]
            ax[0].plot(range(1, len(c) + 1), c, label=lab, linewidth=1.2)
    ax[0].set_xlabel("iteration"); ax[0].set_ylabel("training loss")
    ax[0].set_title("Training Loss Curves"); ax[0].legend()
    if "ablation" in res:
        Ts = sorted(set(r["T"] for r in res["ablation"]))
        alphas = sorted(set(r["alpha"] for r in res["ablation"]))
        grid = np.zeros((len(Ts), len(alphas)))
        for r in res["ablation"]:
            grid[Ts.index(r["T"]), alphas.index(r["alpha"])] = r["val_loss"]
        im = ax[1].imshow(grid, cmap="viridis_r", aspect="auto")
        ax[1].set_xticks(range(len(alphas)))
        ax[1].set_xticklabels([str(a) for a in alphas])
        ax[1].set_yticks(range(len(Ts)))
        ax[1].set_yticklabels([str(t) for t in Ts])
        ax[1].set_xlabel("alpha"); ax[1].set_ylabel("T")
        ax[1].set_title("Ablation Val Loss")
        fig.colorbar(im, ax=ax[1])
        for i in range(len(Ts)):
            for j in range(len(alphas)):
                ax[1].text(j, i, f"{grid[i,j]:.3f}", ha="center", va="center",
                           color="white" if grid[i,j] > grid.mean() else "black")
    plt.tight_layout(); plt.savefig(path); plt.close()
    print(f"[plot] 曲线已保存至 {path}")

# ==========================================
# 7. 消融实验
# ==========================================
def ablate(args, tok, train_ids, teacher, val_ids, bs, block, res, student_cfg):
    grid_T = [1, 2, 4, 8]
    grid_a = [0.3, 0.5, 0.7, 0.9]
    V = tok.vocab_size
    gen = torch.Generator().manual_seed(2024)
    print("[ablate] 缓存教师软标签...")
    cache = []
    teacher.eval()
    with torch.no_grad():
        for _ in range(32):
            x, y = get_batch(train_ids, bs, block, gen)
            cache.append((x, y, teacher(x)[0]))
    rows = []
    for T in grid_T:
        for a in grid_a:
            torch.manual_seed(args.seed + 7)
            m = GPT(tok.vocab_size, student_cfg)
            opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=0.01)
            m.train()
            for it in range(args.abl_iters):
                x, y, t1 = cache[it % len(cache)]
                logits = m(x)[0]
                ce = F.cross_entropy(logits.view(-1, V), y.reshape(-1))
                s_lp = F.log_softmax(logits.reshape(-1, V) / T, dim=-1)
                t_p = F.softmax(t1.reshape(-1, V) / T, dim=-1)
                kd = F.kl_div(s_lp, t_p, reduction="batchmean") * (T * T)
                loss = (1 - a) * ce + a * kd
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
            v = evaluate(m, val_ids, bs, block)
            print(f"  [ablate] T={T} alpha={a:.1f} -> val_loss={v:.4f}")
            rows.append({"T": T, "alpha": a, "val_loss": v, "ppl": math.exp(v)})
    res["ablation"] = rows
    best = min(rows, key=lambda r: r["val_loss"])
    worst = max(rows, key=lambda r: r["val_loss"])
    print(f"\n[ablate] 最优: T={best['T']} alpha={best['alpha']} val={best['val_loss']:.4f}")
    print(f"[ablate] 最差: T={worst['T']} alpha={worst['alpha']} val={worst['val_loss']:.4f}")

# ==========================================
# 8. 【补丁 5】动态量化导出（拓展 D）
# ==========================================
def run_quantization(student, tok, val_ids, bs, block, res):
    """拓展 D：int8 动态量化，记录体积 / 延迟 / loss 变化"""
    try:
        import torch.quantization as tq
    except ImportError:
        print("[quant] 当前 PyTorch 不支持 quantization，跳过")
        return
    def measure(model, iters=20):
        model.eval()
        gen = torch.Generator().manual_seed(1234)
        tot = 0.0
        t0 = time.time()
        with torch.no_grad():
            for _ in range(iters):
                x, y = get_batch(val_ids, bs, block, gen)
                _, loss = model(x, y)
                tot += loss.item()
        dt = (time.time() - t0) / iters
        return tot / iters, dt
    # 深拷贝一份 fp32 基线，避免量化过程污染原模型
    fp32_model = copy.deepcopy(student)
    fp32_model.eval()
    loss_fp32, t_fp32 = measure(fp32_model)
    fp32_path = os.path.join(OUT_DIR, "student_fp32.pt")
    torch.save(fp32_model.state_dict(), fp32_path)
    size_fp32 = os.path.getsize(fp32_path) / 1024  # KB
    try:
        q = tq.quantize_dynamic(fp32_model, {nn.Linear}, dtype=torch.qint8)
        loss_int8, t_int8 = measure(q)
        int8_path = os.path.join(OUT_DIR, "student_int8.pt")
        torch.save(q.state_dict(), int8_path)
        size_int8 = os.path.getsize(int8_path) / 1024
    except Exception as e:
        print(f"[quant] 量化失败：{e}")
        return
    res["quant"] = {
        "fp32_size_KB": round(size_fp32, 1),
        "int8_size_KB": round(size_int8, 1),
        "size_ratio": round(size_fp32 / size_int8, 2),
        "fp32_val_loss": round(loss_fp32, 4),
        "int8_val_loss": round(loss_int8, 4),
        "loss_delta": round(loss_int8 - loss_fp32, 4),
        "fp32_ms_per_iter": round(t_fp32 * 1000, 2),
        "int8_ms_per_iter": round(t_int8 * 1000, 2),
    }
    print(f"[quant] fp32 {size_fp32:.0f}KB -> int8 {size_int8:.0f}KB "
          f"({res['quant']['size_ratio']}x), loss {loss_fp32:.4f}->{loss_int8:.4f}, "
          f"fp32 {t_fp32*1000:.1f}ms -> int8 {t_int8*1000:.1f}ms / iter")

# ==========================================
# 9. 主流程
# ==========================================
STUDENT_CFG = dict(n_layer=2, n_head=4, n_embed=64, block_size=96, dropout=0.0)

def make_data(block):
    train_text = build_corpus(3000, seed=0)
    val_text = build_corpus(300, seed=999)
    tok = Tokenizer(train_text)
    train_ids = torch.tensor(tok.encode(train_text), dtype=torch.long)
    val_ids = torch.tensor(tok.encode(val_text), dtype=torch.long)
    return tok, train_ids, val_ids

def main(args):
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(args.seed)

    # 【补丁 6】教师 cfg 动态构造，支持 --teacher_C 规模扫描
    teacher_cfg = dict(n_layer=2, n_head=4, n_embed=args.teacher_C,
                       block_size=STUDENT_CFG["block_size"], dropout=0.0)
    student_cfg = dict(STUDENT_CFG)  # 学生固定 64 维

    tok, train_ids, val_ids = make_data(teacher_cfg["block_size"])
    bs, block = args.batch_size, teacher_cfg["block_size"]
    print(f"[data] vocab={tok.vocab_size} train_chars={len(train_ids)} "
          f"val_chars={len(val_ids)} block={block} batch={bs}")

    # 【补丁 1】记录完整 cfg，供报告表 5 填写
    res = {
        "vocab": tok.vocab_size, "block": block, "batch": bs,
        "train_chars": int(len(train_ids)), "val_chars": int(len(val_ids)),
        "cfg": {"teacher": teacher_cfg, "student": student_cfg},
        "args": vars(args),
    }

    teacher_path = os.path.join(OUT_DIR, "teacher.pt")

    # ---------- 教师 ----------
    torch.manual_seed(args.seed)
    teacher = GPT(tok.vocab_size, teacher_cfg)
    print(f"[teacher] params = {teacher.n_param / 1e6:.3f} M")

    if os.path.exists(teacher_path) and not args.retrain and args.mode != "teacher":
        teacher.load_state_dict(torch.load(teacher_path))
        tv = evaluate(teacher, val_ids, bs, block)
        # 【补丁 2】教师记录补齐 iters + agree
        res["teacher"] = {
            "params": teacher.n_param, "val_loss": tv, "ppl": math.exp(tv),
            "time_s": 0.0, "cached": True,
            "iters": args.teacher_iters, "agree": 1.0, "curve": [],
        }
        print(f"[teacher] 复用 {os.path.basename(teacher_path)} val_loss={tv:.4f} ppl={math.exp(tv):.2f}")
    else:
        print("[teacher] 开始训练教师模型...")
        th, ttime = train(teacher, train_ids, args.teacher_iters, bs, block,
                          lr=args.lr, seed=args.seed, log_every=args.log_every, tag="teacher")
        tv = evaluate(teacher, val_ids, bs, block)
        torch.save(teacher.state_dict(), teacher_path)
        res["teacher"] = {
            "params": teacher.n_param, "val_loss": tv, "ppl": math.exp(tv),
            "time_s": ttime,
            "iters": args.teacher_iters, "agree": 1.0, "curve": th,
        }
        print(f"[teacher] 训练完毕 val_loss={tv:.4f} ppl={math.exp(tv):.2f} 耗时={ttime:.1f}s")

    # tag 命名：默认跑法 tag=mode；规模扫描 tag 带 C 后缀，避免覆盖
    save_tag = args.mode if args.teacher_C == 160 else f"{args.mode}_C{args.teacher_C}"

    if args.mode == "teacher":
        save(res, teacher, tok, tag=save_tag)
        return

    if args.mode == "ablate":
        ablate(args, tok, train_ids, teacher, val_ids, bs, block, res, student_cfg)
        save(res, teacher, tok, tag=save_tag)
        plot_curves(res, os.path.join(OUT_DIR, f"curves_{save_tag}.png"))
        print(f"\n[done] 结果已写入 {OUT_DIR}")
        return

    # ---------- 学生 ----------
    torch.manual_seed(args.seed + 7)
    probe = GPT(tok.vocab_size, student_cfg)
    sp = probe.n_param
    # 【补丁 1】记录压缩比，供报告表 5 填写
    res["compress_ratio"] = round(teacher.n_param / sp, 2)
    print(f"[student] params = {sp / 1e6:.3f} M (压缩率 {res['compress_ratio']:.2f}x)")

    def new_student():
        torch.manual_seed(args.seed + 7)
        return GPT(tok.vocab_size, student_cfg)

    # 【补丁 3】eval_student 增加 iters 参数，写进 json
    def eval_student(m, hist, tcost, name, iters, **extra):
        v = evaluate(m, val_ids, bs, block)
        row = {
            "params": sp, "val_loss": v, "ppl": math.exp(v),
            "time_s": tcost, "iters": iters,
            "curve": hist, "agree": agreement(m, teacher, val_ids, bs, block),
        }
        row.update(extra)
        print(f"[{name}] val_loss={v:.4f} ppl={math.exp(v):.2f} agree={row['agree']:.3f} {tcost:.1f}s")
        res[name] = row
        return row

    # 从零训练
    print("\n>>> 从零训练 (Scratch) ...")
    s0 = new_student()
    h0, t0 = train(s0, train_ids, args.student_iters, bs, block,
                   lr=args.lr, seed=args.seed, log_every=args.log_every, tag="scratch")
    row0 = eval_student(s0, h0, t0, "scratch", args.student_iters, T=None, alpha=0.0)

    # 蒸馏训练
    print("\n>>> 蒸馏训练 (KD) ...")
    s1 = new_student()
    h1, t1 = train(s1, train_ids, args.student_iters, bs, block,
                   lr=args.lr, seed=args.seed, teacher=teacher,
                   T=args.T, alpha=args.alpha, feat_weight=args.feat,
                   log_every=args.log_every, tag="kd")
    row1 = eval_student(s1, h1, t1, "kd", args.student_iters,
                        T=args.T, alpha=args.alpha, feat=args.feat)

    print(f"\n[对比] 从零 {row0['val_loss']:.4f} -> 蒸馏 {row1['val_loss']:.4f} "
          f"相对下降 {100 * (row0['val_loss'] - row1['val_loss']) / row0['val_loss']:.1f}%")

    # 【补丁 4】生成三组样例，供报告表 8 填写
    print("\n>>> 生成样例（三组对比，prompt='月'）...")
    prompt = "月"
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    samples = {}
    for name, m in [("teacher", teacher), ("scratch", s0), ("kd", s1)]:
        with torch.no_grad():
            g = m.generate(ids, 60, temperature=0.9, top_k=20)
        samples[name] = tok.decode(g[0].tolist())
        print(f"[generate-{name}] " + samples[name].replace("\n", " "))
    res["samples"] = samples

    # 【补丁 5】拓展 D：动态量化
    print("\n>>> 动态量化 (int8) ...")
    run_quantization(s1, tok, val_ids, bs, block, res)

    plot_curves(res, os.path.join(OUT_DIR, f"curves_{save_tag}.png"))
    save(res, s1, tok, tag=save_tag)
    print(f"\n[done] 结果与曲线已写入 {OUT_DIR}")

# ==========================================
# 10. CLI
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Language Model Knowledge Distillation")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["quick", "teacher", "all", "ablate"], help="运行模式")
    parser.add_argument("--seed", type=int, default=42, help="全局随机种子")
    parser.add_argument("--batch_size", type=int, default=24, help="批大小")
    parser.add_argument("--lr", type=float, default=3e-3, help="学习率")
    parser.add_argument("--teacher_iters", type=int, default=120,
                        help="教师训练步数（指南实测 120 步约 5 分钟；报告模板默认写 200，如需对齐请传 200）")
    parser.add_argument("--student_iters", type=int, default=120, help="学生训练步数")
    parser.add_argument("--abl_iters", type=int, default=80, help="消融实验每组步数")
    parser.add_argument("--log_every", type=int, default=50, help="打印日志间隔")
    parser.add_argument("--T", type=float, default=4.0, help="蒸馏温度系数")
    parser.add_argument("--alpha", type=float, default=0.5, help="蒸馏损失权重")
    parser.add_argument("--feat", type=float, default=0.0, help="特征蒸馏权重（选做 A）")
    parser.add_argument("--retrain", action="store_true", help="强制重新训练教师模型")
    # 【补丁 6】拓展 E：教师规模扫描
    parser.add_argument("--teacher_C", type=int, default=160,
                        help="教师 n_embed 宽度（拓展 E：可试 96/160/256）")
    args = parser.parse_args()

    if args.mode == "quick":
        print("[quick] 环境自检模式（结果无参考价值）...")
        args.teacher_iters = 5
        args.student_iters = 5
        args.log_every = 1

    main(args)