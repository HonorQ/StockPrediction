"""反馈训练完整队列（后台串联，避免并发争用同一 SQLite DB）：

  阶段1(--phase 1) → 阶段2(--phase 2) → 清空反馈库 → --all(干净重训)

设计要点：
- 各阶段严格串行，使用日志中的「✅ 完成」标记判断真正结束（不靠报告文件名，
  因为报告文件名固定为 feedback_training_2026年7月.md / _2026年上半年.md，
  且可能被上一次运行残留，只有进程真正跑完才会打印 finally 里的 ✅ 完成）。
- --all 的 per-stock 调权是「基于当前权重的乘性调整」(old_w * 0.5/0.7/1.1)，
  属叠加式。若不清空 stock_factor_weights 直接重跑，会重复累加污染权重。
  故 --all 前清空全部反馈相关表（含 stock_factor_weights / stock_weight_history），
  让 --all 从全局权重(config.yaml)干净重训，产出的两份报告为权威产物。
- 任一阵列进程异常退出（无 ✅ 完成标记）则停止队列并告警，避免进入不一致状态。

监控：
  阶段1: logs/phase1.out.log
  阶段2: logs/phase2.out.log
  --all:  logs/phase_all.out.log
  队列:  logs/launcher.log
"""
import time
import os
import sys
import sqlite3
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(ROOT, "data", "stock_history.db")
LOG = os.path.join(ROOT, "logs", "launcher.log")
DONE_MARK = "✅ 完成"


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def phase_done(log_path, min_age=30):
    """日志含 ✅ 完成 且文件 mtime 距现在 > min_age 秒（确保进程已退出、flush 完毕）。"""
    if not os.path.exists(log_path):
        return False
    try:
        txt = open(log_path, encoding="utf-8", errors="replace").read()
    except Exception:
        return False
    if DONE_MARK not in txt:
        return False
    return (time.time() - os.path.getmtime(log_path)) > min_age


def wait_stage(name, log_path, proc, timeout_min, min_age=30):
    """等待子进程结束且日志出现 ✅ 完成；返回 True 表示成功。"""
    t0 = time.time()
    while True:
        if proc.poll() is not None:
            # 进程已退出
            if phase_done(log_path, min_age=5):
                log(f"{name} 子进程退出且出现完成标记")
                return True
            log(f"[错误] {name} 子进程已退出(PID={proc.pid}, rc={proc.returncode}) "
                f"但日志无「✅ 完成」标记，请查 {log_path}")
            return False
        if phase_done(log_path, min_age=min_age):
            log(f"{name} 检测到完成标记（进程 PID={proc.pid}）")
            return True
        if time.time() - t0 > timeout_min * 60:
            log(f"[错误] {name} 等待超时({timeout_min}min)，停止队列")
            return False
        time.sleep(30)


def clear_feedback_db():
    """清空全部反馈训练相关表，重点是 stock_factor_weights（避免 --all 叠加污染）。"""
    tables = [
        "recommendations",
        "factor_snapshots",
        "factor_attribution",
        "global_weight_history",
        "stock_factor_weights",
        "stock_weight_history",
    ]
    # 校验表确实存在，避免误清其它表
    conn = sqlite3.connect(DB_PATH)
    try:
        existing = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        cleared = []
        for t in tables:
            if t in existing:
                n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                conn.execute(f"DELETE FROM {t}")
                cleared.append(f"{t}({n})")
        conn.commit()
        log("已清空反馈库表: " + ", ".join(cleared))
    finally:
        conn.close()


def run(args, out_name):
    out = open(os.path.join(ROOT, "logs", out_name), "w", encoding="utf-8")
    err = open(os.path.join(ROOT, "logs", out_name.replace(".out.", ".err.")), "w", encoding="utf-8")
    p = subprocess.Popen(
        [sys.executable, "-u", "run_feedback_training.py", *args],
        cwd=ROOT, stdout=out, stderr=err,
    )
    return p


def main():
    log("=== 反馈训练队列启动：阶段1 → 阶段2 → 清库 → --all（干净重训）===")

    # ---- 阶段1：已在外部独立运行，仅等待其完成 ----
    log("等待阶段1 完成（logs/phase1.out.log 出现 ✅ 完成）...")
    p1 = None  # 阶段1 非本脚本启动，仅通过日志判断
    t0 = time.time()
    while not phase_done(os.path.join(ROOT, "logs", "phase1.out.log"), min_age=30):
        if time.time() - t0 > 180 * 60:
            log("[错误] 阶段1 等待超时(180min)，停止队列")
            return
        time.sleep(30)
    log("阶段1 完成")

    # ---- 阶段2 ----
    log("启动阶段2：python -u run_feedback_training.py --phase 2")
    p2 = run(["--phase", "2"], "phase2.out.log")
    log(f"阶段2 子进程 PID={p2.pid}")
    if not wait_stage("阶段2", os.path.join(ROOT, "logs", "phase2.out.log"), p2, 300, min_age=60):
        return

    # ---- 清空反馈库（避免 --all 叠加污染 per-stock 权重）----
    clear_feedback_db()

    # ---- --all：干净重训（阶段1+阶段2 单进程，权威产物）----
    log("启动 --all（清空库后从全局权重干净重训，覆盖报告为权威产物）")
    pa = run(["--all"], "phase_all.out.log")
    log(f"--all 子进程 PID={pa.pid}")
    if not wait_stage("--all", os.path.join(ROOT, "logs", "phase_all.out.log"), pa, 600, min_age=60):
        return

    log("=== 队列结束：阶段1 + 阶段2 + --all 全部完成 ===")


if __name__ == "__main__":
    main()
