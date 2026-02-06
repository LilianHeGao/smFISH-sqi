from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def main():
    outdir = Path("output")
    outdir.mkdir(exist_ok=True)

    # MVP: 先生成一个假数据图，证明 pipeline 跑通（后续换成真实 image stats）
    x = np.linspace(0, 10, 200)
    y = np.sin(x) + 0.1*np.random.randn(len(x))

    plt.figure()
    plt.plot(x, y)
    plt.title("SQI MVP: example plot")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    figpath = outdir / "mvp_plot.png"
    plt.savefig(figpath, dpi=200)
    print(f"[OK] Wrote {figpath}")

if __name__ == "__main__":
    main()
