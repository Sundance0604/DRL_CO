"""Portable renderer backend: no model/platform imports, no external TeX dependency."""
import json
import math
from pathlib import Path
import numpy as np

# Okabe-Ito, supplemented by markers, dashes and hatches for monochrome printing.
COLORS = ["#0072B2","#D55E00","#009E73","#CC79A7","#E69F00","#56B4E9","#000000"]
MARKERS = ["o","s","^","D","v","P","X"]
DASHES = ["-","--","-.",":"]

def render(config, panels, directory):
    import logging
    logging.getLogger("fontTools.subset").setLevel(logging.WARNING)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    width = 3.35 if config["width"] == "single" else 7.0
    columns = min(config["columns"], len(panels))
    rows = math.ceil(len(panels)/columns)
    # Increase panel height with multi-panel figures while retaining paper width.
    height = rows * max(2.4, width/columns*.70)
    style = {"font.family":"serif","font.serif":["DejaVu Serif"],"font.size":8,
             "axes.labelsize":8,"axes.titlesize":8,"legend.fontsize":6.5,
             "xtick.labelsize":7,"ytick.labelsize":7,"axes.spines.top":False,
             "axes.spines.right":False,"axes.linewidth":.65,"lines.linewidth":1.2,
             "pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none",
             "savefig.dpi":config["dpi"],"mathtext.fontset":"dejavuserif"}
    with plt.rc_context(style):
        fig, axes = plt.subplots(rows, columns, figsize=(width,height), squeeze=False, layout="constrained")
        for panel_no, (ax, panel) in enumerate(zip(axes.flat, panels)):
            series = panel["series"]
            if config["kind"] == "heatmap":
                for s in series:
                    xs = sorted({p["x"] for p in s["points"]})
                    ys = sorted({p["y"] for p in s["points"]})
                    z = np.full((len(ys),len(xs)), np.nan)
                    for p in s["points"]: z[ys.index(p["y"]),xs.index(p["x"])] = p["mean"]
                    im = ax.imshow(z,origin="lower",aspect="auto",cmap="cividis")
                    ax.set_xticks(range(len(xs)),[str(x) for x in xs])
                    ax.set_yticks(range(len(ys)),[str(y) for y in ys])
                    # Numeric annotations allow black-and-white interpretation.
                    for y in range(len(ys)):
                        for x in range(len(xs)):
                            if np.isfinite(z[y,x]):
                                ax.text(x,y,f"{z[y,x]:.3g}",ha="center",va="center",fontsize=6,
                                        color="white" if z[y,x] < np.nanmean(z) else "black")
                    fig.colorbar(im,ax=ax,label=panel["ylabel"],fraction=.046,pad=.04)
            elif config["kind"] in {"comparison","paired"}:
                for i, s in enumerate(series):
                    p = s["points"][0]
                    color=COLORS[i%len(COLORS)]
                    ax.scatter([i]*len(s.get("samples",[])),s.get("samples",[]),s=9,
                               facecolors="none",edgecolors=color,alpha=.6)
                    ax.plot(i,p["mean"],marker=MARKERS[i%len(MARKERS)],color=color,markersize=5)
                    if p["ci_low"] is not None:
                        ax.errorbar(i,p["mean"],yerr=[[p["mean"]-p["ci_low"]],[p["ci_high"]-p["mean"]]],
                                    fmt="none",color=color,capsize=3,linewidth=1)
                ax.set_xticks(range(len(series)),[s["label"] for s in series],rotation=25,ha="right")
                if config["kind"] == "paired": ax.axhline(0,color="#333333",linewidth=.7,linestyle="--")
            else:
                for i,s in enumerate(series):
                    points=sorted(s["points"],key=lambda p:p["x"])
                    x=[p["x"] for p in points]; y=[p["mean"] for p in points]
                    color=COLORS[i%len(COLORS)]
                    ax.plot(x,y,label=s["label"],color=color,marker=MARKERS[i%len(MARKERS)],
                            linestyle=DASHES[i%len(DASHES)],markersize=3)
                    low=[p["ci_low"] if p["ci_low"] is not None else np.nan for p in points]
                    high=[p["ci_high"] if p["ci_high"] is not None else np.nan for p in points]
                    ax.fill_between(x,low,high,color=color,alpha=.15)
                ax.legend(frameon=False,loc="best")
            ax.set_title(f"({chr(97+panel_no)}) {panel['title']}",loc="left",wrap=True)
            ax.set_xlabel(panel["xlabel"]); ax.set_ylabel(panel["ylabel"] if config["kind"]!="heatmap" else panel["y_parameter"])
            if config["kind"]!="heatmap": ax.grid(axis="y",color="#dedede",linewidth=.5)
        for ax in list(axes.flat)[len(panels):]: ax.set_visible(False)
        directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
        artifacts=[]
        for fmt in dict.fromkeys(config["formats"]):
            name="figure."+fmt
            fig.savefig(directory/name,dpi=config["dpi"],facecolor="white")
            artifacts.append(name)
        plt.close(fig)
    return artifacts

if __name__ == "__main__":
    here=Path(__file__).resolve().parent
    render(json.loads((here/"plotting-config.json").read_text(encoding="utf-8")),
           json.loads((here/"plot-data.json").read_text(encoding="utf-8")),here)
