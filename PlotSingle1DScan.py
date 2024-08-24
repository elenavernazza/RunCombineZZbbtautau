import os, sys, pdb, uproot
import numpy as np
import matplotlib.pyplot as plt
import mplhep
plt.style.use(mplhep.style.CMS)

import warnings, logging
warnings.filterwarnings("ignore", message=".*Type 3 font.*")
logging.getLogger('matplotlib').setLevel(logging.ERROR)

def GetDeltaLL(LS_file):
    with uproot.open(LS_file) as file:
        limit = file["limit"]
        r = limit.array("r")
        deltaNLL = 2 * limit.array("deltaNLL")
    sorted_indices = np.argsort(r)
    r_sorted = r[sorted_indices]
    deltaNLL_sorted = deltaNLL[sorted_indices]
    return r_sorted[1:], deltaNLL_sorted[1:]

def GetLimit(LS_file):
    file = uproot.open(LS_file)
    limit_tree = file["limit"]
    limit_value = limit_tree["limit"].array()[0]
    return limit_value

def SetStyle(fig, x, line1="", line2="", max=5):
    plt.axhline(y=1, color='gray', linestyle='--', linewidth=2)
    plt.axhline(y=3.84, color='gray', linestyle='--', linewidth=2)
    try:
        plt.text(x[0] + 0.05, 1 + 0.1, '68% C.L.', fontsize=18)
        plt.text(x[0] + 0.05, 3.84 + 0.1, '95% C.L.', fontsize=18)
    except:
        print(" ### INFO: C.L. not found")
    plt.text(0.03, 0.97, line1, ha="left", va="top", transform=plt.gca().transAxes, color="black", bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    plt.text(0.03, 0.92, line2, ha="left", va="top", transform=plt.gca().transAxes, color="black", bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    mplhep.cms.label(data=False)
    plt.xlabel('$\\mu$')
    plt.ylabel('-2 $\\Delta LL$')
    plt.title("")
    # plt.xlim(x[0], x[-1])
    # plt.ylim(0,max)
    plt.grid()

if __name__ == "__main__" :

    if len(sys.argv) < 2:
        print(" ### ERROR: Usage is 'python PlotSingle1DScan.py <file.root>'")
        sys.exit(1)

    LS_file = sys.argv[1]
    x, y = GetDeltaLL(LS_file)

    if len(sys.argv) > 2:
        name = sys.argv[2]
    else:
        name = 'output'

    fig = plt.figure(figsize=(10, 10))
    plt.plot(x, y, label='Data', color='red', linewidth=3)
    SetStyle(fig, x)
    print(f" ### INFO: Saving output to {name}.png")
    plt.savefig(f"{name}.png")
    plt.savefig(f"{name}.pdf")
    plt.close()