import re
import glob
from pathlib import Path
import umap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def feature_map(feature, savename):
    project_dir = Path(__file__).resolve().parents[2]
    rslt_path = project_dir/'result'
    label = np.load(str(rslt_path/'no_train_label.npy')).squeeze()

    embedding = umap.UMAP(random_state=42).fit_transform(feature)
    embedding_x = embedding[:, 0]
    embedding_y = embedding[:, 1]

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    for n in np.unique(label):
        ax.scatter(embedding_x[label == n],
                    embedding_y[label == n],
                    label=n, s=1)
    ax.grid()
    ax.legend()
    plt.savefig(savename)
    plt.close()

def main():
    project_dir = Path(__file__).resolve().parents[2]
    rslt_path = project_dir/'result'
    figs_path = project_dir/'reports/figures'
    for l in glob.glob(str(rslt_path/'*feat.npy')):
        feat = np.load(l)
        feature_map(feat, str(figs_path/(Path(l).stem+'_UMAP.png')))

if __name__ == "__main__":
    main()