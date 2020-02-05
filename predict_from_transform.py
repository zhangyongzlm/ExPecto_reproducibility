import argparse
import xgboost as xgb
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import h5py
from six.moves import reduce

parser = argparse.ArgumentParser(
    description="Predict gene expression based on Spatial transformed chromatin features in npy format."
)
parser.add_argument(
    "--modelList",
    action="store",
    dest="modelList",
    help="A list of paths of binary xgboost model files (if end with .list) or a combined model file (if ends with .csv).",
)
parser.add_argument(
    "--inputFile",
    action="store",
    dest="inputFile",
    help="Spatial transformed chromatin features in npy format",
)
parser.add_argument(
    "--threads",
    action="store",
    dest="threads",
    type=int,
    default=16,
    help="Number of threads.",
)
parser.add_argument(
    "--annoFile",
    action="store",
    dest="annoFile",
    default="/research/dept6/zhangy/project/Hirschsprung/ExPecto/resources/geneanno.csv",
)
parser.add_argument(
    "--oldFormat",
    action="store",
    dest="oldFormat",
    type=bool,
    default=False,
    help="backward compatibility with earlier model format",
)


args = parser.parse_args()

modelList = pd.read_csv(args.modelList, sep="\t", header=0)
models = []
for file in modelList["ModelName"]:
    bst = xgb.Booster({"nthread": args.threads})
    bst.load_model(file.strip())
    models.append(bst)


Xreducedall = np.load(args.inputFile)

if args.oldFormat:
    print("oldFormat == True")
    Xreducedall = np.concatenate(
        [np.zeros((Xreducedall.shape[0], 10, 1)), Xreducedall.reshape((-1, 10, 2002))],
        axis=2,
    ).reshape((-1, 20030))

effect = np.zeros((Xreducedall.shape[0], len(models)))
dtest_alt = xgb.DMatrix(Xreducedall)

for i in range(len(models)):
    print("Apply model {}/{}:".format(i + 1, len(models)))
    effect[:, i] = models[i].predict(dtest_alt)

geneanno = pd.read_csv(args.annoFile)
combined = pd.concat(
    [geneanno, pd.DataFrame(effect, columns=modelList.iloc[:, 1])],
    axis=1,
    ignore_index=False,
)
combined.to_csv(
    "gene_exp_predict/" + args.inputFile + ".exp.csv", header=True, index=False
)

# combined.to_csv(
#     "spec_exp.csv", header=True, index=False
# )

# np.save('allexp.npy', effect)
