"""Training a ExPecto sequence-based expression model.

This script takes an expression profile, specified by the expression values
in the targetIndex-th column in expFile. The expression values can be
RPKM from RNA-seq. The rows
of the expFile must match with the genes or TSSes specified in
./resources/geneanno.csv.

Example:
        $ python ./train.py --expFile ./resources/geneanno.exp.csv --targetIndex 1 --output model.adipose


"""
import argparse
import xgboost as xgb
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import h5py
import os

parser = argparse.ArgumentParser(description="Process some integers.")
# parser.add_argument("--targetIndex", action="store", dest="targetIndex", type=int)
parser.add_argument("--output", action="store", dest="output")
parser.add_argument(
    "--expFile",
    action="store",
    dest="expFile",
    default="./geneanno.exp_noidx_renamed.csv",
)
parser.add_argument(
    "--inputFile",
    action="store",
    dest="inputFile",
    # default="/research/dept6/zhangy/project/Hirschsprung/ExPecto/resources/Xreducedall.2002.npy",
)
parser.add_argument(
    "--annoFile", action="store", dest="annoFile", default="./geneanno.csv"
)
parser.add_argument(
    "--evalFile",
    action="store",
    dest="evalFile",
    default="",
    help="specify to save holdout set predictions",
)
parser.add_argument(
    "--filterStr", action="store", dest="filterStr", type=str, default="all"
)
# parser.add_argument(
#     "--pseudocount", action="store", dest="pseudocount", type=float, default=0.0001
# )
parser.add_argument(
    "--num_round", action="store", dest="num_round", type=int, default=100
)
parser.add_argument("--l2", action="store", dest="l2", type=float, default=100)
parser.add_argument("--l1", action="store", dest="l1", type=float, default=0)
parser.add_argument("--eta", action="store", dest="eta", type=float, default=0.01)
parser.add_argument(
    "--base_score", action="store", dest="base_score", type=float, default=2
)
parser.add_argument("--threads", action="store", dest="threads", type=int, default=16)
parser.add_argument("--modellistName", action="store", dest="modellistName")
args = parser.parse_args()

# read resources
Xreducedall = np.load(args.inputFile)
geneanno = pd.read_csv(args.annoFile)

hematopoietic = [
    "Cells_EBV-transformed_lymphocytes",
    "Mobilized_CD34_Primary_Cells_Female",
    "hepatocyte",
    "Peripheral_Blood_Mononuclear_Primary_Cells",
    "CD4_Memory_Primary_Cells",
    "CD8_Naive_Primary_Cells",
    "osteoblast",
    "GM12878_Roadmap",
    "articular_chondrocyte_of_knee_joint",
    "CD4_Naive_Primary_Cells",
]

epithelial = [
    "Penis_Foreskin_Keratinocyte_Primary_Cells_skin03",
    "Penis_Foreskin_Keratinocyte_Primary_Cells_skin02",
    "Penis_Foreskin_Melanocyte_Primary_Cells_skin03",
    "placental_epithelial_cell",
    "epithelial_cell_of_proximal_tubule",
    "hair_follicular_keratinocyte",
    "NHEK",
    "Penis_Foreskin_Melanocyte_Primary_Cells_skin01",
    "tracheal_epithelial_cell",
    "epithelial_cell_of_umbilical_artery",
    "bronchial_epithelial_cell",
    "melanocyte_of_skin",
    "HMEC",
    "hair_follicle_dermal_papilla_cell",
    "kidney_epithelial_cell",
    "pericardium_fibroblast",
    "human_nasal_epithelial_cells",
    "mammary_epithelial_cell",
    "keratinocyte",
    "airway_epithelial_cell",
    "Breast_vHMEC",
    "Breast_Myoepithelial_Cells",
    "renal_cortical_epithelial_cell",
]

stem = [
    "hematopoietic_multipotent_progenitor_cell",
    "induced_pluripotent_stem_cell",
    "H1_Derived_Mesenchymal_Stem_Cells",
    "H1_BMP4_Derived_Trophoblast_Cultured_Cells",
    "H1_Cell_Line",
    "LHCN-M2",
    "myotube",
    "hESC_Derived_CD56+_Mesoderm_Cultured_Cells",
    "HUES64_Cell_Line",
    "Neurosphere_Cultured_Cells_Ganglionic_Eminence_Derived",
    "hESC_Derived_CD56+_Ectoderm_Cultured_Cells",
    "Neurosphere_Cultured_Cells_Cortex_Derived",
    "hESC_Derived_CD184+_Endoderm_Cultured_Cells",
    "H4",
    "H7-hESC",
    "H1_Derived_Neuronal_Progenitor_Cultured_Cells",
    "H1_BMP4_Derived_Mesendoderm_Cultured_Cells",
    "bipolar_spindle_neuron",
    "4star",
    "neural_progenitor_cell",
]

muscle_fib = [
    "Penis_Foreskin_Fibroblast_Primary_Cells_skin01",
    "Cells_Transformed_fibroblasts",
    "bronchial_fibroblast",
    "fibroblast_of_dermis",
    "Penis_Foreskin_Fibroblast_Primary_Cells_skin02",
    "fibroblast_of_villous_mesenchyme",
    "NHLF",
    "fibroblast_of_lung",
    "myometrial_cell",
    "bronchial_smooth_muscle_cell",
    "aortic_smooth_muscle_cell",
    "fibroblast_of_arm",
    "smooth_muscle_cell_of_the_pulmonary_artery",
    "smooth_muscle_cell_of_bladder",
    "smooth_muscle_cell_of_trachea",
    "fibroblast_of_the_aortic_adventitia",
    "cardiac_atrium_fibroblast",
    "uterine_smooth_muscle_cell",
    "smooth_muscle_cell_of_the_umbilical_artery",
    "regular_cardiac_myocyte",
    "skeletal_muscle_satellite_cell",
    "HSMM",
    "cardiac_ventricle_fibroblast",
    "smooth_muscle_cell_of_the_coronary_artery",
    "skeletal_muscle_myoblast",
    "smooth_muscle_cell",
]

endothelial = [
    "pericyte_cell",
    "HUVEC",
    "mesangial_cell",
    "endothelial_cell_of_umbilical_vein",
    "endometrial_microvascular_endothelial_cell",
    "pulmonary_artery_endothelial_cell",
    "mammary_microvascular_endothelial_cell",
    "bladder_microvascular_endothelial_cell",
    "epithelial_cell_of_alveolus_of_lung",
    "vein_endothelial_cell",
    "dermis_blood_vessel_endothelial_cell",
    "thoracic_aorta_endothelial_cell",
    "lung_microvascular_endothelial_cell",
    "endothelial_cell_of_coronary_artery",
    "glomerular_endothelial_cell",
    "dermis_lymphatic_vessel_endothelial_cell",
    "dermis_microvascular_lymphatic_vessel_endothelial_cell",
]

set_0001 = [
    "Adipose_Subcutaneous",
    "Adipose_Visceral_Omentum",
    "Adrenal_Gland",
    "Artery_Aorta",
    "Artery_Coronary",
    "Artery_Tibial",
    "Bladder",
    "Brain_Amygdala",
    "Brain_Anterior_cingulate_cortex_BA24",
    "Brain_Caudate_basal_ganglia",
    "Brain_Cerebellar_Hemisphere",
    "Brain_Cerebellum_GTEx",
    "Brain_Cortex",
    "Brain_Frontal_Cortex_BA9",
    "Brain_Hippocampus",
    "Brain_Hypothalamus",
    "Brain_Nucleus_accumbens_basal_ganglia",
    "Brain_Putamen_basal_ganglia",
    "Brain_Spinal_cord_cervical_c1",
    "Brain_Substantia_nigra",
    "Breast_Mammary_Tissue",
    "Cells_EBV-transformed_lymphocytes",
    "Cells_Transformed_fibroblasts",
    "Cervix_Ectocervix",
    "Cervix_Endocervix",
    "Colon_Sigmoid",
    "Colon_Transverse",
    "Esophagus_Gastroesophageal_Junction",
    "Esophagus_Mucosa",
    "Esophagus_Muscularis",
    "Fallopian_Tube",
    "Heart_Atrial_Appendage",
    "Heart_Left_Ventricle",
    "Kidney_Cortex",
    "Liver_GTEx",
    "Lung_GTEx",
    "Minor_Salivary_Gland",
    "Muscle_Skeletal",
    "Nerve_Tibial",
    "Ovary_GTEx",
    "Pancreas_GTEx",
    "Pituitary",
    "Prostate",
    "Skin_Not_Sun_Exposed_Suprapubic",
    "Skin_Sun_Exposed_Lower_leg",
    "Small_Intestine_Terminal_Ileum",
    "Spleen_GTEx",
    "Stomach_GTEx",
    "Testis",
    "Thyroid_GTEx",
    "Uterus_GTEx",
    "Vagina",
    "Whole_Blood",
]


if args.filterStr == "pc":
    filt = np.asarray(geneanno.iloc[:, -1] == "protein_coding")
elif args.filterStr == "lincRNA":
    filt = np.asarray(geneanno.iloc[:, -1] == "lincRNA")
elif args.filterStr == "all":
    filt = np.asarray(geneanno.iloc[:, -1] != "rRNA")
else:
    raise ValueError("filterStr has to be one of all, pc, and lincRNA")

groups = hematopoietic + epithelial + stem + muscle_fib + endothelial

geneexp = pd.read_csv(args.expFile, usecols=groups)

# for cell in groups:
#     if cell in geneexp.columns:
#         print(cell, "True")
#     else:
#         print(cell, "False")
#         break


with open(args.modellistName, "wt") as fin:
    fin.write("ModelName\tTissue\n")


i = 1
for cell in groups:
    print("training: {}/{}".format(i, len(groups)))

    # pseudocount = args.pseudocount
    if cell in set_0001:
        pseudocount = 0.0001
    else:
        pseudocount = 0.01

    filt = filt * np.isfinite(np.asarray(np.log(geneexp[cell] + pseudocount)))

    # training

    trainind = (
        np.asarray(geneanno["seqnames"] != "chrX")
        * np.asarray(geneanno["seqnames"] != "chrY")
        * np.asarray(geneanno["seqnames"] != "chr8")
    )
    testind = np.asarray(geneanno["seqnames"] == "chr8")

    dtrain = xgb.DMatrix(Xreducedall[trainind * filt, :])
    dtest = xgb.DMatrix(Xreducedall[(testind) * filt, :])

    dtrain.set_label(
        np.asarray(np.log(geneexp.iloc[trainind * filt, :][cell] + pseudocount))
    )
    dtest.set_label(
        np.asarray(np.log(geneexp.iloc[(testind) * filt, :][cell] + pseudocount))
    )

    param = {
        "booster": "gblinear",
        "base_score": args.base_score,
        "alpha": 0,
        "lambda": args.l2,
        "eta": args.eta,
        "objective": "reg:linear",
        "nthread": args.threads,
        "early_stopping_rounds": 10,
    }

    evallist = [(dtest, "eval"), (dtrain, "train")]
    num_round = args.num_round
    bst = xgb.train(param, dtrain, num_round, evallist)
    ypred = bst.predict(dtest)
    if args.evalFile != "":
        evaldf = pd.DataFrame(
            {
                "pred": ypred,
                "target": np.asarray(
                    np.log(geneexp.iloc[(testind) * filt, :][cell] + pseudocount)
                ),
            }
        )
        evaldf.to_csv(args.evalFile)
    bst.save_model(
        args.output
        + args.filterStr
        + ".pseudocount"
        + str(pseudocount)
        + ".lambda"
        + str(args.l2)
        + ".round"
        + str(args.num_round)
        + ".basescore"
        + str(args.base_score)
        + "."
        + cell
        + ".save"
    )
    bst.dump_model(
        args.output
        + args.filterStr
        + ".pseudocount"
        + str(pseudocount)
        + ".lambda"
        + str(args.l2)
        + ".round"
        + str(args.num_round)
        + ".basescore"
        + str(args.base_score)
        + "."
        + cell
        + ".dump"
    )

    with open(args.modellistName, "at") as fin:
        fin.write(
            os.path.abspath(
                args.output
                + args.filterStr
                + ".pseudocount"
                + str(pseudocount)
                + ".lambda"
                + str(args.l2)
                + ".round"
                + str(args.num_round)
                + ".basescore"
                + str(args.base_score)
                + "."
                + cell
                + ".save"
            )
            + "\t"
            + cell
            + "\n"
        )
    i = i + 1
