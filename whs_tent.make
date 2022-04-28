CC = python
SHELL = bash
PP = PYTHONPATH="$(PYTHONPATH):."

.PHONY: all view plot report

CFLAGS = -O
#CFLAGS = -O -m pdb
#DEBUG = --debug

#the regex of the slices in the target dataset
#for the heart --
G_RGX = slice\d+_\d+

T_DATA = [('IMG', nii_transform, False), ('GT', nii_gt_transform, False), ('GT', nii_gt_transform, False)]
TT_DATA = [('IMG', nii_transform, False), ('GT', nii_gt_transform, False), ('GT', nii_gt_transform, False),('GT', nii_gt_transform, False)]
TTT_DATA = [('IMG', nii_transform, False), ('GT', nii_gt_transform, False), ('GT', nii_gt_transform, False),('GT', nii_gt_transform, False),('GT', nii_gt_transform, False)]
L_ENT = [('SelfEntropy', {'idc': [0,1,2,3,4], 'weights':[0.02, 0.27, 0.18, 0.21, 0.32]}, None, None, None, 1)]
L_Proposal = [('ProposalLoss', {'idc': [0,1,2,3,4], 'weights':[0.02, 0.27, 0.18, 0.21, 0.32]}, None, None, None, 1)]
L_Z = [('CrossEntropy', {'idc': [0,1,2,3,4], 'weights':[1,1,1,1,1]}, None, None, None, 0)]
T_DATA = [('IMG', nii_transform, False), ('GT', nii_gt_transform, False), ('GT', nii_gt_transform, False)]
TTTT_DATA = [('IMG', nii_transform, False), ('GT', nii_gt_transform, False), ('GT', nii_gt_transform, False), ('GT', nii_gt_transform, False), ('GT', nii_gt_transform, False), ('GT', nii_gt_transform, False)]
L_OR = [('CrossEntropy', {'idc': [0,1,2,3,4], 'weights':[0.02, 0.27, 0.18, 0.21, 0.32],'moment_fn':'soft_size'}, None, None, None, 1)]
LOSS=[('NaivePenalty', {'idc': [1,2,3,4]}, 'PreciseBounds', {'margin': 0.10, 'mode': 'percentage'}, 'soft_size', 1e-2), \
	('NaivePenalty', {'idc': [1,2,3,4]}, 'PreciseBounds', {'margin': 0.10, 'mode': 'abs'}, 'soft_centroid', 1e-2), \
	('NaivePenalty', {'idc': [1,2,3,4]}, 'PreciseBounds', {'margin': 0.10, 'mode': 'abs'}, 'soft_dist_centroid', 1e-2), \
	('NaivePenalty', {'idc': [1,2,3,4]}, 'PreciseBounds', {'margin': 0.10, 'mode': 'percentage'}, 'soft_length', 1e-2)] \

LTent = [('EntKLProp', {'moment_fn':'soft_size','lamb_se':1, 'lamb_consprior':0,'ivd':True,'weights_se':[0.02, 0.27, 0.18, 0.21, 0.32],'idc_c': [1,2,3,4],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1,2,3,4],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':';','sizefile':'sizes/whs.csv'},'norm_soft_size',1)]

LSIZElength = [('EntKLPropWMoment', {'moment_fn':'soft_length','lamb_se':1, 'lamb_moment':0.001,'lamb_consprior':1,'ivd':True,'weights_se':[0.02, 0.27, 0.18, 0.21, 0.32],'idc_c': [1,2,3,4],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1,2,3,4],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':';','sizefile':'sizes/whs.csv'},'norm_soft_size',1)]

LSizeInertia=[('EntKLPropWInertia', {'curi':True,'lamb_se':1,'lamb_inertia':0.1,'lamb_consprior':1, 'ivd':False,'weights_se':[0.02, 0.27, 0.18, 0.21, 0.32],'idc_c': [1,2,3,4],'power': 1},'PredictionBounds', \
        {'margin':0,'dir':'high','idc':[1],'predcol':'dumbpredwtags', 'power': 1,'fake':False, 'mode':'percentage','prop':False,'sep':';','sizefile':'sizes/whs.csv'},'norm_soft_size',50)]\

LSIZEnu = [('EntKLPropWMomentNu', {'ind_moment':[1,0,3],'lamb_se':1, 'lamb_moment':0.1,'lamb_consprior':1,'ivd':True,'weights_se':[0.02, 0.27, 0.18, 0.21, 0.32],'idc_c': [1,2,3,4],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1,2,3,4],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':';','sizefile':'sizes/whs.csv'},'norm_soft_size',1)]

LSIZE = [('EntKLProp', {'weights_se':[1,1,1,1,1],'moment_fn':'soft_size','lamb_se':1, 'lamb_consprior':1,'ivd':True,'idc_c': [1,2,3,4],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1,2,3,4],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':';','sizefile':'sizes/whs.csv'},'norm_soft_size',1)]

LSIZEQuadDistclassdistcen = [('EntKLPropWMoment', {'margin':0,'mom_est':[],'moment_fn':'class_dist_centroid','lamb_se':1, 'lamb_moment':0.01,'lamb_consprior':1,'ivd':True,'weights_se':[0.02, 0.27, 0.18, 0.21, 0.32],'idc_c': [1,2,3,4],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1,2,3,4],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':';','sizefile':'sizes/whs.csv'},'norm_soft_size',1),('EntKLPropWMoment', {'margin':0.2,'mom_est':[[74.72, 18.36, 13.79, 12.83, 7.47],[74.53, 21.23, 12.4, 14.23, 14.67]],'moment_fn':'soft_dist_centroid','lamb_se':1, 'lamb_moment':0.0001,'lamb_consprior':1,'ivd':True,'weights_se':[0.02, 0.27, 0.18, 0.21, 0.32],'idc_c': [1,2,3,4],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1,2,3,4],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':';','sizefile':'sizes/whs.csv'},'norm_soft_size',1)]

LSIZEecc = [('EntKLPropWMoment', {'rel_diff':False,'matrix':False,'linreg':False,'temp':1.0,'margin':1,'mom_est':[0.09, 0.84, 1.0, 1.0, 1.0],'moment_fn':'soft_eccentricity','lamb_se':1, 'lamb_moment':0.001,'lamb_consprior':1,'ivd':True,'weights_se':[0.02, 0.27, 0.18, 0.21, 0.32],'idc_c': [1,2,3,4],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1,2,3,4],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':';','sizefile':'sizes/whs.csv'},'norm_soft_size',1)]

LSIZECentroidQuadDist = [('EntKLPropWMoment', {'mom_est':[[127.5, 92.48, 136.82, 92.44, 139.91],[127.5, 104.13, 152.48, 107.08, 199.52]],'moment_fn':'soft_centroid','lamb_se':1, 'lamb_moment':0.001,'lamb_consprior':1,'ivd':True,'weights_se':[0.02, 0.27, 0.18, 0.21, 0.32],'idc_c': [1,2,3,4],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1,2,3,4],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':';','sizefile':'sizes/whs.csv'},'norm_soft_size',1),('EntKLPropWMoment', {'mom_est':[[74.72, 18.36, 13.79, 12.83, 7.47],[74.53, 21.23, 12.4, 14.23, 14.67]],'moment_fn':'soft_dist_centroid','lamb_se':1, 'lamb_moment':0.0001,'lamb_consprior':1,'ivd':True,'weights_se':[0.02, 0.27, 0.18, 0.21, 0.32],'idc_c': [1,2,3,4],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1,2,3,4],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':';','sizefile':'sizes/whs.csv'},'norm_soft_size',1)]

LSIZEQuadDistEcc = [('EntKLPropWMoment', {'mom_est':[('EntKLPropWMoment', {'margin':0,'mom_est':[0.09, 0.84, 1.0, 1.0, 1.0],'moment_fn':'soft_eccentricity','lamb_se':1, 'lamb_moment':0.1,'lamb_consprior':1,'ivd':True,'weights_se':[0.02, 0.27, 0.18, 0.21, 0.32],'idc_c': [1,2,3,4],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1,2,3,4],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':';','sizefile':'sizes/whs.csv'},'norm_soft_size',1),('EntKLPropWMoment', {'mom_est':[[74.72, 18.36, 13.79, 12.83, 7.47],[74.53, 21.23, 12.4, 14.23, 14.67]],'moment_fn':'soft_dist_centroid','lamb_se':1, 'lamb_moment':0.0001,'lamb_consprior':1,'ivd':True,'weights_se':[0.02, 0.27, 0.18, 0.21, 0.32],'idc_c': [1,2,3,4],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1,2,3,4],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':';','sizefile':'sizes/whs.csv'},'norm_soft_size',1)]

LSIZEinertia = [('EntKLPropWMoment', {'margin':0.5,'mom_est':[-0.0, 0.0, -0.01, -0.0, -0.03],'moment_fn':'soft_inertia','lamb_se':1, 'lamb_moment':0.1,'lamb_consprior':1,'ivd':True,'weights_se':[0.02, 0.27, 0.18, 0.21, 0.32],'idc_c': [1,2,3,4],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1,2,3,4],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':';','sizefile':'sizes/whs.csv'},'norm_soft_size',1)]

LSIZEclassdistcen = [('EntKLPropWMoment', {'margin':0,'mom_est':[],'moment_fn':'class_dist_centroid','lamb_se':1, 'lamb_moment':0.01,'lamb_consprior':1,'ivd':True,'weights_se':[0.02, 0.27, 0.18, 0.21, 0.32],'idc_c': [1,2,3,4],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1,2,3,4],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':';','sizefile':'sizesource/whs.csv'},'norm_soft_size',1)]

LSIZEaltcomp = [('EntKLPropWMoment', {'matrix':False,'linreg':False,'temp':1.00,'margin':0.2,'mom_est':[1.24, 39.41, 10.95, 10.45, 12.99],'moment_fn':'saml_compactness','lamb_se':1, 'lamb_moment':0.001,'lamb_consprior':1,'ivd':True,'weights_se':[0.02, 0.27, 0.18, 0.21, 0.32],'idc_c': [1,2,3,4],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1,2,3,4],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':';','sizefile':'sizes/whs.csv'},'norm_soft_size',1)]

LSIZElen = [('EntKLPropWMoment', {'margin':0.5,'mom_est':[618.0, 922.0, 522.0, 532.0, 434.0],'moment_fn':'soft_length','lamb_se':1, 'lamb_moment':0.1,'lamb_consprior':1,'ivd':True,'weights_se':[0.02, 0.27, 0.18, 0.21, 0.32],'idc_c': [1,2,3,4],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1,2,3,4],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':';','sizefile':'sizes/whs.csv'},'norm_soft_size',1)]


LSIZEclassdistcen = [('EntKLPropWMoment', {'temp':1.1,'margin':0,'mom_est':[],'moment_fn':'class_dist_centroid','lamb_se':1, 'lamb_moment':0.01,'lamb_consprior':1,'ivd':True,'weights_se':[0.02, 0.27, 0.18, 0.21, 0.32],'idc_c': [1,2,3,4],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1,2,3,4],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':';','sizefile':'sizesource/whs.csv'},'norm_soft_size',1)]

LSIZENu = [('EntKLPropWMoment', {'linreg':False,'temp':1.01,'margin':0.1,'mom_est':[[0.3, 0.43, 0.31, 0.28, 0.24],[0.3, 0.5, 0.29, 0.3, 0.46]],'moment_fn':'soft_nu','lamb_se':1, 'lamb_moment':0.0001,'lamb_consprior':1,'ivd':True,'weights_se':[0.02, 0.27, 0.18, 0.21, 0.32],'idc_c': [1,2,3,4],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1,2,3,4],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':';','sizefile':'sizes/whs.csv'},'norm_soft_size',1)]

#linreg on source reg/dist_centroid.txt reg/dist_centroid_vertical.txt
# 'lamb_moment':0.0001,
LSIZEQuadDist = [('EntKLPropWMoment', {'lamb_se':1,'lamb_moment':0.0001,'margin':0.1,'rel_diff':False,'matrix':False,'linreg':False,'temp':1.01,'reg':'reg/tentwhs_cenklsizedistav.txt','reg2':'reg/tentwhs_cenklsizedistav.txt','mom_est':[[74.72, 18.36, 13.79, 12.83, 7.47],[74.53, 21.23, 12.4, 14.23, 14.67]],'moment_fn':'soft_dist_centroid', 'lamb_consprior':1,'ivd':True,'weights_se':[0.02, 0.27, 0.18, 0.21, 0.32],'idc_c': [1,2,3,4],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1,2,3,4],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':';','sizefile':'sizes/whs.csv'},'norm_soft_size',1)]

LSIZECentroid = [('EntKLPropWMoment', {'temp':1.01,'weights_se':[0.02, 0.27, 0.18, 0.21, 0.32],'lamb_se':1,'lamb_moment':1,'margin':0.1,'rel_diff':False,'matrix':False,'linreg':False,'reg':'reg/tentwhs_cenklsizecentav.txt','reg2':'reg/tentwhs_cenklsizecentav.txt','mom_est':[[127.5, 92.48, 136.82, 92.44, 139.91],[127.5, 104.13, 152.48, 107.08, 199.52]],'moment_fn':'soft_centroid','lamb_consprior':1,'ivd':True,'idc_c': [1,2,3,4],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1,2,3,4],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':';','sizefile':'sizes/whs.csv'},'norm_soft_size',1)]

LSIZECentroidDist = [('EntKLPropWMoment', {'lamb_moment':0.001,'margin':0.2,'rel_diff':False,'matrix':False,'linreg':False,'temp':1.01,'reg':'reg/tentwhs_cenklsizecentav.txt','reg2':'reg/tentwhs_cenklsizecentav.txt','mom_est':[[127.5, 92.48, 136.82, 92.44, 139.91],[127.5, 104.13, 152.48, 107.08, 199.52]],'moment_fn':'soft_centroid','lamb_se':1,'lamb_consprior':1,'ivd':True,'weights_se':[0.02, 0.27, 0.18, 0.21, 0.32],'idc_c': [1,2,3,4],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1,2,3,4],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':';','sizefile':'sizes/whs.csv'},'norm_soft_size',1),('EntKLPropWMoment', {'lamb_moment':0.0001,'margin':0.2,'rel_diff':False,'matrix':False,'linreg':False,'temp':1.01,'reg':'reg/tentwhs_cenklsizedistav.txt','reg2':'reg/tentwhs_cenklsizedistav.txt','mom_est':[[74.72, 18.36, 13.79, 12.83, 7.47],[74.53, 21.23, 12.4, 14.23, 14.67]],'moment_fn':'soft_dist_centroid', 'lamb_se':1,'lamb_consprior':1,'ivd':True,'weights_se':[0.02, 0.27, 0.18, 0.21, 0.32],'idc_c': [1,2,3,4],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1,2,3,4],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':';','sizefile':'sizes/whs.csv'},'norm_soft_size',1)]

LSIZEcomp = [('EntKLPropWMoment', {'linreg':False,'temp':1.00,'margin':0.5,'mom_est':[0, 0, 0, 0, 0],'moment_fn':'saml_compactness','lamb_se':1, 'lamb_moment':0.001,'lamb_consprior':1,'ivd':True,'weights_se':[0.02, 0.27, 0.18, 0.21, 0.32],'idc_c': [1,2,3,4],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1,2,3,4],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':';','sizefile':'sizes/whs.csv'},'norm_soft_size',1)]


NET = UNet

# the folder containing the target dataset
T_FOLD = /data/users/mathilde/ccnn/CDA/data/ct_nii

#the network weights used as initialization of the adaptation
M_WEIGHTS_ul = results/whs/cesource/last.pkl
M_WEIGHTS_ul = oneresults/tentwhs/enklsize3/
#M_WEIGHTS_ul = oneresults/tentwhs/enklsize2/

EPC=150
#run the main experiment
TRN = results/whs/tent_selfent$(EPC)bis
TRN = results/tentwhs/selfent
TRN = oneresults/tentwhs/enklsizebis
#TRN = oneresults/tentwhs/quadist_centroidbis
#TRN = oneresultsslice/tentwhs/enklsize
TRN = oneresults/tentwhs/enquadinertia
TRN = oneresults/tentwhs/proposalquadinertia2
TRN = oneresults/tentwhs/enklsizequadinertia4
TRN = oneresults/tentwhs/debug
TRN = oneresults/tentwhs/proposalquadinertiaenklsize
TRN = oneresults/tentwhs/enklsizequadinertia4
TRN = oneresults/tentwhs/cenklsizequad2mom oneresults/tentwhs/cenklsizequadcentroid5
TRN = oneresults/tentwhs/cenklsizequadinertia
TRN = oneresults/tentwhs/cenklsizequadlen
#TRN = oneresults/whs/enklsize
TRN = oneresults/tentwhs/cenklsizequadecc6
TRN = oneresults/tentwhs/cenklsizequadcomp5
TRN = oneresults/tentwhs/cenklsizeclassdiscen8
TRN = oneresults/tentwhsnb/cenklsizedistavdivnomar2
#TRN = oneresults/tentwhsnb/cenklsizedistavdivnomar
TRN = oneresults/tentwhs/cenklsizedistavbis
TRN = oneresults/tentwhs/cenklsizedistavim
TRN = oneresults/tentwhs/cenklsizedistavf
#TRN = oneresults/tentwhs/cenklsizedistavwidethresh
#TRN = oneresults/tentwhsnb/cklsizecentav2
#TRN = oneresults/tentwhs/cenklsizedistav
#TRN = oneresults/tentwhs/2mom
#TRN = results/whs/cesourceim
#TRN = oneresults/tentwhs/enklsizecentlr
#TRN = oneresults/tentwhs/cenklsizedistmed
#TRN = oneresults/tentwhs/cenklsizequadcompzero
#TRN = oneresults/tentwhs/tentw
#TRN = oneresults/tentwhs/fs

#oneresults/tentwhs/enklsize3
#TRN = oneresults/tentwhs/enklsizeinertia2
#TRN = oneresults/tentwhs/qua_distcentroid2
#TRN = oneresults/tentwhs/enklsizeecc

REPO = $(shell basename `git rev-parse --show-toplevel`)
DATE = $(shell date +"%y%m%d")
HASH = $(shell git rev-parse --short HEAD)
HOSTNAME = $(shell hostname)
PBASE = archives
PACK = $(PBASE)/$(REPO)-$(DATE)-$(HASH)-$(HOSTNAME)-CSize.tar.gz

all: pack

plot: $(PLT)

pack: $(PACK) report
$(PACK): $(TRN) $(INF_0) $(TRN_1) $(INF_1) $(TRN_2) $(TRN_3) $(TRN_4)
	mkdir -p $(@D)
	tar cf - $^ | pigz > $@
	chmod -w $@
# tar -zc -f $@ $^  # Use if pigz is not available

# first train on the source dataset only:
results/whs/cesource: OPT =  --target_losses="$(L_OR)" --target_dataset "data/mr" \
	     --network UNet --model_weights="" --lr_decay 1 \

results/whs/cesourceim: OPT =  --mode makeim --target_losses="$(L_OR)" --n_epoch 1 --saveim True --global_model --model_weights="results/whs/cesource/last.pkl" \

# full supervision
results/whs/tent_fs: OPT =  --target_losses="$(L_OR)" \

oneresults/tentwhs/zero: OPT =  --target_losses="$(L_Z)" --l_rate 0 \

results/tentwhs/selfent: OPT =  --target_losses="$(L_ENT)" \

oneresults/tentwhs/4desc: OPT =  --val_target_folders="$(TTTT_DATA)" --target_folders="$(TTTT_DATA)" --target_losses="$(LOSS)" \

oneresults/tentwhs/debug: OPT = --batch_size 1 --val_target_folders="$(TT_DATA)" --target_folders="$(TT_DATA)"  --target_losses="$(L_ENT)"+"[('NaivePenalty', {'idc': [1,2,3,4]}, 'PreciseBounds', {'margin': 0, 'mode': 'percentage'}, 'soft_inertia_moment', 1e-2)]"\

results/tentwhs/enklcomp: OPT =  --target_losses="[('EntKLPropWCompLen', {'curi':True,'lamb_se':1,'lamb_comp':100,'lamb_consprior':0, 'lamb_len':0,'ivd':False,'weights_se':[0.02, 0.27, 0.18, 0.21, 0.32],'idc_c': [1,2,3,4],'power': 1},'PredictionBounds', \
        {'margin':0,'dir':'high','idc':[1],'predcol':'dumbpredwtags', 'power': 1,'fake':False, 'mode':'percentage','prop':False,'sep':';','sizefile':'sizes/whs.csv'},'norm_soft_size',50)]" \

results/tentwhs/enklsizecomplen: OPT =  --target_losses="[('EntKLPropWCompLen', {'curi':True,'lamb_se':1,'lamb_comp':1,'lamb_consprior':1, 'lamb_len':1,'ivd':False,'weights_se':[0.02, 0.27, 0.18, 0.21, 0.32],'idc_c': [1,2,3,4],'power': 1},'PredictionBounds', \
        {'margin':0,'dir':'high','idc':[1],'predcol':'dumbpredwtags', 'power': 1,'fake':False, 'mode':'percentage','prop':False,'sep':';','sizefile':'sizes/whs.csv'},'norm_soft_size',50)]" \

results/tentwhs/enklsizecomp: OPT =  --target_losses="[('EntKLPropWComp', {'curi':True,'lamb_se':1,'lamb_comp':1,'lamb_consprior':1, 'ivd':False,'weights_se':[0.02, 0.27, 0.18, 0.21, 0.32],'idc_c': [1,2,3,4],'power': 1},'PredictionBounds', \
        {'margin':0,'dir':'high','idc':[1],'predcol':'dumbpredwtags', 'power': 1,'fake':False, 'mode':'percentage','prop':False,'sep':';','sizefile':'sizes/whs.csv'},'norm_soft_size',50)]"\

oneresults/tentwhs/enklsizequalen: OPT =  --val_target_folders="$(TT_DATA)" --target_folders="$(TT_DATA)" --target_losses="[('EntKLProp', {'curi':True,'lamb_se':1,'lamb_consprior':1, 'ivd':False,'weights_se':[0.02, 0.27, 0.18, 0.21, 0.32],'idc_c': [1,2,3,4],'power': 1},'PredictionBounds', \
        {'margin':0,'dir':'high','idc':[1],'predcol':'dumbpredwtags', 'power': 1,'fake':False, 'mode':'percentage','prop':False,'sep':';','sizefile':'sizes/whs.csv'},'norm_soft_size',50),('NaivePenalty', {'idc': [1,2,3,4]}, 'PreciseBounds', {'margin': 0.10, 'mode': 'percentage'}, 'soft_length', 1e-2)]"\
oneresults/tentwhs/qualen: OPT =  --val_target_folders= --target_losses="[('NaivePenalty', {'idc': [1,2,3,4]}, 'PreciseBounds', {'margin': 0.10, 'mode': 'percentage'}, 'soft_length', 1e-2)]"\
oneresults/tentwhs/qua_distcentroid2: OPT = --target_losses="[('NaivePenalty', {'idc': [1,2,3,4]}, 'PreciseBounds', {'margin': 0, 'mode': 'percentage'}, 'soft_dist_centroid', 1e-2)]"\
oneresults/tentwhs/enquadinertia: OPT = --val_target_folders="$(TT_DATA)" --target_folders="$(TT_DATA)"  --target_losses="$(L_ENT)"+"[('NaivePenalty', {'idc': [1,2,3,4]}, 'PreciseBounds', {'margin': 0, 'mode': 'percentage'}, 'soft_inertia_moment', 1e-2)]"\
oneresults/tentwhs/proposalquadinertia2: OPT =  --val_target_folders="$(TT_DATA)" --target_folders="$(TT_DATA)"  --target_losses="$(L_Proposal)"+"[('NaivePenalty', {'idc': [1,2,3,4]}, 'PreciseBounds', {'margin': 0, 'mode': 'percentage'}, 'soft_inertia_moment', 1e-3)]"\
oneresults/tentwhs/proposalquadinertiaenklsize: OPT =  --val_target_folders="$(TTT_DATA)" --target_folders="$(TTT_DATA)"  --target_losses="$(LSIZE)"+"$(L_Proposal)"+"[('NaivePenalty', {'idc': [1,2,3,4]}, 'PreciseBounds', {'margin': 0, 'mode': 'percentage'}, 'soft_inertia_moment', 1e-3)]"\
oneresults/tentwhs/enklsizequadinertia4: OPT = --val_target_folders="$(TT_DATA)" --target_folders="$(TT_DATA)"  --target_losses="$(LSIZE)"+"[('NaivePenalty', {'idc': [1,2,3,4]}, 'PreciseBounds', {'margin': 0, 'mode': 'percentage'}, 'soft_inertia_moment', 0)]"\
oneresults/tentwhs/enklsizequadinertia5: OPT = --val_target_folders=  --target_losses="$(LSIZE)"\
oneresults/tentwhs/enalphaklsize: OPT = --target_losses="[('AlphaEntKLProp', {'curi':True,'lamb_se':1,'lamb_consprior':1, 'ivd':False,'weights_se':[0.02, 0.27, 0.18, 0.21, 0.32],'idc_c': [1,2,3,4],'power': 1},'PredictionBounds', \
        {'margin':0,'dir':'high','idc':[1],'predcol':'dumbpredwtags', 'power': 2,'fake':False, 'mode':'percentage','prop':False,'sep':';','sizefile':'sizes/whs.csv'},'norm_soft_size',50)]"\

oneresults/tentwhs/cenklsizedistavim: OPT = --model_weights oneresults/tentwhs/cenklsizedistav --batch_size 18 --pprint --n_epoch 1 --saveim True --do_asd 1 --target_losses="$(LSIZE)" --l_rate 0 \

oneresults/tentwhs/cenklsizecentavim: OPT = --model_weights oneresults/tentwhs/cenklsizecentav --batch_size 18 --pprint --n_epoch 1 --saveim True --do_asd 1 --target_losses="$(LSIZE)" --l_rate 0 \

oneresults/tentwhs/tentw: OPT = --saveim True --do_asd 1 --target_losses="$(LTent)" --global_model --model_weights results/whs/cesource/last.pkl \

oneresults/tentwhs/ennowklsize: OPT = --global_model --batch_size 32  --l_rate 5e-4 --model_weights results/whs/cesource/last.pkl --target_losses="$(LSIZE)"\

oneresultsslice/tentwhs/enklsize: OPT = --oneslice --target_losses="$(LSIZE)"\

oneresults/tentwhs/enklsizequaddist: OPT =  --target_losses="$(LSIZEQuadDist)" --saveim True\

oneresults/tentwhs/cenklsizequadlen: OPT =  --target_losses="$(LSIZElen)" --saveim True\


oneresults/tentwhs/cenklsizeclassdiscen3: OPT =  --target_losses="$(LSIZEclassdistcen)" --saveim True --l_rate 2.5e-4 --n_epoch 150 --pprint \

oneresults/tentwhs/cenklsizequadinertia: OPT =  --target_losses="$(LSIZEinertia)" --saveim True \

oneresults/tentwhs/cenklsizequad2mom: OPT = --val_target_folders="$(TT_DATA)" --target_folders="$(TT_DATA)" --ind_mom 1 --target_losses="$(LSIZECentroidQuadDist)" --saveim True  --val_target_folders="$(TT_DATA)" --target_folders="$(TT_DATA)" \

oneresults/tentwhs/QuadDistclassdistcen: OPT = --target_losses="$(LSIZEQuadDistclassdistcen)" --saveim True  --val_target_folders="$(TT_DATA)" --target_folders="$(TT_DATA)" \

# to keep
oneresults/tentwhs/cenklsizequaddist6oct: OPT = --ind_mom 1 --target_losses="$(LSIZEQuadDist)" --n_epoch 150 --softmax_temp 0.5 \

oneresults/tentwhs/cenklsizeNu: OPT = --ind_mom 1 --target_losses="$(LSIZENu)" --saveim True --n_epoch 150 --softmax_temp 0.5 \

oneresults/tentwhs/cenklsizequadecc6: OPT =  --target_losses="$(LSIZEecc)" --saveim True \

oneresults/tentwhs/cenklsizequadcompzero: OPT = --softmax_temp 0.5 --target_losses="$(LSIZEcomp)" --saveim True  \

oneresults/tentwhs/fs: OPT = --softmax_temp 0.5 --target_losses="$(L_OR)" --saveim True --do_asd 1  \

oneresults/tentwhs/cenklsizeclassdiscen8: OPT = --l_rate 2.5e-4 --softmax_temp 0.5 --target_losses="$(LSIZEclassdistcen)" --do_asd 1 --saveim True \

oneresults/tentwhs/cenklsizecentavquin: OPT = --do_asd 1 --pprint --saveim True --batch_size 18 --n_epoch 150  --update_mom_est --ind_mom 1  --target_losses="$(LSIZECentroid)" \

oneresults/tentwhs/cenklsizecentf15: OPT = --adw --batch_size 22 --softmax_temp 0.5 --n_epoch 200  --update_mom_est --ind_mom 1  --target_losses="$(LSIZECentroid)" \

oneresults/tentwhs/cenklsizedistadw3: OPT = --adw  --batch_size 22 --n_epoch 200  --update_mom_est --ind_mom 1  --target_losses="$(LSIZEQuadDist)" \

oneresults/tentwhs/cenklsizedistavf: OPT =  --batch_size 42 --n_epoch 200  --update_mom_est --ind_mom 1  --target_losses="$(LSIZEQuadDist)" \

oneresults/tentwhs/cenklsizeecc3: OPT = --n_epoch 150   --target_losses="$(LSIZEecc)" \

oneresults/tentwhsnb/cklsizedistav2: OPT =  --model_weights results/whs/cesource/last.pkl --global_model  --n_warmup 200 --n_epoch 400 --l_rate 5e-4  --update_mom_est --ind_mom 1  --target_losses="$(LSIZEQuadDist)" \

oneresults/tentwhsnb/cklsizecentav: OPT =  --model_weights results/whs/cesource/last.pkl --global_model  --n_warmup 200 --n_epoch 400 --l_rate 5e-4  --update_mom_est --ind_mom 1  --target_losses="$(LSIZECentroid)" \

oneresults/tentwhs/2mom: OPT = --val_target_folders="$(TT_DATA)" --target_folders="$(TT_DATA)" --batch_size 18 --n_epoch 150  --update_mom_est --ind_mom 1  --target_losses="$(LSIZECentroidDist)" \

oneresults/tentwhsnb/cenklsizecentavdivnomar: OPT = --batch_size 28 --update_mom_est --ind_mom 1 --target_losses="$(LSIZECentroid)" --saveim True \

oneresults/tentwhs/enklsizedistlr: OPT = --update_lin_reg --softmax_temp 0.5 --ind_mom 1 --target_losses="$(LSIZEQuadDist)" --saveim True \

oneresults/tentwhs/cenklsizeeccmed: OPT =  --update_mom_est --softmax_temp 0.5 --target_losses="$(LSIZEecc)" --saveim True \

oneresults/tentwhs/cenklsizedistav: OPT =  --pprint --do_asd 1 --saveim True --ind_mom 1  --n_epoch 150 --update_mom_est  --target_losses="$(LSIZEQuadDist)" \

oneresults/tentwhs/cenklsizedistf: OPT =   --ind_mom 1  --n_epoch 200 --update_mom_est  --target_losses="$(LSIZEQuadDist)" \

oneresults/tentwhs/cenklsizedistf3: OPT = --ind_mom 1  --n_epoch 200 --update_mom_est  --target_losses="$(LSIZEQuadDist)" \

oneresults/tentwhswa/cenklsizedistav: OPT =  --batch_size 18 --l_rate 1e-4  --model_weights oneresults/tentwhs/enklsize2 --n_warmup 50 --ind_mom 1 --n_epoch 250 --update_mom_est  --target_losses="$(LSIZEQuadDist)" \
oneresults/tentwhswa/cenklsizecentav: OPT =  --l_rate 1e-4 --model_weights oneresults/tentwhs/enklsize2 --n_warmup 50 --ind_mom 1 --n_epoch 250 --update_mom_est  --target_losses="$(LSIZECentroid)" \
oneresults/tentwhsnb/cenklsizedistavdivnomar2: OPT =  --batch_size 28 --n_epoch 250 --update_mom_est --target_losses="$(LSIZEQuadDist)" --saveim True \


# SFDA. Remove --saveim True --entmap --do_asd 1 --do_hd 1 to speed up
results/whs/sfda: OPT = --target_losses="[('EntKLProp', {'curi':True,'lamb_se':1,'lamb_consprior':1, 'ivd':False,'weights_se':[0.02, 0.27, 0.18, 0.21, 0.32],'idc_c': [1,2,3,4],'power': 1},'PredictionBounds', \
        {'margin':0,'dir':'high','idc':[1],'predcol':'dumbpredwtags', 'power': 1,'fake':False, 'mode':'percentage','prop':False,'sep':';','sizefile':'sizes/whs.csv'},'norm_soft_size',50)]"\
          --ontest --l_rate 0.000001 --lr_decay 0.7 --weight_decay 1e-3 \
          --saveim True --entmap --do_asd 1 --do_hd 1 \

#inference mode : saves the segmentation masks for a specific model saved as pkl file (ex. "results/whs/cesource/last.pkl" below):
results/whs/tent_selfent$(EPC)bis: OPT =  --target_losses="$(L_ENT)" \
	     --batch_size 8 --do_asd 1 --pprint --n_epoch=$(EPC) --l_rate 1e-5 --testonly  --ontest --lr_decay 1 --notent \

$(TRN) :
	$(CC) $(CFLAGS) main_sfda_tent2.py --regex_list "['ctslice81003','ctslice81008','ctslice81014','ctslice81019']" --testonly  --ontest --notent --batch_size 22 --n_class 5 --workdir $@_tmp --target_dataset  "$(T_FOLD)" \
                --train_grp_regex="$(G_RGX)" --metric_axis 1 2 3 4 --n_epoch 150 --dice_3d --l_rate 1e-4 --weight_decay 1e-4 --grp_regex="$(G_RGX)" --network=$(NET) \
                  --update_mom_est --lr_decay 0.9  --model_weights="$(M_WEIGHTS_ul)" --val_target_folders="$(T_DATA)" --target_folders="$(T_DATA)" $(OPT) $(DEBUG)
	mv $@_tmp $@
	$(CC) get_learning_stats.py $@ whs


