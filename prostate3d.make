CC = python
SHELL = bash
PP = PYTHONPATH="$(PYTHONPATH):."

.PHONY: all view plot report

CFLAGS = -O
#CFLAGS = -O -m pdb
#DEBUG = --debug

#the regex of the slices in the target dataset
#for the prostate
RGX = Case
G_RGX = Case\d+_\d+
#RGL= ['Case06','Case28','Case29','Case14','Case16','Case18','Case19','Case21','Case23','Case24','Case25','Case27','Case00','Case01','Case03','Case04','Case09','Case10','Case11','Case13']
RGL= ['Case28','Case29','Case14','Case16','Case18','Case19','Case21','Case23','Case24','Case25','Case27','Case00','Case01','Case03','Case04','Case09','Case10','Case11','Case13']
S_DATA = [('IMG', nii_transform, False), ('GT', nii_gt_transform, False), ('GT', nii_gt_transform, False)]
SAUG_DATA = [('IMGaug', nii_transform, False), ('GTaug', nii_gt_transform, False), ('GTaug', nii_gt_transform, False)]
SAUG_DATA_NORM = [('IMGaug', nii_transform_normalize, False), ('GTaug', nii_gt_transform, False), ('GTaug', nii_gt_transform, False)]
T_DATA = [('IMG', nii_transform, False), ('GTNew', nii_gt_transform, False), ('GTNew', nii_gt_transform, False)]
T_DATA3d = [('IMG', nii_transform_3d, False), ('GT', nii_gt_transform_3d, False), ('GT', nii_gt_transform_3d, False)]
T_DATANORM = [('IMG', nii_transform_normalize, False), ('GTNew', nii_gt_transform, False), ('GTNew', nii_gt_transform, False)]
TT_DATA = [('IMG', nii_transform, False), ('GTNew', nii_gt_transform, False), ('GTNew', nii_gt_transform, False),('GTNew', nii_gt_transform, False)]
L_OR = [('CrossEntropy', {'idc': [0,1], 'weights':[0.1,0.9],'moment_fn':'soft_size',}, None, None, None, 1),]
NET = UNet

LTent = [('EntKLProp', {'moment_fn':'soft_size','lamb_se':1, 'lamb_consprior':0,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/prostate.csv'},'norm_soft_size',1)]

LSIZE = [('EntKLProp', {'moment_fn':'soft_size','lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/prostate.csv'},'norm_soft_size',1)]

LSIZENu2 = [('EntKLPropWMoment', {'temp':1.01,'linreg':False,'margin':0.1,'mom_est':[[0.3, 0.31],[0.3, 0.27]],'moment_fn':'soft_nu','lamb_se':1, 'lamb_moment':0.0001,'lamb_consprior':1,'ivd':True,'weights_se':[1,1],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/prostate.csv'},'norm_soft_size',1)]
LSIZElength = [('EntKLPropWMoment', {'moment_fn':'soft_length','lamb_se':1, 'lamb_moment':0.001,'lamb_consprior':1,'ivd':True,'weights_se':[1,1],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/prostate.csv'},'norm_soft_size',1)]
LSIZElen = [('EntKLPropWMoment', {'margin':0.5,'mom_est':[718.0, 838.0],'moment_fn':'soft_length','lamb_se':1, 'lamb_moment':0.001,'lamb_consprior':1,'ivd':True,'weights_se':[1,1],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/prostate.csv'},'norm_soft_size',1)]
LSIZEnu = [('EntKLPropWMomentNu', {,'moment_fn':'soft_centroid','ind_moment':[1,2,1],'lamb_se':1, 'lamb_moment':0.1,'lamb_consprior':1,'ivd':True,'weights_se':[1,1],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/prostate.csv'},'norm_soft_size',1)]

LSIZEQuadDist = [('EntKLPropWMoment', {'linreg':False,'temp':1.01,'margin':0.1,'reg':'reg/prostate_dist_centroid.txt','reg2':'reg/prostate_dist_centroid_vertical.txt','mom_est':[[112.33, 21.51],[112.17, 18.08]],'moment_fn':'soft_dist_centroid','lamb_se':1, 'lamb_moment':0.0001,'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/prostate.csv'},'norm_soft_size',1)]

# linreg on source : reg/prostate_dist_centroid.txt reg/prostate_dist_centroid_vertical.txt
#linreg on target initialized w/ reg/tentprostate_cenklsizedistav.txt
# 0.0001
LSIZEQuadDist = [('EntKLPropWMoment', {'lamb_moment':0.0001,'matrix':False, 'rel_diff':False,'linreg':False,'temp':1.01,'margin':0.1,'reg':'reg/tentprostate_cenklsizedistav.txt','reg2':'reg/tentprostate_cenklsizedistav.txt','mom_est':[[112.33, 21.51],[112.17, 18.08]],'moment_fn':'soft_dist_centroid','lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/prostate.csv'},'norm_soft_size',1)]

#0.00001
LSIZECentroid = [('EntKLPropWMoment', {'weights_se':[0.1,0.9],'lamb_se':1,'lamb_moment':0.0001,'rel_diff':False, 'matrix':False, 'linreg':False,'temp':1.01,'margin':0,'reg':'reg/tentprostate_cenklsizecentav.txt','reg2':'reg/tentprostate_cenklsizecentav.txt','mom_est':[[191.52, 192.12],[191.55, 188.53]],'moment_fn':'soft_centroid', 'lamb_consprior':1,'ivd':True,'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/prostate.csv'},'norm_soft_size',1)]

LSIZECentroid3d= [('EntKLPropWMoment2', {'weights_se':[0.1,0.9],'lamb_se':1,'lamb_moment':0,'rel_diff':False, 'matrix':False, 'linreg':False,'temp':1.01,'margin':0,'reg':'reg/tentprostate_cenklsizecentav.txt','reg2':'reg/tentprostate_cenklsizecentav.txt','mom_est':[[23, 191.52, 188.53]],'moment_fn':'soft_centroid_3d', 'lamb_consprior':1,'ivd':True,'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/prostate.csv'},'norm_soft_size',1)]

LSIZECentroid3dGT= [('EntKLPropWMoment2', {'weights_se':[0.1,0.9],'lamb_se':1,'lamb_moment':0.0001,'rel_diff':False, 'matrix':False, 'linreg':False,'temp':1.01,'margin':0,'reg':'reg/tentprostate_cenklsizecentav.txt','reg2':'reg/tentprostate_cenklsizecentav.txt','mom_est':[[23, 191.52, 188.53]],'moment_fn':'soft_centroid_3d', 'lamb_consprior':1e4,'ivd':True,'idc_c': [1],'curi':False,'power': 1},'PreciseBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/prostate.csv'},'norm_soft_size_3d',1)]

LSIZEDist3dGT= [('EntKLPropWMoment2', {'weights_se':[0.1,0.9],'lamb_se':1,'lamb_moment':0.001,'rel_diff':False, 'matrix':False, 'linreg':False,'temp':1.01,'margin':0,'reg':'reg/tentprostate_cenklsizecentav.txt','reg2':'reg/tentprostate_cenklsizecentav.txt','mom_est':[[23, 191.52, 188.53]],'moment_fn':'soft_dist_centroid_3d', 'lamb_consprior':1e4,'ivd':True,'idc_c': [1],'curi':False,'power': 1},'PreciseBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/prostate.csv'},'norm_soft_size_3d',1)]

LCentroid3dGT= [('EntKLPropWMoment2', {'weights_se':[0.1,0.9],'lamb_se':1,'lamb_moment':0.001,'rel_diff':False, 'matrix':False, 'linreg':False,'temp':1.01,'margin':0,'reg':'reg/tentprostate_cenklsizecentav.txt','reg2':'reg/tentprostate_cenklsizecentav.txt','mom_est':[[23, 191.52, 188.53]],'moment_fn':'soft_centroid_3d', 'lamb_consprior':1e3,'ivd':True,'idc_c': [1],'curi':False,'power': 1},'PreciseBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/prostate.csv'},'norm_soft_size_3d',1)]

LSIZE3dGT= [('EntKLPropWMoment2', {'weights_se':[0.1,0.9],'lamb_se':1,'lamb_moment':0,'rel_diff':False, 'matrix':False, 'linreg':False,'temp':1.01,'margin':0,'reg':'reg/tentprostate_cenklsizecentav.txt','reg2':'reg/tentprostate_cenklsizecentav.txt','mom_est':[[23, 191.52, 188.53]],'moment_fn':'soft_centroid_3d', 'lamb_consprior':1e4,'ivd':True,'idc_c': [1],'curi':False,'power': 5},'PreciseBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags','power': 5, 'mode':'percentage','sep':',','sizefile':'sizes/prostate.csv'},'norm_soft_size_3d',1)]

LSelfEnt= [('EntKLPropWMoment2', {'weights_se':[0.1,0.9],'lamb_se':1,'lamb_moment':0,'rel_diff':False, 'matrix':False, 'linreg':False,'temp':1.01,'margin':0,'reg':'reg/tentprostate_cenklsizecentav.txt','reg2':'reg/tentprostate_cenklsizecentav.txt','mom_est':[[23, 191.52, 188.53]],'moment_fn':'soft_centroid_3d', 'lamb_consprior':0,'ivd':True,'idc_c': [1],'curi':False,'power': 1},'PreciseBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/prostate.csv'},'norm_soft_size_3d',1)]


LSIZECentroidQuadDist = [('EntKLPropWMoment', {'lamb_moment':0.0001,'matrix':False, 'rel_diff':False,'linreg':False,'temp':1.01,'margin':0.1,'reg':'reg/tentprostate_cenklsizedistav.txt','reg2':'reg/tentprostate_cenklsizedistav.txt','mom_est':[[112.33, 21.51],[112.17, 18.08]],'moment_fn':'soft_dist_centroid','lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/prostate.csv'},'norm_soft_size',1),('EntKLPropWMoment', {'lamb_moment':0.001,'rel_diff':True, 'matrix':False, 'linreg':False,'temp':1.01,'margin':0.1,'reg':'reg/tentprostate_cenklsizecentav.txt','reg2':'reg/tentprostate_cenklsizecentav.txt','mom_est':[[191.52, 192.12],[191.55, 188.53]],'moment_fn':'soft_centroid','lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/prostate.csv'},'norm_soft_size',1)]

LSIZEecc = [('EntKLPropWMoment', {'weights_se':[0.1,0.9],'rel_diff':False,'temp':1,'linreg':False,'margin':0,'reg':'reg/prostate_ecc.txt','reg2':'reg/prostate_ecc.txt','mom_est':[0.02, 0.66],'moment_fn':'soft_eccentricity','lamb_se':1, 'lamb_moment':1,'lamb_consprior':1,'ivd':True,'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/prostate.csv'},'norm_soft_size',1)]

LSIZEin = [('EntKLPropWMoment', {'weights_se':[0.1,0.9],'rel_diff':False,'temp':1,'linreg':False,'margin':0,'reg':'reg/prostate_ecc.txt','reg2':'reg/prostate_ecc.txt','mom_est':[0.02, 0.66],'moment_fn':'soft_inertia','lamb_se':1, 'lamb_moment':1,'lamb_consprior':1,'ivd':True,'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/prostate.csv'},'norm_soft_size',1)]

LSIZEcomp = [('EntKLPropWMoment', {'temp':1.01,'linreg':False,'margin':0.1,'reg':'reg/prostate_comp.txt','reg2':'reg/prostate_comp.txt','mom_est':[0, 0],'moment_fn':'saml_compactness','lamb_se':1, 'lamb_moment':0.001,'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/prostate.csv'},'norm_soft_size',1)]


# the folder containing the target dataset - site A is the target dataset and site B is the source one
T_FOLD = ./data/prostate3d/

#the network weights used as initialization of the adaptation
M_WEIGHTS_ul = /data/users/mathilde/ccnn/CDA/SFDA/results/prostate/cesource/last.pkl
M_WEIGHTS_ul = oneresults/tentprostate/enklsize2/
M_WEIGHTS_ul = oneresults3d/prostate/selfent/
M_WEIGHTS_ul = oneresults3d/prostate/enklsize3Dd/
#M_WEIGHTS_ul = oneresults/tentprostate/enklsize5/
#M_WEIGHTS_ul = oneresults/tentprostate/enklsize3/
#M_WEIGHTS_ul = oneresults/tentprostate/enklsize9/

#run the main experiment
TRN = oneresults3d/prostate/enklsizecent3Df
#TRN = oneresults3d/prostate/enklsizecent3Dc

REPO = $(shell basename `git rev-parse --show-toplevel`)
DATE = $(shell date +"%y%m%d")
HASH = $(shell git rev-parse --short HEAD)
HOSTNAME = $(shell hostname)
PBASE = archives
PACK = $(PBASE)/$(REPO)-$(DATE)-$(HASH)-$(HOSTNAME)-CSize.tar.gz

all: pack
plot: $(PLT)

pack: $(PACK) report
$(PACK): $(TRN)
	mkdir -p $(@D)
	tar cf - $^ | pigz > $@
	chmod -w $@
# tar -zc -f $@ $^  # Use if pigz is not available

# first train on the source dataset only and on target only:
results/prostate/cesourceaugnorm: OPT = --val_target_folders="$(SAUG_DATA_NORM)" --target_folders="$(SAUG_DATA_NORM)" --direct --batch_size 32  --target_losses="$(L_OR)" --target_dataset "/data/users/mathilde/ccnn/SAML/data/SB" \
	     --network UNet --model_weights="" --lr_decay 1 \

results/prostate/cesourceim: OPT =  --global_model --target_losses="$(L_OR)" \
	    --mode makeim  --l_rate 0 --saveim True --model_weights="/data/users/mathilde/ccnn/CDA/SFDA/results/prostate/cesource/last.pkl" --n_epoch 1\

oneresults3d/prostate/fs_allvarup: OPT =  --target_losses="$(L_OR)" --saveim True  --n_epoch 100 --l_rate 1e-3 --do_not_config_mod \

# on target

oneresults3d/prostate/cenklsizecent2: OPT = --saveim True --do_asd 1 --batch_size 32 --lr_decay 1 --n_epoch 150 --l_rate 5e-4 --ind_mom 1 --target_losses="$(LSIZECentroid3d)" \

oneresults3d/prostate/enklsize2D: OPT = --saveim True --batch_size 32 --lr_decay 1 --n_epoch 150 --l_rate 5e-4 --ind_mom 1 --target_losses="$(LSIZECentroid3dGT)" \

oneresults3d/prostate/enklsize3Dc: OPT =  --softmax_temp 0.1 --model_weights="/data/users/mathilde/ccnn/CDA/SFDA/results/prostate/cesource/last.pkl" --global_model --saveim True --batch_size 32 --lr_decay 1 --n_epoch 150 --l_rate 5e-4 --ind_mom 1 --target_losses="$(LSIZE3dGT)" \

oneresults3d/prostate/enklsize3Dd: OPT =  --softmax_temp 0.1  --saveim True --batch_size 32 --lr_decay 1 --n_epoch 150 --l_rate 5e-4 --ind_mom 1 --target_losses="$(LSIZE3dGT)" \

oneresults3d/prostate/enklsizecent3D: OPT =  --softmax_temp 0.1 --model_weights="/data/users/mathilde/ccnn/CDA/SFDA/results/prostate/cesource/last.pkl" --global_model --saveim True --batch_size 32 --lr_decay 1 --n_epoch 150 --l_rate 5e-4 --ind_mom 1 --target_losses="$(LSIZECentroid3dGT)" \

oneresults3d/prostate/enklsizecent3Df: OPT =   --saveim True --batch_size 32 --lr_decay 1  --l_rate 5e-4 --ind_mom 1 --target_losses="$(LSIZECentroid3dGT)" \

oneresults3d/prostate/enklsizedist2: OPT =   --saveim True --batch_size 32 --lr_decay 1  --l_rate 5e-4 --ind_mom 1 --target_losses="$(LSIZEDist3dGT)" \

oneresults3d/prostate/selfent: OPT =  --model_weights="/data/users/mathilde/ccnn/CDA/SFDA/results/prostate/cesource/last.pkl" --global_model --saveim True --batch_size 32 --lr_decay 1 --n_epoch 150 --l_rate 5e-4 --ind_mom 1 --target_losses="$(LSelfEnt)" \

oneresults3d/prostate/cent3dc: OPT =  --saveim True --batch_size 32 --lr_decay 1 --n_epoch 150 --l_rate 5e-4 --ind_mom 1 --target_losses="$(LCentroid3dGT)" \


#inference mode : saves the segmentation masks for a specific model saved as pkl file (ex. "results/prostate/cesource/last.pkl" below):
results/prostate/cesourceimN: OPT =  --target_losses="$(L_OR)" \
	   --pprint --do_asd True --mode makeim  --l_rate 0 --model_weights="$(M_WEIGHTS_ul)" --n_epoch 1\

results/prostate/fsim: OPT =  --target_losses="$(L_OR)" \
	   --mode makeim --do_asd 1 --do_hd 1   --l_rate 0 --model_weights="results/prostate/fsNew/best_3d.pkl" --pprint --n_epoch 1 --saveim True --entmap\

results/prostate/adaentim: OPT =  --target_losses="$(L_OR)" \
	   --mode makeim --do_asd 1 --do_hd 1   --l_rate 0 --model_weights="results/prostate/adaent/best_3d.pkl" --pprint --n_epoch 1 --saveim True --entmap\


$(TENTTRN) :
	$(CC) $(CFLAGS) tent_main.py --batch_size 20 --n_class 2 --workdir $@_tmp \
                --wh 384 --metric_axis 1  --n_epoch 150 --dice_3d --l_rate 5e-4 --weight_decay 1e-4 --train_grp_regex="$(G_RGX)" --grp_regex="$(G_RGX)" --network=$(NET) --val_target_folders="$(TT_DATA)"\
                  --lr_decay 0.9  --model_weights="$(M_WEIGHTS_ul)"  --target_folders="$(TT_DATA)" $(OPT) $(DEBUG)
	mv $@_tmp $@


$(ST) :
	$(CC) $(CFLAGS) main_sfda.py --batch_size 24 --n_class 2 --workdir $@_tmp --target_dataset "$(T_FOLD)" \
                --wh 384 --metric_axis 1  --n_epoch 151 --dice_3d --l_rate 5e-4 --weight_decay 1e-4 --train_grp_regex="$(RGX)" --grp_regex="$(RGX)" --network=$(NET) --val_target_folders="$(T_DATA)"\
                     --lr_decay 0.9  --model_weights="$(M_WEIGHTS_ul)"  --target_folders="$(T_DATA)" $(OPT) $(DEBUG)
	mv $@_tmp $@


$(TRN) :
	$(CC) $(CFLAGS) ttmain.py --valonly  --ontest --notent --regex_list "['Case22','Case17','Case26','Case05','Case02','Case07','Case08','Case12','Case15','Case20']" --batch_size 32 --n_class 2 --workdir $@_tmp --target_dataset "$(T_FOLD)" \
                --wh 384 --metric_axis 1  --n_epoch 151 --dice_3d --l_rate 1e-4 --weight_decay 1e-4 --train_grp_regex="$(G_RGX)" --grp_regex="$(G_RGX)" --network=$(NET) --val_target_folders="$(T_DATA3d)"\
                     --lr_decay 0.9  --model_weights="$(M_WEIGHTS_ul)"  --target_folders="$(T_DATA3d)" $(OPT) $(DEBUG)
	mv $@_tmp $@
	$(CC) get_learning_stats.py $@ prostate
	$(CC) plot_by_subj.py $@ 0

