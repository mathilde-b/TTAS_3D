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

LSIZECentroid3d= [('EntKLPropWMoment2', {'weights_se':[0.1,0.9],'lamb_se':1,'lamb_moment':0.0001,'rel_diff':False, 'matrix':False, 'linreg':False,'temp':1.01,'margin':0,'reg':'reg/tentprostate_cenklsizecentav.txt','reg2':'reg/tentprostate_cenklsizecentav.txt','mom_est':[[23, 191.52, 188.53]],'moment_fn':'soft_centroid_3d', 'lamb_consprior':1,'ivd':True,'idc_c': [1],'curi':False,'power': 1},'PreciseBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/prostate.csv'},'norm_soft_size',1)]

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
T_FOLD = /data/users/mathilde/ccnn/SAML/data/SA/

#the network weights used as initialization of the adaptation
M_WEIGHTS_ul = /data/users/mathilde/ccnn/CDA/SFDA/results/prostate/cesource/last.pkl
M_WEIGHTS_ul = oneresults/tentprostate/enklsize2/
#M_WEIGHTS_ul = oneresults/tentprostate/enklsize5/
#M_WEIGHTS_ul = oneresults/tentprostate/enklsize3/
#M_WEIGHTS_ul = oneresults/tentprostate/enklsize9/

#run the main experiment

#TRN = oneresults/tentprostate/cenklsizequadcentroid2
TRN = oneresults/tentprostate/cenklsizequadcomp3
TRN = oneresults/tentprostate/cenklsizequadecc5
TRN = oneresults/tentprostate/enklsize5
TRN = oneresults/tentprostate/cenklsizequaddist7 oneresults/tentprostate/cenklsizequadcomp7 oneresults/tentprostate/cenklsizequadecc9
TRN = oneresults/tentprostate/fs2
TRN = oneresults/tentprostate/cenklsizecentavdiv2
#TRN = oneresults/tentprostate/cenklsizedistav4
TRN = oneresults/tentprostate/cenklsizecentavdiv3
#TRN = oneresults/tentprostate/cenklsizecentav oneresults/tentprostate/cenklsizeeccav
#TRN = oneresults/tentprostate/cenklsizecompzero
TRN = oneresults/prostateot/enklsize3
TRN = oneresults/prostateot/enklsizenorm3
TRN = oneresults/tentprostate/cenklsizeecc2 oneresults/tentprostate/cenklsizeecc3 oneresults/tentprostate/cenklsizeecc4 oneresults/tentprostate/cenklsizeecc5
TRN = oneresults3d/tentprostate/enklsizecent
ST = results/prostate/cesourceaugnorm


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

oneresults/tentprostate/fs2: OPT =  --target_losses="$(L_OR)" --saveim True --do_asd 1 --n_epoch 250 \

# on target
oneresults/tentprostate/tentw: OPT =  --softmax_temp 0.5 --target_losses="$(LTent)" --saveim True --do_asd 1 --model_weights /data/users/mathilde/ccnn/CDA/SFDA/results/prostate/cesource/last.pkl --global_model\

oneresults/tentprostate/enklsize8: OPT =  --model_weights results/prostate/cesourceaug/best_3d.pkl --global_model  --batch_size 32 --l_rate 5e-4 --lr_decay 0.95  --target_losses="$(LSIZE)" \

oneresults/tentprostate/enklsize9: OPT =  --model_weights results/prostate/cesourceaug/last.pkl --global_model  --batch_size 32 --l_rate 5e-4 --lr_decay 0.7  --target_losses="$(LSIZE)" \

oneresults/prostateot/enklsizenorm3: OPT = --target_losses="$(LSIZE)" --target_folders="$(T_DATANORM)" --val_target_folders="$(T_DATANORM)" --l_rate 5e-4 --batch_size 22 --lr_decay_epoch 40 --n_epoch 200 --trainonly --model_weights results/prostate/cesourceaugnorm/best_3d.pkl --global_model --regex_list "$(RGL)"    --trainval \

oneresults/prostateot/enklsize3: OPT = --target_losses="$(LSIZE)" --l_rate 1e-3 --batch_size 32 --lr_decay_epoch 40 --n_epoch 300 --trainonly --model_weights results/prostate/cesourceaug/last.pkl --global_model --regex_list "$(RGL)" --trainval \

oneresults/prostateot/enklsize2ter: OPT = --target_losses="$(LSIZE)" --l_rate 5e-4 --batch_size 32 --lr_decay_epoch 20 --n_epoch 200 --trainonly --model_weights /data/users/mathilde/ccnn/CDA/SFDA/results/prostate/cesource/last.pkl --global_model --regex_list "$(RGL)" --trainval \

oneresults/prostateot/enklgtsizeim: OPT = --n_epoch 1 --trainonly --model_weights oneresults/prostateot/enklgtsize --regex_list "$(RGL)" --batch_size 32 --l_rate 5e-4  --trainval --target_losses="$(LSIZE)" \

oneresults/prostateot/cesourceaugim2: OPT = --n_epoch 1 --trainonly --model_weights results/prostate/cesourceaug/last.pkl --global_model --regex_list "$(RGL)" --batch_size 32 --l_rate 5e-4  --trainval --target_losses="$(LSIZE)" \

oneresults/prostate/enklsize: OPT = --l_rate 5e-6 --model_weights /data/users/mathilde/ccnn/CDA/SFDA/results/prostate/cesource/last.pkl --global_model   --do_not_config_mod --lr_decay 1 --n_epoch 250 --target_losses="$(LSIZE)"  \

oneresults/tentprostate/enklsizenu21bis: OPT = --target_losses="$(LSIZEnu)"  \
oneresults/tentprostate/cenklsizeNu: OPT =  --ind_mom 1 --target_losses="$(LSIZENu2)" --saveim True --do_asd 1\
oneresults/tentprostate/cenklsizequadcentroid2: OPT = --ind_mom 1 --target_losses="$(LSIZECentroid)" --saveim True --do_asd 1\
oneresults/tentprostate/foo: OPT = --target_losses="$(LSIZECentroid)" --saveim True --do_asd 1 --batch_size 4 \
oneresults/tentprostate/cenklsizequadlen: OPT = --target_losses="$(LSIZElen)" --saveim True --batch_size 16 \
oneresults/tentprostate/cenklsizequadcomp7: OPT =  --target_losses="$(LSIZEcomp)" --saveim True  --do_asd 1 --entmap --l_rate 1e-4 --n_epoch 151 \
oneresults/tentprostate/cenklsizecompzero: OPT =  --target_losses="$(LSIZEcomp)" --saveim True --do_asd 1\
oneresults/tentprostate/cenklsizequadecc9: OPT =   --target_losses="$(LSIZEecc)" --saveim True  --do_asd 1 --entmap \
oneresults/tentprostate/cenklsizeeccmed: OPT =  --update_mom_est --target_losses="$(LSIZEecc)" --saveim True --do_asd 1\
oneresults/tentprostate/cenklsizequad2mom: OPT = --val_target_folders="$(TT_DATA)" --target_folders="$(TT_DATA)" --ind_mom 1 --target_losses="$(LSIZECentroidQuadDist)" \
oneresults/tentprostate/enklsizedistlr: OPT = --update_lin_reg --ind_mom 1 --target_losses="$(LSIZEQuadDist)" --saveim True --do_asd 1 \
oneresults/tentprostate/enklsizecentlr: OPT = --update_lin_reg --ind_mom 1 --target_losses="$(LSIZECentroid)" --saveim True --do_asd 1 \
oneresults/tentprostate/cklsizedistav: OPT = --n_epoch 250 --lr_decay 0.95 --l_rate 1e-4 --update_mom_est --ind_mom 1 --target_losses="$(LSIZEQuadDist)" --saveim True --do_asd 1 \

oneresults/tentprostate/cenklsizequadeccbis: OPT =  --batch_size 10 --l_rate 5e-4 --target_losses="$(LSIZEecc)"

oneresults/tentprostate/cenklsizequaddistf2: OPT =  --target_losses="$(LSIZEQuadDist)" --n_epoch 250 \

oneresults/tentprostate/cenklsizecentf3qua: OPT = --do_asd 1 --batch_size 32 --lr_decay 1 --n_epoch 150 --l_rate 5e-4 --update_mom_est --ind_mom 1 --target_losses="$(LSIZECentroid)" \

oneresults3d/tentprostate/cenklsizecent2: OPT = --saveim True --do_asd 1 --batch_size 32 --lr_decay 1 --n_epoch 150 --l_rate 5e-4 --ind_mom 1 --target_losses="$(LSIZECentroid3d)" \

oneresults3d/tentprostate/enklsizecent: OPT = --saveim True --do_asd 1 --batch_size 32 --lr_decay 1 --n_epoch 150 --l_rate 5e-4 --ind_mom 1 --target_losses="$(LSIZECentroid3d)" \

oneresults/tentprostatenb/cenklsizecent: OPT =  --n_epoch 150 --saveim True --l_rate 1e-4 --update_mom_est --ind_mom 1 --target_losses="$(LSIZECentroid)"\
oneresults/tentprostatenb/cklsizecentav: OPT =  --model_weights /data/users/mathilde/ccnn/CDA/SFDA/results/prostate/cesource/last.pkl --global_model  --n_warmup 200 --n_epoch 400 --l_rate 5e-4 --update_mom_est --ind_mom 1 --target_losses="$(LSIZECentroid)"\
oneresults/tentprostatenb/cklsizedistav2: OPT = --model_weights /data/users/mathilde/ccnn/CDA/SFDA/results/prostate/cesource/last.pkl --global_model  --n_warmup 150 --n_epoch 300 --l_rate 1e-4 --lr_decay 0.9 --update_mom_est --ind_mom 1 --target_losses="$(LSIZEQuadDist)" \


oneresults/tentprostate/cenklsizedistav4bis: OPT = --n_epoch 250 --lr_decay 0.95 --l_rate 1e-4 --update_mom_est --ind_mom 1 --target_losses="$(LSIZEQuadDist)" --saveim True --do_asd 1 \

oneresults/tentprostate/cenklsizedist2qua: OPT = --do_asd 1 --saveim True --med --thl "low" --n_epoch 250 --lr_decay 0.9 --l_rate 5e-4  --update_mom_est --ind_mom 1 --target_losses="$(LSIZEQuadDist)" \

oneresults/tentprostate/cenklsizedistf: OPT =  --med --thl "low" --n_epoch 250 --lr_decay 0.9  --update_mom_est --ind_mom 1 --target_losses="$(LSIZEQuadDist)" \

oneresults/tentprostate/cenklsizedist3: OPT =  --med --thl "low" --batch_size 32 --n_epoch 250 --lr_decay 0.9 --l_rate 5e-4  --update_mom_est --ind_mom 1 --target_losses="$(LSIZEQuadDist)" \

oneresults/tentprostate/cenklsizedist2quin: OPT = --do_asd 1 --saveim True --med --thl "low" --n_epoch 250 --lr_decay 0.9 --l_rate 5e-4  --update_mom_est --ind_mom 1 --target_losses="$(LSIZEQuadDist)" \

oneresults/tentprostate/cenklsizecent3ter: OPT = --med --thl "low" --do_asd 1 --saveim True --n_epoch 250 --lr_decay_epoch 40 --lr_decay 0.9 --l_rate 1e-4  --update_mom_est --ind_mom 1 --target_losses="$(LSIZECentroid)" \

oneresults/tentprostate/cenklsizeecc2: OPT = --do_asd 1 --saveim True --batch_size 32 --med --thl "low" --n_epoch 250 --lr_decay_epoch 40 --lr_decay 0.9 --l_rate 1e-4  --update_mom_est --target_losses="$(LSIZEecc)" \
oneresults/tentprostate/cenklsizeecc3: OPT = --do_asd 1 --saveim True --batch_size 32 --med  --n_epoch 250 --lr_decay_epoch 40 --lr_decay 0.9 --l_rate 1e-4  --update_mom_est --target_losses="$(LSIZEecc)" \
oneresults/tentprostate/cenklsizeecc4: OPT = --do_asd 1 --saveim True --batch_size 32   --n_epoch 250 --lr_decay_epoch 40 --lr_decay 0.9 --l_rate 1e-4  --update_mom_est --target_losses="$(LSIZEecc)" \
oneresults/tentprostate/cenklsizeecc5: OPT = --do_asd 1 --saveim True --batch_size 32 --thl "low"  --n_epoch 250 --lr_decay_epoch 40 --lr_decay 0.9 --l_rate 1e-4  --update_mom_est --target_losses="$(LSIZEecc)" \

oneresults/tentprostate/cenklsizequadecc7ter: OPT = --batch_size 40 --dic_params "oneresults/tentprostate/cenklsizequadecc7/learnparams.txt" --target_losses="$(LSIZEecc)" \

oneresults/tentprostate/cenklsizequadecc7wupdate2: OPT = --thl "med" --update_mom_est --dic_params "oneresults/tentprostate/cenklsizequadecc7/learnparams.txt" --target_losses="$(LSIZEecc)" \

#oneresults/tentprostate/cenklsizein: OPT = --batch_size 32 --med --thl "low" --n_epoch 250 --lr_decay_epoch 40 --lr_decay 0.9 --l_rate 1e-4  --update_mom_est --target_losses="$(LSIZEin)" \



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
	$(CC) $(CFLAGS) main_sfda_tent2.py --valonly  --ontest --notent --regex_list "['Case22','Case17','Case26','Case05','Case02','Case07','Case08','Case12','Case15','Case20']" --batch_size 24 --n_class 2 --workdir $@_tmp --target_dataset "$(T_FOLD)" \
                --wh 384 --metric_axis 1  --n_epoch 151 --dice_3d --l_rate 1e-4 --weight_decay 1e-4 --train_grp_regex="$(G_RGX)" --grp_regex="$(G_RGX)" --network=$(NET) --val_target_folders="$(T_DATA)"\
                     --lr_decay 0.9  --model_weights="$(M_WEIGHTS_ul)"  --target_folders="$(T_DATA)" $(OPT) $(DEBUG)
	mv $@_tmp $@
	$(CC) get_learning_stats.py $@ prostate
	$(CC) plot_by_subj.py $@ 0

