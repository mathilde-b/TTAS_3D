CC = python
SHELL = bash
PP = PYTHONPATH="$(PYTHONPATH):."

.PHONY: all view plot report

CFLAGS = -O

#the regex of the slices in the target dataset
#for the ivd
G_RGX = Subj_\d+_\d+

T_DATA = [('Inn', png_transform, False), ('GT', gtpng_transform, False),('GT', gtpng_transform, False)]
TAUG_DATA = [('Innaug', png_transform, False), ('GTaug', gtpng_transform, False),('GTaug', gtpng_transform, False)]
S_DATA = [('Wat', png_transform, False), ('GT', gtpng_transform, False),('GT', gtpng_transform, False)]
S_DATAAUG = [('Wataug', png_transform, False), ('GTaug', gtpng_transform, False),('GTaug', gtpng_transform, False)]
L_OR = [('CrossEntropy', {'idc': [0,1], 'weights':[1,1]}, None, None, None, 1)]
NET = UNet

LSIZE = [('EntKLProp', {'moment_fn':'soft_size','lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':';','sizefile':'sizes/ivd.csv'},'norm_soft_size',1)]

LSIZEQuadDist = [('EntKLPropWMoment', {'mom_est':[[112.33, 21.51],[112.17, 18.08]],'moment_fn':'soft_dist_centroid','lamb_se':1, 'lamb_moment':0.0001,'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':';','sizefile':'sizes/ivd.csv'},'norm_soft_size',1)]

LSIZECentroid = [('EntKLPropWMoment', {'mom_est':[[191.52, 192.12],[191.55, 188.53]],'moment_fn':'soft_centroid','lamb_se':1, 'lamb_moment':0.1,'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':';','sizefile':'sizes/ivd.csv'},'norm_soft_size',1)]

LSIZElength = [('EntKLPropWMoment', {'moment_fn':'soft_length','lamb_se':1, 'lamb_moment':0.001,'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':';','sizefile':'sizes/ivd.csv'},'norm_soft_size',1)]

LSIZEecc = [('EntKLPropWMoment', {'mom_est':[0.02, 0.63],'moment_fn':'soft_eccentricity','lamb_se':1, 'lamb_moment':1,'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':';','sizefile':'sizes/ivd.csv'},'norm_soft_size',1)]

LSIZEcomp = [('EntKLPropWMoment', {'mom_est':[0.04, 1.68],'moment_fn':'saml_compactness','lamb_se':1, 'lamb_moment':0.001,'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':';','sizefile':'sizes/ivd.csv'},'norm_soft_size',1)]

LSIZElen = [('EntKLPropWMoment', {'margin':0.5,'mom_est':[718.0, 838.0],'moment_fn':'soft_length','lamb_se':1, 'lamb_moment':0.001,'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':';','sizefile':'sizes/ivd.csv'},'norm_soft_size',1)]

LSIZEnu = [('EntKLPropWMomentNu', {,'moment_fn':'soft_centroid','ind_moment':[1,2,1],'lamb_se':1, 'lamb_moment':0.1,'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':';','sizefile':'sizes/ivd.csv'},'norm_soft_size',1)]


# the folder containing the target dataset -
T_FOLD = /data/users/mathilde/ccnn/CDA/data/all_transverse

#the network weights used as initialization of the adaptation
M_WEIGHTS_ul = /data/users/mathilde/ccnn/CDA/SFDAA/results/ivd/cesource/last.pkl
#M_WEIGHTS_ul = oneresults/tentivd/enklsize2/


TRN = oneresults/tentivd/cenklsizequaddist2
#TRN = oneresults/tentivd/cenklsizequadcentroid2
TRN = oneresults/tentivd/cenklsizequadcomp3
TRN = oneresults/tentivd/cenklsizequadecc5
TRN = oneresults/ivd/enklsize

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

# first train on the source dataset only:
results/ivd/cesource: OPT =  --target_losses="$(L_OR)" --target_dataset "/data/users/mathilde/ccnn/SAML/data/SB" --val_target_folders="$(S_DATA)" --target_folders="$(S_DATA)"\
	     --network UNet --model_weights="" --lr_decay 1 \


# on target
oneresults/tentivd/enklsize: OPT = --global_model --lr_decay 1 --n_epoch 250 --target_losses="$(LSIZE)" \

oneresults/ivd/enklsize: OPT = --l_rate 0 --global_model --lr_decay 1 --n_epoch 250 --target_losses="$(LSIZE)"  \

oneresults/tentivd/enklsizenu21bis: OPT = --target_losses="$(LSIZEnu)"  \

oneresults/tentivd/cenklsizequaddist2: OPT =  --ind_mom 1 --target_losses="$(LSIZEQuadDist)" --saveim True --do_asd 1\

oneresults/tentivd/cenklsizequadcentroid2: OPT = --ind_mom 1 --target_losses="$(LSIZECentroid)" --saveim True --do_asd 1\

oneresults/tentivd/foo: OPT = --target_losses="$(LSIZECentroid)" --saveim True --do_asd 1 --batch_size 4 \

oneresults/tentivd/cenklsizequadlen: OPT = --target_losses="$(LSIZElen)" --saveim True --batch_size 16 \

oneresults/tentivd/cenklsizequadcomp3: OPT =  --target_losses="$(LSIZEcomp)" --saveim True  --do_asd 1 --entmap --l_rate 1e-4 --n_epoch 151 \

oneresults/tentivd/cenklsizequadecc5: OPT =  --target_losses="$(LSIZEecc)" --saveim True  --do_asd 1 --entmap \



#inference mode : saves the segmentation masks for a specific model saved as pkl file (ex. "results/ivd/cesource/last.pkl" below):
results/ivd/cesourceimN: OPT =  --target_losses="$(L_OR)" \
	   --pprint --do_asd True --mode makeim  --l_rate 0 --model_weights="$(M_WEIGHTS_ul)" --n_epoch 1\

results/ivd/fsim: OPT =  --target_losses="$(L_OR)" \
	   --mode makeim --do_asd 1 --do_hd 1   --l_rate 0 --model_weights="results/ivd/fsNew/best_3d.pkl" --pprint --n_epoch 1 --saveim True --entmap\

results/ivd/adaentim: OPT =  --target_losses="$(L_OR)" \
	   --mode makeim --do_asd 1 --do_hd 1   --l_rate 0 --model_weights="results/ivd/adaent/best_3d.pkl" --pprint --n_epoch 1 --saveim True --entmap\


$(TENTTRN) :
	$(CC) $(CFLAGS) tent_main.py --batch_size 20 --n_class 2 --workdir $@_tmp \
                --wh 384 --metric_axis 1  --n_epoch 150 --dice_3d --l_rate 5e-4 --weight_decay 1e-4 --train_grp_regex="$(G_RGX)" --grp_regex="$(G_RGX)" --network=$(NET) --val_target_folders="$(TT_DATA)"\
                  --lr_decay 0.9  --model_weights="$(M_WEIGHTS_ul)"  --target_folders="$(TT_DATA)" $(OPT) $(DEBUG)
	mv $@_tmp $@


$(TRN) :
	$(CC) $(CFLAGS) main_sfda_tent2.py --valonly  --ontest --notent --regex_list "['Subj_5','Subj_15','Subj_0']" --batch_size 24 --n_class 2 --workdir $@_tmp --target_dataset "$(T_FOLD)" \
                --wh 384 --metric_axis 1  --n_epoch 151 --dice_3d --l_rate 5e-4 --weight_decay 1e-4 --train_grp_regex="$(G_RGX)" --grp_regex="$(G_RGX)" --network=$(NET) --val_target_folders="$(T_DATA)"\
                     --lr_decay 0.9  --model_weights="$(M_WEIGHTS_ul)"  --target_folders="$(T_DATA)" $(OPT) $(DEBUG)
	mv $@_tmp $@
	$(CC) get_learning_stats.py $@ ivd


