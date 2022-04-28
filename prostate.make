CC = python
SHELL = bash
PP = PYTHONPATH="$(PYTHONPATH):."

.PHONY: all view plot report

CFLAGS = -O
#CFLAGS = -O -m pdb
#DEBUG = --debug

#the regex of the slices in the target dataset
#for the prostate
G_RGX = Case\d+_\d+

S_DATA = [('IMG', nii_transform2, False), ('GT', nii_gt_transform2, False), ('GT', nii_gt_transform2, False)]
TT_DATA = [('IMG', nii_transform2, False), ('GTNew', nii_gt_transform2, False), ('GTNew', nii_gt_transform2, False)]
L_OR = [('CrossEntropy', {'idc': [0,1], 'weights':[1,1]}, None, None, None, 1)]
NET = UNet

# the folder containing the target dataset - site A is the target dataset and site B is the source one
T_FOLD = data/prostate/SA
T_FOLD = /data/users/mathilde/ccnn/CDA/data/SAD
T_FOLD = /data/users/mathilde/ccnn/SAML/data/SA/

#the network weights used as initialization of the adaptation
M_WEIGHTS_ul = results/prostate/cesource/last.pkl

#run the main experiment
TRN = results/prostate/fsNew
TRN = results/prostate/ results/prostate/sfdaNew10_1e-6bis
#TRN = results/prostate/cesourceimN
#TENTTRN = resultstent/prostate/sfda
#TRN = results/prostate/adaentim
TRN = results/prostate/entkl
TRN = results/prostate/sfdaNew11_1e-6
TRN = results/prostate/sfdacomp0

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
results/prostate/cesource: OPT =  --target_losses="$(L_OR)" --target_dataset "/data/users/mathilde/ccnn/SAML/data/SB" --val_target_folders="$(S_DATA)" --target_folders="$(S_DATA)"\
	     --network UNet --model_weights="" --lr_decay 1 \
	    
# full supervision
results/prostate/fsNew: OPT =  --target_losses="$(L_OR)" \
	     --network UNet --lr_decay 1 \

# SFDA. Remove --saveim True --entmap --do_asd 1 --do_hd 1 to speed up
results/prostate/sfdaCase28: OPT = --target_losses="[('EntKLProp', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/prostate.csv'},'norm_soft_size',1)]" \
           --l_rate 0.000001 --train_grp_regex Case28*.nii \


resultstent/prostate/sfda: OPT = --dataset_list "['train']" --val_dataset "train" --base_fold "$(T_FOLD)" --target_losses="[('EntKLProp', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/prostate.csv'},'norm_soft_size',1)]" \
           --l_rate 0.000001 \



results/prostate/adaent2: OPT = --target_losses="[('EntKLProp', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/prostate.csv'},'norm_soft_size',1)]" \
           --l_rate 0.000001 --lr_decay 0.4 --do_asd True --pprint \

results/prostate/sfdaNew10_1e-6bis: OPT = --target_losses="[('EntKLProp', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[1,1],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/prostateNew.csv'},'norm_soft_size',1)]" \
           --l_rate 0.000001  --weight_decay 1e-3 --lr_decay 0.7 --do_asd True --pprint \

results/prostate/sfdaNew10: OPT = --target_losses="[('EntKLProp', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[1,1],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/prostateNew.csv'},'norm_soft_size',1)]" \
           --l_rate 0.0000001  --weight_decay 1e-3 --lr_decay 0.7 --do_asd True --pprint \

results/prostate/sfdacomp0: OPT = --target_losses="[('EntKLPropWComp', {'lamb_se':1,'lamb_comp':0, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/prostateNew.csv'},'norm_soft_size',1)]" \
           --l_rate 0.000001  --weight_decay 1e-3 --lr_decay 0.7 --pprint \

results/prostate/sfdaNew11_1e-6: OPT = --target_losses="[('EntKLProp', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/prostateNew.csv'},'norm_soft_size',1)]" \
           --l_rate 0.000001  --weight_decay 1e-3 --lr_decay 0.7 --pprint \

results/prostate/sfdaNew11: OPT = --target_losses="[('EntKLProp', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/prostateNew.csv'},'norm_soft_size',1)]" \
           --l_rate 0.0000001  --weight_decay 1e-3 --lr_decay 0.7 --do_asd True --pprint \

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

$(TRN) :
	$(CC) $(CFLAGS) main_sfda.py --batch_size 24 --n_class 2 --workdir $@_tmp --target_dataset "$(T_FOLD)" \
                --wh 384 --metric_axis 1  --n_epoch 150 --dice_3d --l_rate 5e-4 --weight_decay 1e-4 --train_grp_regex="$(G_RGX)" --grp_regex="$(G_RGX)" --network=$(NET) --val_target_folders="$(TT_DATA)"\
                  --lr_decay 0.9  --model_weights="$(M_WEIGHTS_ul)"  --target_folders="$(TT_DATA)" $(OPT) $(DEBUG)
	mv $@_tmp $@


