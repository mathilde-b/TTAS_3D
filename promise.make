CC = python
SHELL = bash
PP = PYTHONPATH="$(PYTHONPATH):."

.PHONY: all view plot report

#CFLAGS = -O
#DEBUG = --debug

#the regex of the slices in the target dataset
#for the promise
G_RGX = Case\d+_\d+

TT_DATA = [('IMG', nii_transform_normalize, False), ('GT', nii_gt_transform, False), ('GT', nii_gt_transform, False)]
S_DATA = [('IMG', nii_transform_normalize, False), ('GT', nii_gt_transform_expand, False), ('GT', nii_gt_transform_expand, False)]

#TT_DATA = [('IMG', nii_transform2, False), ('GTNew', nii_gt_transform2, False), ('GTNew', nii_gt_transform2, False)]
L_OR = [('CrossEntropy', {'idc': [0,1], 'weights':[1,1]}, None, None, None, 1)]
NET = UNet

# the folder containing the target dataset - site A is the target dataset and site B is the source one
T_FOLD = data/promise/SA
T_FOLD = /data/users/mathilde/ccnn/CDA/data/SAD
T_FOLD = /data/users/mathilde/SAML/data/prom/

#the network weights used as initialization of the adaptation
M_WEIGHTS_ul = results/promise/cesource_othernorm_tmp/best_3d.pkl

#run the main experiment
TRN = results/promise/cesource_othernorm
TRN = results/promise/cesourceim


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
results/promise/cesource: OPT =  --target_losses="$(L_OR)" --target_dataset "/data/users/mathilde/ccnn/CDA/data/SAD" \
	     --network UNet --model_weights="" --lr_decay 1 --ontest \

results/promise/cesource_othernorm: OPT =  --target_losses="$(L_OR)" --target_dataset "/data/users/mathilde/SAML/data/sa" \
	     --network UNet --model_weights="" --lr_decay 1  \

# full supervision
results/promise/fs: OPT =  --target_losses="$(L_OR)" \
	     --network UNet --lr_decay 1 \

# SFDA. Remove --saveim True --entmap --do_asd 1 --do_hd 1 to speed up

results/promise/sfda_proposal: OPT = --target_losses="[('EntKLProp', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'proposal_size','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/promise.csv'},'norm_soft_size',1)]" \
           --l_rate 0.000001 \

results/promise/sfda_notag2: OPT = --target_losses="[('EntKLPropNoTag', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbprednotags','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/promise2.csv'},'norm_soft_size',1)]" \
           --l_rate 0.000001 \

results/promise/sfda_proselect2: OPT = --target_losses="[('EntKLPropSelect', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredselect','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/promise2.csv'},'norm_soft_size',1)]" \
           --l_rate 0.000001 \

results/promise/sfda: OPT = --target_losses="[('EntKLProp', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/promise.csv'},'norm_soft_size',1)]" \
           --l_rate 0.000001 \

#inference mode : saves the segmentation masks for a specific model saved as pkl file (ex. "results/promise/cesource/last.pkl" below):
results/promise/cesourceim: OPT =  --target_losses="$(L_OR)" \
	   --do_asd 1  --pprint --mode makeim  --batch_size 1  --l_rate 0 --model_weights="$(M_WEIGHTS_ul)"  --n_epoch 1\

results/promise/sfdaim: OPT =  --target_losses="$(L_OR)" \
	   --mode makeim --saveim True --entmap --do_asd 1 --do_hd 1  --batch_size 1  --l_rate 0 --model_weights="results/promise/sfda/best_3d.pkl" --pprint --n_epoch 1 --saveim True --entmap\

results/promise/sfda_onsource: OPT =  --target_losses="$(L_OR)" \
	   --mode makeim --l_rate 0 --model_weights="results/promise/sfda3/best_3d.pkl" --pprint --n_epoch 1 --val_target_folders="$(S_DATA)"  \

results/promise/cesource_onsource: OPT =  --target_losses="$(L_OR)" \
	   --mode makeim --l_rate 0  --pprint --n_epoch 1 --val_target_folders="$(S_DATA)"  \


$(TRN) :
	$(CC) $(CFLAGS) main_sfda.py --batch_size 28 --n_class 2 --workdir $@_tmp --target_dataset "$(T_FOLD)" \
                --wh 384 --metric_axis 1  --n_epoch 150 --dice_3d --l_rate 5e-4 --weight_decay 1e-4 --grp_regex="$(G_RGX)" --network=$(NET) --val_target_folders="$(TT_DATA)"\
                  --lr_decay 0.9  --model_weights="$(M_WEIGHTS_ul)"  --target_folders="$(TT_DATA)" $(OPT) $(DEBUG)
	mv $@_tmp $@


