CC = python
SHELL = bash
PP = PYTHONPATH="$(PYTHONPATH):."

.PHONY: all view plot report

#CFLAGS = -O
#CFLAGS = -m pdb
#DEBUG = --debug

#the regex of the subjects in the target dataset
#for the ivdsag
G_RGX = Subj_\d+

TT_DATA = [('Inn', png_transform, False), ('GT', gtpng_transform_remap, False),('GT', gtpng_transform_remap, False)]
TTrain_DATA = [('Inn', png_transform, False), ('GT', gtpng_transform_remap, False),('GT', gtpng_transform_remap, False)]
TTest_DATA = [('Inn', png_transform, False), ('GT', gtpng_transform, False),('GT', gtpng_transform, False)]
S_DATA = [('Wat', png_transform, False), ('GT', gtpng_transform_remap, False),('GT', gtpng_transform_remap, False)]
S_DATAAUG = [('Wataug', png_transform, False), ('GTaug', gtpng_transform, False),('GTaug', gtpng_transform, False)]
L_OR = [('CrossEntropy', {'idc': [0,1], 'weights':[1,1]}, None, None, None, 1)]
NET = UNet

# the folder containing the datasets
B_FOLD = data/ivd_transverse/
B_FOLD = /data/users/mathilde/ccnn/CDA/data/all_sagittal2/

#the network weights used as initialization of the adaptation
M_WEIGHTS_ul = results/ivdsag/cesource/last.pkl
#M_WEIGHTS_ul = results/ivdsag/cesourceaug/last.pkl

NBSUBJ = 7 

#run the main experiment
TRN = results/ivdsag/hpsfda results/ivdsag/hpsfda6 
#TRN = results/ivdsag/hpsfda5 results/ivdsag/hpsfda4
#TRN = results/ivdsag/hpsfda3 results/ivdsag/hpsfda2
TRN = results/ivdsag/hpsfda2

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


results/ivdsag/hpsfda6: OPT = --target_losses="[('EntKLProp', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags6','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/ivdsag.csv'},'norm_soft_size',1)]" --lr_decay 0.7 --l_rate 1e-6 \
	--target_folders="$(TT_DATA)" --val_target_folders="$(TT_DATA)" \

results/ivdsag/hpsfda5: OPT = --target_losses="[('EntKLProp', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags5','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/ivdsag.csv'},'norm_soft_size',1)]" --lr_decay 0.7 --l_rate 1e-6 \
	--target_folders="$(TT_DATA)" --val_target_folders="$(TT_DATA)" \

results/ivdsag/hpsfda4: OPT = --target_losses="[('EntKLProp', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags4','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/ivdsag.csv'},'norm_soft_size',1)]" --lr_decay 0.7 --l_rate 1e-6 \
	--target_folders="$(TT_DATA)" --val_target_folders="$(TT_DATA)" \

results/ivdsag/hpsfda3: OPT = --target_losses="[('EntKLProp', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags3','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/ivdsag.csv'},'norm_soft_size',1)]" --lr_decay 0.7 --l_rate 1e-6 \
	--target_folders="$(TT_DATA)" --val_target_folders="$(TT_DATA)" \

results/ivdsag/hpsfda2: OPT = --target_losses="[('EntKLProp', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags2','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/ivdsag.csv'},'norm_soft_size',1)]" --lr_decay 0.7 --l_rate 1e-6 \
	--target_folders="$(TT_DATA)" --val_target_folders="$(TT_DATA)" \

results/ivdsag/hpsfda: OPT = --target_losses="[('EntKLProp', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/ivdsag.csv'},'norm_soft_size',1)]" --lr_decay 0.7 --l_rate 1e-6 \
	--target_folders="$(TT_DATA)" --val_target_folders="$(TT_DATA)" \


$(TRN) :
	$(CC) $(CFLAGS) main_sfda.py --batch_size 24 --n_class 2 --workdir $@_tmp --target_dataset "$(B_FOLD)"  \
                --grp_regex="$(G_RGX)"  --target_folders="$(TTrain_DATA)" --val_target_folders="$(TTest_DATA)"\
                --model_weights="$(M_WEIGHTS_ul)" --network=$(NET) \
                --lr_decay 0.9 --metric_axis 1  --n_epoch 150 --dice_3d --l_rate 1e-6 --lr_decay_epoch 20 --weight_decay 1e-4 $(OPT) $(DEBUG)\

	mv $@_tmp $@


