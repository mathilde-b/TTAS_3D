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
TTTT_DATA = [('Inn', png_transform, False), ('GT', gtpng_transform_remap, False),('GT', gtpng_transform_remap, False),('GT', gtpng_transform_remap, False),('GT', gtpng_transform_remap, False),('GT', gtpng_transform_remap, False)]
TTrain_DATA = [('Inn', png_transform, False), ('GT', gtpng_transform_remap, False),('GT', gtpng_transform_remap, False)]
TTest_DATA = [('Inn', png_transform, False), ('GT', gtpng_transform, False),('GT', gtpng_transform, False)]
S_DATA = [('Wat', png_transform, False), ('GT', gtpng_transform_remap, False),('GT', gtpng_transform_remap, False)]
S_DATAAUG = [('Wataug', png_transform, False), ('GTaug', gtpng_transform, False),('GTaug', gtpng_transform, False)]
L_OR = [('CrossEntropy', {'idc': [0,1], 'weights':[1,1]}, None, None, None, 1)]
NET = UNet
LOSS=[('NaivePenalty', {'idc': [1]}, 'PreciseBounds', {'margin': 0.10, 'mode': 'percentage'}, 'soft_size', 1e-2), \
	('NaivePenalty', {'idc': [1]}, 'PreciseBounds', {'margin': 0.10, 'mode': 'abs'}, 'soft_centroid', 1e-2), \
	('NaivePenalty', {'idc': [1]}, 'PreciseBounds', {'margin': 0.10, 'mode': 'abs'}, 'soft_dist_centroid', 1e-2), \
	('NaivePenalty', {'idc': [1]}, 'PreciseBounds', {'margin': 0.10, 'mode': 'percentage'}, 'soft_length', 1e-2)] \

LOSS=[('NaivePenalty', {'idc': [1]}, 'PreciseBounds', {'margin': 0.10, 'mode': 'percentage'}, 'soft_size', 1e-2)] \


# the folder containing the datasets
B_FOLD = data/ivd_transverse/
B_FOLD = /data/users/mathilde/ccnn/CDA/data/all_sagittal2/

#the network weights used as initialization of the adaptation
M_WEIGHTS_ul = results/ivdsag/cesource/last.pkl
#M_WEIGHTS_ul = results/ivdsag/cesourceaug/last.pkl

NBSUBJ = 7 

#run the main experiment
TRN = results/ivdsag/sfda_zero_train_subj15
TRN = results/ivdsag/sfdaim
TRN = results/ivdsag/sfda2subj$(NBSUBJ)
TRN = results/ivdsag/cesourceNew
TRN = results/ivdsag/4desc
TRN = results/ivdsag/quadsize2

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
results/ivdsag/cesourceaug: OPT =  --target_losses="$(L_OR)" --target_folders="$(S_DATAAUG)" --val_target_folders="$(S_DATA)" \
	     --network UNet --model_weights="" --lr_decay 0.5 --n_epoch 200 --l_rate 5e-4 \
	    
# full supervision
results/ivdsag/fs2: OPT =  --target_losses="$(L_OR)" \
	     --network UNet --lr_decay 1 --saveim True --entmap --do_asd 1  \

results/ivdsag/quadsize2: OPT = --target_losses="$(LOSS)" --lr_decay 0.9 --l_rate 1e-5 \


results/ivdsag/4desc: OPT = --target_losses="$(LOSS)" --lr_decay 0.9 --l_rate 1e-6 \
	--target_folders="$(TTTT_DATA)" --val_target_folders="$(TTTT_DATA)" \

# SFDA. Remove --saveim True --entmap --do_asd 1 --do_hd 1 to speed up
results/ivdsag/sfdaNew: OPT = --target_losses="[('EntKLProp', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags2','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/ivdsag.csv'},'norm_soft_size',1)]" --l_rate 1e-6 --lr_decay_epoch 20 \

results/ivdsag/kldumbpred: OPT = --target_losses="[('AdaEntKLProp', {'lamb_se':0, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags2','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/ivdsag.csv'},'norm_soft_size',1)]" --l_rate 5e-5 --lr_decay 0.2 --lr_decay_epoch 20 \

results/ivdsag/sfda5osp: OPT = --target_losses="[('EntKLProp', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags4','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/ivdsag.csv'},'norm_soft_size',1)]" --l_rate 2e-5 --lr_decay 0.1 --lr_decay_epoch 10 \


results/ivdsag/sfda4: OPT = --target_losses="[('EntKLProp', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags2','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/ivdsag.csv'},'norm_soft_size',1)]" --l_rate 1e-5  --n_epoch 300 --lr_decay_epoch 20 \


results/ivdsag/sfda4augvbn: OPT = --target_losses="[('EntKLProp', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags2','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/ivdsag.csv'},'norm_soft_size',1)]" --l_rate 1e-5  --n_epoch 300 --lr_decay_epoch 20 --pprint \

results/ivdsag/sfda5: OPT = --target_losses="[('EntKLProp', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags2','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/ivdsag.csv'},'norm_soft_size',1)]" --l_rate 2e-5 --lr_decay 0.1 --lr_decay_epoch 10 \


results/ivdsag/sfda2qua: OPT = --target_losses="[('EntKLProp', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags2','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/ivdsag.csv'},'norm_soft_size',1)]" --l_rate 5e-5 --lr_decay 0.2 --lr_decay_epoch 20 --do_asd True\


results/ivdsag/sfda_select2: OPT = --target_losses="[('EntKLPropSelect', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredselect2','power': 1, 'mode':'percentage','sep':';','sizefile':'sizes/ivdsag_select2.csv'},'norm_soft_size',1)]" --lr_decay 0.9 --l_rate 1e-6 \


results/ivdsag/sfda_selectter: OPT = --target_losses="[('EntKLPropSelect', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredselect','power': 1, 'mode':'percentage','sep':';','sizefile':'sizes/ivdsag_select.csv'},'norm_soft_size',1)]" --lr_decay 0.2 --l_rate 3e-05 --do_asd True\

results/ivdsag/sfdanotag: OPT = --target_losses="[('EntKLProp', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbprednotags2','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/ivdsag.csv'},'norm_soft_size',1)]" --lr_decay 0.2 \

results/ivdsag/sfdaNew11: OPT = --target_losses="[('EntKLProp', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags6','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/ivdsag.csv'},'norm_soft_size',1)]" --lr_decay 0.9 --l_rate 5e-5 --lr_decay_epoch 10\
	--target_folders="$(TT_DATA)" --val_target_folders="$(TT_DATA)" \


results/ivdsag/sfdaNew18bis: OPT = --target_losses="[('EntKLProp', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags3','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/ivdsag.csv'},'norm_soft_size',1)]" --lr_decay 0.5 --l_rate 1e-6 --do_asd True\
	--target_folders="$(TT_DATA)" --val_target_folders="$(TT_DATA)" \


results/ivdsag/sfdaNew23: OPT = --target_losses="[('EntKLProp', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags2','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/ivdsag.csv'},'norm_soft_size',1)]" --lr_decay 0.9 --l_rate 1e-6 \
	--target_folders="$(TT_DATA)" --val_target_folders="$(TT_DATA)" \

results/ivdsag/sfdaNew22: OPT = --target_losses="[('EntKLProp', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags3','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/ivdsag.csv'},'norm_soft_size',1)]" --lr_decay 0.9 --l_rate 5e-7 \
	--target_folders="$(TT_DATA)" --val_target_folders="$(TT_DATA)" \

results/ivdsag/sfdaNew21: OPT = --target_losses="[('EntKLProp', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags3','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/ivdsag.csv'},'norm_soft_size',1)]" --lr_decay 0.7 --l_rate 5e-7 \
	--target_folders="$(TT_DATA)" --val_target_folders="$(TT_DATA)" \


results/ivdsag/sfdaNew20: OPT = --target_losses="[('EntKLProp', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags3','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/ivdsag.csv'},'norm_soft_size',1)]" --lr_decay 0.2 --l_rate 1e-6 \
	--target_folders="$(TT_DATA)" --val_target_folders="$(TT_DATA)" \


results/ivdsag/sfdaNew19: OPT = --target_losses="[('EntKLProp', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags2','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/ivdsag.csv'},'norm_soft_size',1)]" --lr_decay 0.5 --l_rate 1e-6 \
	--target_folders="$(TT_DATA)" --val_target_folders="$(TT_DATA)" \


results/ivdsag/sfdagtsize2: OPT = --target_losses="[('EntKLProp', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'val_gt_size','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/ivdsag.csv'},'norm_soft_size',1)]" --lr_decay 0.9 --l_rate 1e-5 \
	--target_folders="$(TT_DATA)" --val_target_folders="$(TT_DATA)" \


results/ivdsag/sfdaNew10: OPT = --target_losses="[('EntKLProp', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags5','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/ivdsag.csv'},'norm_soft_size',1)]" --lr_decay 0.9 --l_rate 1e-6 --lr_decay_epoch 10\
	--target_folders="$(TT_DATA)" --val_target_folders="$(TT_DATA)" \


results/ivdsag/sfdaNew9_ost: OPT = --target_losses="[('EntKLProp', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags4','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/ivdsag.csv'},'norm_soft_size',1)]"  --lr_decay 1 --n_epoch 300\


results/ivdsag/sfdaNew9aug: OPT = --target_losses="[('EntKLProp', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags4','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/ivdsag.csv'},'norm_soft_size',1)]"  --lr_decay 1 --n_epoch 300\


results/ivdsag/sfdaNew12: OPT = --target_losses="[('EntKLProp', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags4','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/ivdsag.csv'},'norm_soft_size',1)]" --lr_decay 0.8 --l_rate 3e-5 --lr_decay_epoch 10\
	--target_folders="$(TT_DATA)" --val_target_folders="$(TT_DATA)" \


results/ivdsag/sfdaNew8: OPT = --target_losses="[('EntKLProp', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags2','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/ivdsag.csv'},'norm_soft_size',1)]" --lr_decay 0.7 --l_rate 1e-6 --lr_decay_epoch 10\
	--target_folders="$(TT_DATA)" --val_target_folders="$(TT_DATA)" \

results/ivdsag/sfdaNew7: OPT = --target_losses="[('EntKLProp', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags2','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/ivdsag.csv'},'norm_soft_size',1)]" --lr_decay 0.8 \
	--target_folders="$(TT_DATA)" --val_target_folders="$(TT_DATA)" \

results/ivdsag/sfdaNew6: OPT = --target_losses="[('EntKLProp', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags2','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/ivdsag.csv'},'norm_soft_size',1)]" --lr_decay 0.7 \
	--target_folders="$(TT_DATA)" --val_target_folders="$(TT_DATA)" \

results/ivdsag/sfda2subj$(NBSUBJ): OPT = --train_case_nb=$(NBSUBJ) --target_losses="[('EntKLProp', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags2','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/ivdsag.csv'},'norm_soft_size',1)]" --lr_decay 0.2 \
	--target_folders="$(TT_DATA)" --val_target_folders="$(TT_DATA)" --n_epoch 300 \
          #--saveim True --entmap --do_asd 1 --do_hd 1  \

results/ivdsag/adaent4bis: OPT = --l_rate 5e-5 --target_losses="[('AdaEntKLProp', {'lamb_se':1, 'lamb_consprior':10,'ivd':True,'weights_se':[1,1],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags2','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/ivdsag.csv'},'norm_soft_size',1)]" --lr_decay 0.2 \
          --saveim True --entmap --do_asd 1 --do_hd 1  \


#inference mode : saves the segmentation masks + entropy masks for a specific model saved as pkl file (ex. "$(M_WEIGHTS_ul)" below):
results/ivdsag/cesourceim: OPT =  --target_losses="$(L_OR)" \
	   --mode makeim  --do_asd 1 --l_rate 0 --pprint --n_epoch 1 --saveim True --entmap \

results/ivdsag/fooim: OPT =  --target_losses="$(L_OR)" \
	   --mode makeim   --model_weights="results/ivdsag/sfda3/best_3d.pkl" --do_asd True --l_rate 0 --pprint --n_epoch 1 \

results/ivd/sfda_onsource: OPT =  --target_losses="$(L_OR)" \
	   --val_target_folders="$(S_DATA)"  --mode makeim   --l_rate 0 --pprint --n_epoch 1 --model_weights="results/ivd/sfda4/best_3d.pkl" \

results/ivd/cesource_onsource: OPT =  --target_losses="$(L_OR)" \
	   --val_target_folders="$(S_DATA)"  --mode makeim   --l_rate 0 --pprint --n_epoch 1 \

results/ivdsag/cdaim: OPT =  --target_losses="$(L_OR)" \
	   --val_target_folders="$(T_DATA)" --mode makeim  --l_rate 0 --pprint --n_epoch 1 --do_asd 1 --saveim True --entmap  --model_weights="/data/users/mathilde/ccnn/CDA/results/ivdsag/CSizewTag_tmp/best_3d.pkl" \


$(TRN) :
	$(CC) $(CFLAGS) main_sfda.py --batch_size 24 --n_class 2 --workdir $@_tmp --target_dataset "$(B_FOLD)"  \
                --grp_regex="$(G_RGX)"  --target_folders="$(TTrain_DATA)" --val_target_folders="$(TTest_DATA)"\
                --model_weights="$(M_WEIGHTS_ul)" --network=$(NET) \
                --lr_decay 0.9 --metric_axis 1  --n_epoch 150 --dice_3d --l_rate 1e-6 --lr_decay_epoch 20 --weight_decay 1e-4 $(OPT) $(DEBUG)\

	mv $@_tmp $@


