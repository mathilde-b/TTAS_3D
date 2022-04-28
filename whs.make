CC = python
SHELL = bash
PP = PYTHONPATH="$(PYTHONPATH):."

.PHONY: all view plot report

CFLAGS = -O
#CFLAGS = -m pdb
#DEBUG = --debug

#the regex of the slices in the target dataset
#for the heart --
G_RGX = ctslice\d+_\d+
G_RGX =

T_DATA = [('IMG', nii_transform, False), ('GT', nii_gt_transform, False), ('GT', nii_gt_transform, False)]
TTTT_DATA = [('IMG', nii_transform, False), ('GT', nii_gt_transform, False), ('GT', nii_gt_transform, False), ('GT', nii_gt_transform, False), ('GT', nii_gt_transform, False), ('GT', nii_gt_transform, False)]
L_OR = [('CrossEntropy', {'idc': [0,1,2,3,4], 'weights':[1,1,1,1,1]}, None, None, None, 1)]
LOSS=[('NaivePenalty', {'idc': [1,2,3,4]}, 'PreciseBounds', {'margin': 0.10, 'mode': 'percentage'}, 'soft_size', 1e-2), \
	('NaivePenalty', {'idc': [1,2,3,4]}, 'PreciseBounds', {'margin': 0.10, 'mode': 'abs'}, 'soft_centroid', 1e-2), \
	('NaivePenalty', {'idc': [1,2,3,4]}, 'PreciseBounds', {'margin': 0.10, 'mode': 'abs'}, 'soft_dist_centroid', 1e-2), \
	('NaivePenalty', {'idc': [1,2,3,4]}, 'PreciseBounds', {'margin': 0.10, 'mode': 'percentage'}, 'soft_length', 1e-2)] \

NET = UNet


# the folder containing the target dataset
T_FOLD = data/whs/ct
T_FOLD = /data/users/mathilde/ccnn/CDA/data/ct_nii

#the network weights used as initialization of the adaptation
M_WEIGHTS_ul = results/whs/cesource/last.pkl

#run the main experiment
TRN = results/whs/cesourceim
TRN = results/whs/enklsizecomp2
TRN = results/whs/enklsizecomptestonly
TRN = oneresults/whs/sfda1003


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
	    
# full supervision
results/whs/fs: OPT =  --target_losses="$(L_OR)" \
	     --network UNet --lr_decay 1 \
            --ontest --saveim True  --do_asd 1 --do_hd 1\

results/whs/4desc: OPT =  --target_losses="$(LOSS)" --target_folders="$(TTTT_DATA)" --val_target_folders="$(TTTT_DATA)" \
          --ontest --l_rate 0.000001 --lr_decay 0.9 --weight_decay 1e-3 \

results/whs/enklsizecomptestonly: OPT =  --target_losses="[('EntKLPropWComp', {'curi':True,'lamb_se':1,'lamb_comp':1,'lamb_consprior':1, 'ivd':False,'weights_se':[0.02, 0.27, 0.18, 0.21, 0.32],'idc_c': [1,2,3,4],'power': 1},'PredictionBounds', \
        {'margin':0,'dir':'high','idc':[1],'predcol':'dumbpredwtags', 'power': 1,'fake':False, 'mode':'percentage','prop':False,'sep':';','sizefile':'sizes/whs.csv'},'norm_soft_size',50)]"\
          --ontest --testonly --l_rate 0.000001 --lr_decay 0.9 --weight_decay 1e-3 \


# SFDA. Remove --saveim True --entmap --do_asd 1 --do_hd 1 to speed up
oneresults/whs/sfda1003: OPT = --tta --target_losses="[('EntKLProp', {'curi':True,'lamb_se':1,'lamb_consprior':1, 'ivd':False,'weights_se':[0.02, 0.27, 0.18, 0.21, 0.32],'idc_c': [1,2,3,4],'power': 1},'PredictionBounds', \
        {'margin':0,'dir':'high','idc':[1],'predcol':'dumbpredwtags', 'power': 1,'fake':False, 'mode':'percentage','prop':False,'sep':';','sizefile':'sizes/whs.csv'},'norm_soft_size',50)]"\
          --train_grp_regex="$(G_RGX)" --lr_decay_epoch 600 --n_epoch 1800 --train_case_nb 1 --specific_subj ctslice81003 --ontest --testonly --l_rate 0.000001 --lr_decay 0.7 --weight_decay 1e-3 \

results/whs/sfda_procutselect: OPT = --target_losses="[('EntKLPropSelect', {'curi':True,'lamb_se':1,'lamb_consprior':1, 'ivd':False,'weights_se':[0.02, 0.27, 0.18, 0.21, 0.32],'idc_c': [1,2,3,4],'power': 1},'PredictionBounds', \
        {'margin':0,'dir':'high','idc':[1,2,3,4],'predcol':'proposalselect', 'power': 1,'fake':False, 'mode':'percentage','prop':False,'sep':';','sizefile':'sizes/whs_select4.csv'},'norm_soft_size',50)]"\
          --ontest --l_rate 0.000001 --lr_decay 0.7 --weight_decay 1e-3 \

results/whs/sfda_proselect_th4_update: OPT = --batch_size 24 --target_losses="[('EntKLPropSelect', {'curi':True,'lamb_se':1,'lamb_consprior':1, 'ivd':False,'weights_se':[0.02, 0.27, 0.18, 0.21, 0.32],'idc_c': [1,2,3,4],'power': 1},'PredictionBounds', \
        {'margin':0,'dir':'high','idc':[1,2,3,4],'predcol':'dumbpredselect', 'power': 1,'fake':False, 'mode':'percentage','prop':False,'sep':';','sizefile':'sizes/whs_select_th4_prop50.csv'},'norm_soft_size',50)]"\
          --ontest --l_rate 0.000001 --lr_decay 0 --weight_decay 1e-3 \


results/whs/sfda_proselect_th4ter: OPT = --batch_size 24 --target_losses="[('EntKLPropSelect', {'curi':True,'lamb_se':1,'lamb_consprior':1, 'ivd':False,'weights_se':[0.02, 0.27, 0.18, 0.21, 0.32],'idc_c': [1,2,3,4],'power': 1},'PredictionBounds', \
        {'margin':0,'dir':'high','idc':[1,2,3,4],'predcol':'dumbpredselect', 'power': 1,'fake':False, 'mode':'percentage','prop':False,'sep':';','sizefile':'sizes/whs_select_th4.csv'},'norm_soft_size',50)]"\
          --ontest --l_rate 0.000001 --lr_decay 0 --weight_decay 1e-3 \


results/whs/sfda_proselect_th4_1019: OPT =  --target_losses="[('EntKLPropSelect', {'curi':True,'lamb_se':1,'lamb_consprior':1, 'ivd':False,'weights_se':[0.02, 0.27, 0.18, 0.21, 0.32],'idc_c': [1,2,3,4],'power': 1},'PredictionBounds', \
        {'margin':0,'dir':'high','idc':[1,2,3,4],'predcol':'dumbpredselect', 'power': 1,'fake':False, 'mode':'percentage','prop':False,'sep':';','sizefile':'sizes/whs_select_th4.csv'},'norm_soft_size',50)]"\
          --ontest --l_rate 0.000001 --lr_decay 0.4 --weight_decay 1e-3 --testonly --specific_subj ctslice81019 --tta --train_case_nb=1 --target_folders="$(TT_DATA)" --lr_decay_epoch 600 --n_epoch 1800\

results/whs/sfdanotag: OPT = --target_losses="[('EntKLPropNoTag', {'curi':True,'lamb_se':1,'lamb_consprior':1, 'ivd':False,'weights_se':[0.02, 0.27, 0.18, 0.21, 0.32],'idc_c': [1,2,3,4],'power': 1},'PredictionBounds', \
        {'margin':0,'dir':'high','idc':[1,2,3,4],'predcol':'dumbprednotags', 'power': 1,'fake':False, 'mode':'percentage','prop':False,'sep':';','sizefile':'sizes/whs.csv'},'norm_soft_size',50)]"\
          --ontest --l_rate 0.000001 --lr_decay 0.7 --weight_decay 1e-3 \

results/whs/sfdanotag_notrain1019_: OPT = --target_losses="[('EntKLPropSelect', {'curi':True,'lamb_se':1,'lamb_consprior':1, 'ivd':False,'weights_se':[0.02, 0.27, 0.18, 0.21, 0.32],'idc_c': [1,2,3,4],'power': 1},'PredictionBounds', \
        {'margin':0,'dir':'high','idc':[1,2,3,4],'predcol':'dumbpredselect', 'power': 1,'fake':False, 'mode':'percentage','prop':False,'sep':';','sizefile':'sizes/whs_select3.csv'},'norm_soft_size',50)]"\
          --ontest --l_rate 0.000001 --lr_decay 0.7 --weight_decay 1e-3 \
          --testonly --specific_subj ctslice81019 --tta --train_case_nb=1 --target_folders="$(TT_DATA)" --lr_decay_epoch 600 --n_epoch 1800\

results/whs/sfda_notrain1008: OPT = --target_losses="[('EntKLProp', {'curi':True,'lamb_se':1,'lamb_consprior':1, 'ivd':False,'weights_se':[0.02, 0.27, 0.18, 0.21, 0.32],'idc_c': [1,2,3,4],'power': 1},'PredictionBounds', \
        {'margin':0,'dir':'high','idc':[1],'predcol':'dumbpredwtags', 'power': 1,'fake':False, 'mode':'percentage','prop':False,'sep':';','sizefile':'sizes/whs.csv'},'norm_soft_size',50)]"\
          --ontest --l_rate 0.000001 --lr_decay 0.7 --weight_decay 1e-3 \
         --train_grp_regex="$(G_RGX)" --testonly --specific_subj ctslice81008 --tta --train_case_nb=1 --target_folders="$(TT_DATA)" --lr_decay_epoch 600 --n_epoch 1800\


#inference mode : saves the segmentation masks for a specific model saved as pkl file (ex. "results/whs/cesource/last.pkl" below):
results/whs/fooim: OPT = --ontest --target_losses="$(L_OR)" \
	   --do_asd True --mode makeim  --model_weights="results/whs/sfda_proselect_th4_update_tmp/best_3d.pkl" --l_rate 0  --pprint --lr_decay 1 --n_epoch 1\

$(TRN) :
	$(CC) $(CFLAGS) main_sfda.py --batch_size 24 --n_class 5 --workdir $@_tmp --target_dataset "$(T_FOLD)" \
                --metric_axis 1 2 3 4 --n_epoch 150 --dice_3d --l_rate 5e-4 --weight_decay 1e-4 --grp_regex="$(G_RGX)" --network=$(NET) --val_target_folders="$(T_DATA)"\
                  --lr_decay 0.9  --model_weights="$(M_WEIGHTS_ul)"  --target_folders="$(T_DATA)" $(OPT) $(DEBUG)
	mv $@_tmp $@


