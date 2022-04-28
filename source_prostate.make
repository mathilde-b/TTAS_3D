CC = python
SHELL = bash
PP = PYTHONPATH="$(PYTHONPATH):."

.PHONY: all view plot report

CFLAGS = -O

#the regex of the slices in the target dataset
#for the prostate
G_RGX = C

S_DATA = [('IMG', nii_transform, False), ('GT', nii_gt_transform, False), ('GT', nii_gt_transform, False)]
TT_DATA = [('IMG', nii_transform, False), ('GTNew', nii_gt_transform, False), ('GTNew', nii_gt_transform, False)]
L_OR = [('CrossEntropy', {'idc': [0,1], 'weights':[1,1]}, None, None, None, 1)]
NET = UNet

# the folder containing the target dataset - site A is the target dataset and site B is the source one
T_FOLD = /data/users/mathilde/ccnn/SAML/data/SA/

TRN = sourceresults/prostate/dist_centroid sourceresults/prostate/comp sourceresults/prostate/dist_centroid_vertical sourceresults/prostate/ecc
TRN = sourceresults/prostate/nu_vertical

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

sourceresults/prostate/ecc: OPT =  --moment_fn soft_eccentricity \

sourceresults/prostate/dist_centroid: OPT =  --moment_fn soft_dist_centroid \

sourceresults/prostate/dist_centroid_vertical: OPT =  --moment_fn soft_dist_centroid --ind_mom 1\

sourceresults/prostate/comp: OPT =  --moment_fn saml_compactness \

sourceresults/prostate/length: OPT =  --moment_fn soft_length \

sourceresults/prostate/size: OPT =  --moment_fn soft_size \

sourceresults/prostate/nu_vertical: OPT =  --moment_fn soft_nu  --ind_mom 1\

$(TRN) :
	$(CC) $(CFLAGS) get_moments.py --target_dataset "/data/users/mathilde/ccnn/SAML/data/SB" --val_target_folders="$(S_DATA)" --target_folders="$(S_DATA)" --valonly  --target_losses="$(L_OR)" --ontest --notent --regex_list "['Case26','Case05','Case02','Case07','Case08','Case12','Case15','Case17','Case20','Case22']" --batch_size 24 --n_class 2 --workdir $@_tmp --target_dataset "$(T_FOLD)" \
                --wh 384 --metric_axis 1  --n_epoch 151 --dice_3d --l_rate 5e-4 --weight_decay 1e-4 --train_grp_regex="$(G_RGX)" --grp_regex="$(G_RGX)" --val_target_folders="$(TT_DATA)"\
                     --lr_decay 0.9   --target_folders="$(TT_DATA)" $(OPT) $(DEBUG)
	mv $@_tmp $@
	$(CC) tr_csv.py $@/moment.csv $@ 2 -1000 10


