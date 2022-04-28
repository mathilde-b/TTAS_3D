CC = python
SHELL = bash
PP = PYTHONPATH="$(PYTHONPATH):."

.PHONY: all view plot report

CFLAGS = -O
#CFLAGS = -O -m pdb
#DEBUG = --debug

#the regex of the slices in the target dataset
#for the heart --
G_RGX = s

T_DATA = [('IMG', nii_transform, False), ('GT', nii_gt_transform, False), ('GT', nii_gt_transform, False)]
TT_DATA = [('IMG', nii_transform, False), ('GT', nii_gt_transform, False), ('GT', nii_gt_transform, False),('GT', nii_gt_transform, False)]
TTT_DATA = [('IMG', nii_transform, False), ('GT', nii_gt_transform, False), ('GT', nii_gt_transform, False),('GT', nii_gt_transform, False),('GT', nii_gt_transform, False)]
L_OR = [('CrossEntropy', {'idc': [0,1,2,3,4], 'weights':[1,1,1,1,1]}, None, None, None, 1)]


NET = UNet

# the folder containing the target dataset
S_FOLD = /data/users/mathilde/ccnn/CDA/data/mr_nii
#T_FOLD = /data/users/mathilde/ccnn/CDA/data/ct_nii


EPC=150
#run the main experiment
TRN = sourceresults/whs/ecc sourceresults/whs/dist_centroid sourceresults/whs/comp sourceresults/whs/dist_centroid_vertical
TRN = targetresults/whs/centroid_vertical
TRN = sourceresults/whs/nu sourceresults/whs/nu_vertical
TRN = sourceresults/whs/altcomp

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

sourceresults/whs/ecc: OPT =  --moment_fn soft_eccentricity \

sourceresults/whs/inertia: OPT =  --moment_fn soft_inertia \

sourceresults/whs/dist_centroid: OPT =  --moment_fn soft_dist_centroid \

sourceresults/whs/length: OPT =  --moment_fn soft_length \

sourceresults/whs/size: OPT =  --moment_fn soft_size \

sourceresults/whs/nu: OPT =  --moment_fn soft_nu \

sourceresults/whs/nu_vertical: OPT =  --moment_fn soft_nu --ind_mom 1\

sourceresults/whs/dist_centroid_vertical: OPT =  --moment_fn soft_dist_centroid --ind_mom 1\

targetresults/whs/centroid: OPT =  --moment_fn soft_centroid --target_dataset  "$(T_FOLD)" --ontest --testonly --train_grp_regex="c" --grp_regex="c" \

targetresults/whs/centroid_vertical: OPT =  --moment_fn soft_centroid --target_dataset  "$(T_FOLD)" --ontest --testonly --train_grp_regex="c" --grp_regex="c" --ind_mom 1\


sourceresults/whs/comp: OPT =  --moment_fn saml_compactness \

sourceresults/whs/altcomp: OPT =  --moment_fn soft_compactness \

$(TRN) :
	$(CC) $(CFLAGS) get_moments.py --target_losses="$(L_OR)" --regex_list "['ctslice81019','ctslice81003','ctslice81008','ctslice81014']" \
	--ontrain  --notent --batch_size 22 --n_class 5 --workdir $@_tmp --target_dataset  "$(S_FOLD)" --val_target_folders="$(T_DATA)" \
                --train_grp_regex="$(G_RGX)" --metric_axis 1 2 3 4  --grp_regex="$(G_RGX)"  \
                 --target_folders="$(T_DATA)" $(OPT) $(DEBUG)
	mv $@_tmp $@
	$(CC) tr_csv.py $@/moment.csv $@ 5 -1000 10


