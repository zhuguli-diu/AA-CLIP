declare -a dataset=(MVTec BTAD MPDD Brain Liver Retina Colon_clinicDB Colon_colonDB Colon_Kvasir Colon_cvc300)
save_path="ckpt/shot0-1"
for i in "${dataset[@]}"; do
    python test.py --save_path $save_path --dataset $i
done