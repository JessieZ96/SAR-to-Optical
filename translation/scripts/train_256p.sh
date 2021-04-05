SCENE=${1}


set -ex
python train.py --name ${SCENE} --dataroot ./datasets/${SCENE} --label_nc 0 --no_instance