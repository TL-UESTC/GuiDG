d=0 
s=16
 
python phase1.py -d ImageNet --source 1 --targets 0 -b 32 --device $d --n-shot $s --lr 1e-4
python phase1.py -d ImageNet --source 4 --targets 0 -b 32 --device $d --n-shot $s --lr 1e-4
python phase1.py -d ImageNet --source 5 --targets 0 -b 32 --device $d --n-shot $s --lr 1e-4

python phase1.py -d ImageNet --source 1 --targets 2 -b 8 --device $d --n-shot $s --lr 1e-4
python phase1.py -d ImageNet --source 4 --targets 2 -b 8 --device $d --n-shot $s --lr 1e-4 
python phase1.py -d ImageNet --source 5 --targets 2 -b 8 --device $d --n-shot $s --lr 1e-4 

python phase1.py -d ImageNet --source 1 --targets 3 -b 32 --device $d --n-shot $s --lr 1e-4 
python phase1.py -d ImageNet --source 4 --targets 3 -b 32 --device $d --n-shot $s --lr 1e-4 
python phase1.py -d ImageNet --source 5 --targets 3 -b 32 --device $d --n-shot $s --lr 1e-4 