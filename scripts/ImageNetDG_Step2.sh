d=0
s=16
u=clipood   #   baseline method


python phase2.py --root DomainBed/domainbed/data/ -d ImageNet --targets 0 -b 12 --device $d --n-shot $s --lr 5e-6 --lam 0.3 --beta 0.1  --baseline $u 
python phase2.py --root DomainBed/domainbed/data/ -d ImageNet --targets 2 -b 12 --device $d --n-shot $s --lr 5e-6 --lam 0.3 --beta 0.1  --baseline $u 
python phase2.py --root DomainBed/domainbed/data/ -d ImageNet --targets 3 -b 12 --device $d --n-shot $s --lr 5e-6 --lam 0.3 --beta 0.1  --baseline $u 