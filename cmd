python explain.py --image img/church.JPEG --method lime
python explain.py --image img/church.JPEG --method pyramid
python evaluate.py curves --image img/church.JPEG --out eval_mean --del-fill mean
