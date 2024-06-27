python -u main.py --dataset=CAMRa2011 --num_negatives=6 --num_enc_layers=1 --num_dec_layers=3 --sce_alpha=1 --drop_ratio=0.0 --epoch=30 --device=cuda:1

python -u main.py --dataset=Mafengwo --num_negatives=10 --num_enc_layers=2 --num_dec_layers=3 --sce_alpha=2 --drop_ratio=0.1 --epoch=200 --device=cuda:1

python -u main.py --dataset=MafengwoS --num_negatives=8 --num_enc_layers=3 --num_dec_layers=1 --sce_alpha=1 --drop_ratio=0.0 --epoch=200 --device=cuda:1

python -u main.py --dataset=MovieLens --num_negatives=4 --num_enc_layers=2 --num_dec_layers=1 --sce_alpha=2 --drop_ratio=0.0 --epoch=30 --device=cuda:1

python -u main.py --dataset=WeeplacesS --num_negatives=10 --num_enc_layers=1 --num_dec_layers=3 --sce_alpha=2 --drop_ratio=0.0 --epoch=30 --device=cuda:1