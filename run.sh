# for voc
# python main_formula_TokenCut.py --dataset VOC07 --set trainval --arch vit_base
# python main_formula_TokenCut.py --dataset VOC12 --set trainval --arch vit_base

# python main_formula_LOST.py --dataset VOC07 --set trainval 
# python main_formula_LOST.py --dataset VOC12 --set trainval

# for coco
python main_formula_TokenCut.py --dataset COCO20k --set train --arch vit_base

# for coco
python main_formula_LOST.py --dataset COCO20k --set train