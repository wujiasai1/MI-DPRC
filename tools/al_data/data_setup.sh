### 以前代码 
# VOCPATH=$1
# mkdir -p data/VOC0712
# mkdir -p data/VOC0712/annotations
# mkdir -p data/VOC0712/images
# cp data/VOCdevkit/VOC2007/JPEGImages/*.jpg data/VOC0712/images
# cp data/VOCdevkit/VOC2012/JPEGImages/*.jpg data/VOC0712/images
# cp $VOCPATH data/VOC0712/annotations

# mkdir -p data/active_learning/coco/
# mkdir -p data/active_learning/voc/

# # python tools/al_data/create_al_dataset.py \
# #        --oracle-path data/coco/annotations/instances_train2017.json \
# #        --out-root data/active_learning/coco \
# #        --n-diff 3 \
# #        --n-labeled 2365 \
# #        --dataset coco

# # python tools/al_data/create_al_dataset.py \
# #        --oracle-path data/VOC0712/annotations/trainval_0712.json \
# #        --out-root data/active_learning/voc \
# #        --n-diff 3 \
# #        --n-labeled 827 \
# #        --dataset voc


# python tools/al_data/create_al_dataset.py \
#        data/coco/annotations/instances_train2017.json \
#        data/active_learning/coco \
#        3 \
#        2365 \
#        coco

# python tools/al_data/create_al_dataset.py \
#        data/VOC0712/annotations/trainval_0712.json \
#        data/active_learning/voc \
#        3 \
#        827 \
#        voc
### 以前代码 


### 现在代码，获取更少的未标注样本，减少推理时间
VOCPATH=$1
mkdir -p data/VOC0712
mkdir -p data/VOC0712/annotations
mkdir -p data/VOC0712/images
cp data/VOCdevkit/VOC2007/JPEGImages/*.jpg data/VOC0712/images
cp data/VOCdevkit/VOC2012/JPEGImages/*.jpg data/VOC0712/images
cp $VOCPATH data/VOC0712/annotations

mkdir -p data/active_learning/coco/
mkdir -p data/active_learning/voc/

python tools/al_data/create_al_dataset.py \
       data/coco/annotations/instances_train2017.json \
       data/active_learning/coco \
       3 \
       2365 \
       coco

python tools/al_data/create_al_dataset.py \
       data/VOC0712/annotations/trainval_0712.json \
       data/active_learning/voc \
       3 \
       16541 \
       voc
### 现在代码，获取更少的未标注样本，减少推理时间，10个标注