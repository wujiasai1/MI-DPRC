config_dir  = 'configs/active_learning/'
work_dir    = 'work_dirs/'
python_path = 'python'
port        = 29500
gpus        = 1
oracle_path         = 'data/VOC0712/annotations/trainval_0712.json'
init_label_json     = 'data/active_learning/voc/voc_827_labeled.json'
init_unlabeled_json = 'data/active_learning/voc/voc_827_unlabeled.json'
init_model          = None

train_config             = config_dir + 'al_train/retinanet.py'
uncertainty_infer_config = config_dir + 'al_inference/retinanet_uncertainty.py'
diversity_infer_config   = config_dir + 'al_inference/retinanet_diversity.py'

round_num             = 7
budget                = 414  
budget_expand_ratio   = 4
uncertainty_pool_size = budget * budget_expand_ratio + gpus - (budget * budget_expand_ratio) % gpus

uncertainty_sampler_config = dict(
    type='DCUSSampler',
    n_sample_images=uncertainty_pool_size,
    oracle_annotation_path=oracle_path,
    score_thr=0.05,
    class_weight_ub=0.2,
    class_weight_alpha=0.3,
    dataset_type='voc')
diversity_sampler_config = dict(
    type='DiversitySampler',
    n_sample_images=budget,
    oracle_annotation_path=oracle_path,
    dataset_type='voc')

output_dir  = work_dir + 'retinanet_voc_7rounds_5percent_to_20percent'