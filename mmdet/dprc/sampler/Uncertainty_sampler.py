eps = 1e-10

@SAMPLER.register_module()
class DCUSSampler(BaseALSampler):
    def __init__(
        self,
        n_sample_images,
        oracle_annotation_path,
        score_thr,
        class_weight_ub,
        class_weight_alpha,
        dataset_type,
    ):
        super(DCUSSampler, self).__init__(
            n_sample_images,
            oracle_annotation_path,
            is_random=False,
            dataset_type=dataset_type)

        self.score_thr = score_thr
        self.class_weight_ub = class_weight_ub
        self.class_weight_alpha = class_weight_alpha
        self.log_init_info()
    def compute_iou(self,box1, box2):
        x1_1, y1_1, w1, h1 = box1
        x2_1 = x1_1 + w1
        y2_1 = y1_1 + h1
        x1_2, y1_2, w2, h2 = box2
        x2_2 = x1_2 + w2
        y2_2 = y1_2 + h2
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)
        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area != 0 else 0
    def al_acquisition(self, result_json, last_label_path):  
        with open(result_json) as f:
            results = json.load(f)
        category_uncertainty = OrderedDict()
        category_count = OrderedDict()
        for res in results:  
            img_id = res['image_id']
            img_size = (self.oracle_data[img_id]['image']['width'], self.oracle_data[img_id]['image']['height'])
            if not self.is_box_valid(res['bbox'],img_size):
                continue
            if res['score'] < self.score_thr:
                continue
            uncertainty = res['cls_uncertainty']
            label = res['category_id']
            if label not in category_uncertainty.keys():
                category_uncertainty[label] = 0.
                category_count[label] = 0.
            category_uncertainty[label] += uncertainty
            category_count[label] += 1
        category_avg_uncertainty = OrderedDict()
        for k in category_uncertainty.keys():
            category_avg_uncertainty[k] = category_uncertainty[k] / (category_count[k] + 1e-5)
        with open(last_label_path) as f:
            last_labeled_data = json.load(f)
            last_labeled_img_ids = [x['id'] for x in last_labeled_data['images']]
        image_hit = dict()
        for img_id in self.oracle_data.keys():
            image_hit[img_id] = 0
        for img_id in last_labeled_img_ids:
            image_hit[img_id] = 1
        image_uncertainties = OrderedDict()
        for img_id in self.oracle_data.keys():
            if image_hit[img_id] == 0:
                image_uncertainties[img_id] = [0.]
        selected_bboxes = {}
        for res in results:  
            img_id = res['image_id']
            img_size = (self.oracle_data[img_id]['image']['width'], self.oracle_data[img_id]['image']['height'])
            if not self.is_box_valid(res['bbox'], img_size):
                continue
            if res['score'] < self.score_thr:
                continue
            if img_id not in selected_bboxes:
                selected_bboxes[img_id] = []
            skip = False
            for prev_box in selected_bboxes[img_id]:
                if self.compute_iou(res['bbox'], prev_box) >= 0.5:
                    skip = True
                    break
                if skip:
                    continue
            uncertainty = res['cls_uncertainty']
            label = res['category_id']
            reverse_q = 1 - (res['classification_weights']/100) 
            b = np.exp(1. / 0.3) - 1
            anchore_weights = 1 + 0.3 * np.log(b * res['anchore_weights1'] + 1) * 0.2
            regression_weights = 1 + 0.3 * np.log(b * res['regression_weights'] + 1) * 0.2  
            regression_information = 1 + 0.3 * np.log(b * res['regression_information'] + 1) * 0.2
            classification_weights = 1 + 0.3 * np.log(b * reverse_q + 1) * 0.2
            classification_information = 1 + 0.3 * np.log(b * res['classification_information'] + 1) * 0.2
            image_uncertainties[img_id].append(anchore_weights* (regression_weights*regression_information+classification_weights*classification_information))
            selected_bboxes[img_id].append(res['bbox'])
        for img_id in image_uncertainties.keys():
            _img_uncertainties = image_uncertainties[img_id]
            image_uncertainties[img_id] = _img_uncertainties.sum()
        img_ids = []
        merged_img_uncertainties = []
        for k, v in image_uncertainties.items():
            img_ids.append(k)
            merged_img_uncertainties.append(v)
        img_ids = np.array(img_ids)
        merged_img_uncertainties = np.array(merged_img_uncertainties)
        inds_sort = np.argsort(-1. * merged_img_uncertainties)
        sampled_inds = inds_sort[:self.n_images]
        unsampled_img_ids = inds_sort[self.n_images:]
        sampled_img_ids = img_ids[sampled_inds].tolist()
        unsampled_img_ids = img_ids[unsampled_img_ids].tolist()
        return sampled_img_ids, unsampled_img_ids
    def al_round(self, result_path, last_label_path, out_label_path, out_unlabeled_path):
        sys_echo('\n\n>> Starting Active Learning Acquisition!!!')
        self.round += 1
        self.log_info(result_path, out_label_path, out_unlabeled_path)
        self.latest_labeled = last_label_path
        sampled_img_ids, rest_img_ids = self.al_acquisition(result_path, last_label_path)
        self.create_jsons(sampled_img_ids, rest_img_ids, last_label_path, out_label_path, out_unlabeled_path)
        sys_echo('>> Active Learning Acquisition Complete!!!\n\n')

    def create_jsons(self, sampled_img_ids, unsampled_img_ids, last_labeled_json, out_label_path, out_unlabeled_path):
        with open(last_labeled_json) as f:
            last_labeled_data = json.load(f)

        last_labeled_img_ids = [x['id'] for x in last_labeled_data['images']]
        all_labeled_img_ids = last_labeled_img_ids + sampled_img_ids
        assert len(set(all_labeled_img_ids)) == len(last_labeled_img_ids) + len(sampled_img_ids)
        assert len(all_labeled_img_ids) + len(unsampled_img_ids) == self.image_pool_size

        sys_echo('---------------------------------------------')
        sys_echo('--->>> Creating new image sets:')
        sys_echo('--->>> Last round labeled set size: %d (%.2f%%)' % (len(last_labeled_img_ids),100.*float(len(last_labeled_img_ids))/self.image_pool_size))
        sys_echo('--->>> New uncertainty pool set size: %d (%.2f%%)'%(len(sampled_img_ids),100.*float(len(sampled_img_ids))/self.image_pool_size))
        sys_echo('---------------------------------------------')

        labeled_data = dict(images=[], annotations=[], categories=self.categories)
        unlabeled_data = dict(images=[], categories=self.categories)
        for img_id in sampled_img_ids:
            # no annotation here because the annotating happens in the diversity step
            labeled_data['images'].append(self.oracle_data[img_id]['image'])

        with open(out_label_path, 'w') as f:
            json.dump(labeled_data, f)
        self.latest_labeled = out_label_path

    def log_info(self, result_path,  out_label_path, out_unlabeled_path):
        sys_echo('>>>> Round: %d' % self.round)
        sys_echo('>>>> Dataset: %s' % self.dataset_type)
        sys_echo('>>>> Oracle annotation path: %s' % self.oracle_path)
        sys_echo('>>>> Image pool size: %d' % self.image_pool_size)
        sys_echo('>>>> Uncertainty pool size per Round: %d (%.2f%%)'%(self.n_images, 100.*float(self.n_images)/self.image_pool_size))
        sys_echo('>>>> Unlabeled results path: %s' % result_path)
        sys_echo('>>>> Uncertainty pool image info file path: %s' % out_label_path)
        sys_echo('>>>> Score threshold: %s' % self.score_thr)
        sys_echo('>>>> Class weight upper bound: %.2f' % self.class_weight_ub)
        sys_echo('>>>> Class quality alpha: %.2f' % self.class_weight_alpha)

    def log_init_info(self):
        sys_echo('>> %s initialized:'%self.__class__.__name__)
        sys_echo('>>>> Dataset: %s' % self.dataset_type)
        sys_echo('>>>> Oracle annotation path: %s' % self.oracle_path)
        sys_echo('>>>> Image pool size: %d' % self.image_pool_size)
        sys_echo('>>>> Uncertainty pool size per round: %d (%.2f%%)'%(self.n_images, 100.*float(self.n_images)/self.image_pool_size))
        sys_echo('>>>> Score threshold: %s' % self.score_thr)
        sys_echo('>>>> Class weight upper bound: %.2f' % self.class_weight_ub)
        sys_echo('>>>> Class quality alpha: %.2f' % self.class_weight_alpha)
        sys_echo('\n')
