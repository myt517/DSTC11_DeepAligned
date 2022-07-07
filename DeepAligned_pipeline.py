from model import *
from init_parameter import *
from dataloader import *
from pretrain import *
from util import *
from torch.nn.functional import normalize
from kmeans import *
from classifier import *
import datetime

class ModelManager:

    def __init__(self, args, data, pretrained_model=None):

        if pretrained_model is None:
            pretrained_model = BertForModel.from_pretrained(args.bert_model, num_labels=data.n_known_cls)
            if os.path.exists(args.pretrain_dir):
                pretrained_model = self.restore_model(args, pretrained_model)
        self.pretrained_model = pretrained_model

        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pretrained_model.to(self.device)
        # if args.cluster_num_factor > 1:
        #    self.num_labels = self.predict_k(args, data)
        # else:
        #    self.num_labels = data.num_labels
        self.num_labels = data.n_unknown_cls
        print(self.num_labels)
        # self.num_labels = self.predict_k(args, data)
        self.model = BertForModel.from_pretrained(args.bert_model, num_labels=self.num_labels)
        # trainable_count = np.sum([K.count_params(w) for w in self.model.trainable_weights])

        if args.pretrain:
            self.load_pretrained_model(args)

        if args.freeze_bert_parameters:
            self.freeze_parameters(self.model)

        # total = sum([param.nelement() for param in self.model.parameters() if param.requires_grad])
        # print("Number of parameter: % .2fM" % (total / 1e6))
        # exit()

        self.model.to(self.device)

        num_train_examples = len(data.train_unlabeled_examples)
        self.num_train_optimization_steps = int(num_train_examples / args.train_batch_size) * args.num_train_epochs

        self.optimizer = self.get_optimizer(args)

        print(self.model)
        self.best_eval_score = 0
        self.training_SC_epochs = {}
        self.centroids = None

        self.test_results = {}
        self.predictions = None
        self.true_labels = None

    def load_models(self, args):
        print("loading models ....")
        self.model = self.restore_model_v2(args, self.model)

    def get_features_labels(self, dataloader, model, args):

        model.eval()
        total_features = torch.empty((0, args.feat_dim)).to(self.device)
        total_labels = torch.empty(0, dtype=torch.long).to(self.device)

        for batch in tqdm(dataloader, desc="Extracting representation"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.no_grad():
                feature = model(input_ids, segment_ids, input_mask, feature_ext=True)

            total_features = torch.cat((total_features, feature))
            total_labels = torch.cat((total_labels, label_ids))

        return total_features, total_labels

    def predict_k(self, args, data):

        feats, _ = self.get_features_labels(data.train_unlabeled_dataloader, self.pretrained_model, args)
        feats = feats.cpu().numpy()
        km = KMeans(n_clusters=data.num_labels).fit(feats)
        y_pred = km.labels_

        pred_label_list = np.unique(y_pred)
        drop_out = len(feats) / data.num_labels
        print('drop', drop_out)

        cnt = 0
        for label in pred_label_list:
            num = len(y_pred[y_pred == label])
            if num < drop_out:
                cnt += 1

        num_labels = len(pred_label_list) - cnt
        print('pred_num', num_labels)

        return num_labels

    def get_optimizer(self, args):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.lr,
                             warmup=args.warmup_proportion,
                             t_total=self.num_train_optimization_steps)
        return optimizer

    def tsne_visualization_2(self, x, y, predicted, args):
        label_list = [50, 51, 52, 53, 54, 55, 56, 57, 58]
        path = args.save_results_path
        # TSNE_visualization(x, y, label_list, os.path.join(path, "pca_train_b2.png"+str(args.seed)))
        TSNE_visualization(x, y, label_list, os.path.join(path, "pca_train_" + str(args.seed) + ".png"))

    def evaluation(self, args, data):
        eval_dataloader = data.train_unlabeled_dataloader
        feats, labels = self.get_features_labels(eval_dataloader, self.model, args)
        feats = feats.cpu().numpy()
        km = KMeans(n_clusters=self.num_labels).fit(feats)

        y_pred = km.labels_
        y_true = labels.cpu().numpy()

        results = clustering_score(y_true, y_pred)
        print('results', results)

        #ind, _ = hungray_aligment(y_true, y_pred)
        #map_ = {i[0]: i[1] for i in ind}
        #y_pred = np.array([map_[idx] for idx in y_pred])

        #cm = confusion_matrix(y_true, y_pred)
        #print('confusion matrix', cm)
        self.test_results = results

        #self.save_results(args)

        return results,y_pred

    def visualize_training(self, args, data):
        feats, labels = self.get_features_labels(data.train_unlabeled_dataloader, self.model, args)
        # feats = normalize(feats, dim=1)
        feats = feats.cpu().numpy()
        km = KMeans(n_clusters=self.num_labels).fit(feats)

        y_pred = km.labels_
        y_true = labels.cpu().numpy()
        score = metrics.silhouette_score(feats, km.labels_)

        results = clustering_score(y_true, y_pred)
        results["SC"] = score

        self.test_results = results

        self.tsne_visualization_2(feats, y_true, y_pred, args)

    def alignment(self, km, args):

        if self.centroids is not None:

            old_centroids = self.centroids.cpu().numpy()
            new_centroids = km.cluster_centers_

            DistanceMatrix = np.linalg.norm(old_centroids[:, np.newaxis, :] - new_centroids[np.newaxis, :, :], axis=2)
            row_ind, col_ind = linear_sum_assignment(DistanceMatrix)

            new_centroids = torch.tensor(new_centroids).to(self.device)
            self.centroids = torch.empty(self.num_labels, args.feat_dim).to(self.device)

            alignment_labels = list(col_ind)
            for i in range(self.num_labels):
                label = alignment_labels[i]
                self.centroids[i] = new_centroids[label]

            pseudo2label = {label: i for i, label in enumerate(alignment_labels)}
            pseudo_labels = np.array([pseudo2label[label] for label in km.labels_])

        else:
            self.centroids = torch.tensor(km.cluster_centers_).to(self.device)
            pseudo_labels = km.labels_

        pseudo_labels = torch.tensor(pseudo_labels, dtype=torch.long).to(self.device)

        return pseudo_labels

    def update_pseudo_labels(self, pseudo_labels, args, data):
        train_data = TensorDataset(data.semi_input_ids, data.semi_input_mask, data.semi_segment_ids, pseudo_labels)
        train_sampler = SequentialSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
        return train_dataloader

    def training_process_eval(self, args, data, epoch):
        self.model.eval()

        feats, labels = self.get_features_labels(data.train_unlabeled_dataloader, self.model, args)
        feats = feats.cpu().numpy()
        labels = labels.cpu().numpy()
        km = KMeans(n_clusters=self.num_labels).fit(feats)
        score = metrics.silhouette_score(feats, km.labels_)

        # results = clustering_score(labels, km.labels_)

        self.training_SC_epochs["epoch:" + str(epoch)] = score

        return score

    def train(self, args, data):

        best_score = 0
        best_model = None
        wait = 0
        e_step = 0
        best_epoch = 0

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):

            feats, labels = self.get_features_labels(data.train_unlabeled_dataloader, self.model, args)
            feats = feats.cpu().numpy()
            labels = labels.cpu().numpy()
            km = KMeans(n_clusters=self.num_labels).fit(feats)

            # SC_score = self.training_process_eval(args, data, e_step)
            # e_step += 1
            # print(SC_score)

            score = metrics.silhouette_score(feats, km.labels_)
            results = clustering_score(labels, km.labels_)
            self.training_SC_epochs["epoch:" + str(epoch)] = results["NMI"]
            print(results["ACC"])
            print('score', score)

            if score > best_score:
                best_model = copy.deepcopy(self.model)
                wait = 0
                best_score = score
                best_epoch = epoch
            else:
                wait += 1
                if wait >= args.wait_patient:
                    self.model = best_model
                    break

            pseudo_labels = self.alignment(km, args)
            # pseudo_labels = km.labels_
            # pseudo_labels = torch.tensor(pseudo_labels, dtype=torch.long).to(self.device)
            train_dataloader = self.update_pseudo_labels(pseudo_labels, args, data)

            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            self.model.train()

            for batch in tqdm(train_dataloader, desc="Pseudo-Training"):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                loss = self.model(input_ids, segment_ids, input_mask, label_ids, mode='train')

                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                self.optimizer.step()
                self.optimizer.zero_grad()

            tr_loss = tr_loss / nb_tr_steps
            print('train_loss', tr_loss)

        if args.save_model:
            self.save_model(args)

        return best_epoch

    def load_pretrained_model(self, args):
        pretrained_dict = self.pretrained_model.state_dict()
        classifier_params = ['classifier.weight', 'classifier.bias']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in classifier_params}
        self.model.load_state_dict(pretrained_dict, strict=False)

    def load_final_model(self, args):
        pretrained_dict = self.pretrained_model.state_dict()
        classifier_params = ['classifier.weight', 'classifier.bias']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in classifier_params}
        self.model.load_state_dict(pretrained_dict, strict=False)

    def restore_model(self, args, model):
        output_model_file = os.path.join(args.pretrain_dir, WEIGHTS_NAME)
        model.load_state_dict(torch.load(output_model_file))
        return model

    def restore_model_v2(self, args, model):
        output_model_file = os.path.join(args.model_dir, WEIGHTS_NAME)
        model.load_state_dict(torch.load(output_model_file))
        return model

    def freeze_parameters(self, model):
        for name, param in model.bert.named_parameters():
            param.requires_grad = False
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True

    def save_model(self, args):
        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)
        self.save_model = self.model.module if hasattr(self.model, 'module') else self.model
        model_file = os.path.join(args.model_dir, WEIGHTS_NAME)
        model_config_file = os.path.join(args.model_dir, CONFIG_NAME)
        torch.save(self.save_model.state_dict(), model_file)
        with open(model_config_file, "w") as f:
            f.write(self.save_model.config.to_json_string())

    def save_results(self, args):
        if not os.path.exists(args.save_results_path):
            os.makedirs(args.save_results_path)

        var = [args.dataset, args.method, args.known_cls_ratio, args.IND_ratio, args.labeled_ratio,
               args.cluster_num_factor, args.seed, self.num_labels]
        names = ['dataset', 'method', 'known_cls_ratio', 'IND_ratio', 'labeled_ratio', 'cluster_num_factor', 'seed',
                 'K']
        vars_dict = {k: v for k, v in zip(names, var)}
        results = dict(self.test_results, **vars_dict)
        keys = list(results.keys())
        values = list(results.values())

        file_name = 'results_INDnoise_1.csv'
        results_path = os.path.join(args.save_results_path, file_name)

        if not os.path.exists(results_path):
            ori = []
            ori.append(values)
            df1 = pd.DataFrame(ori, columns=keys)
            df1.to_csv(results_path, index=False)
        else:
            df1 = pd.read_csv(results_path)
            new = pd.DataFrame(results, index=[1])
            df1 = df1.append(new, ignore_index=True)
            df1.to_csv(results_path, index=False)
        data_diagram = pd.read_csv(results_path)

        # self.save_training_process(args)

        print('test_results', data_diagram)

    def save_training_process(self, args):
        if not os.path.exists(args.save_results_path):
            os.makedirs(args.save_results_path)

        results = dict(self.training_SC_epochs)
        keys = list(results.keys())
        values = list(results.values())

        file_name = 'results_analysis_V3_11_trainigEpoch.csv'
        results_path = os.path.join(args.save_results_path, file_name)

        if not os.path.exists(results_path):
            ori = []
            ori.append(values)
            df1 = pd.DataFrame(ori, columns=keys)
            df1.to_csv(results_path, index=False)
        else:
            df1 = pd.read_csv(results_path)
            new = pd.DataFrame(results, index=[1])
            df1 = df1.append(new, ignore_index=True)
            df1.to_csv(results_path, index=False)
        data_diagram = pd.read_csv(results_path)

        print('training_process_dynamic:', data_diagram)


if __name__ == '__main__':

    print('Data and Parameters Initialization...')
    parser = init_model()
    args = parser.parse_args()
    data = Data(args)

    if args.method == "classify":
        data.change_data(args)
        manager_new = classifier_all(args, data)
        #manager_new.train(args, data)
        manager_new.evaluation(args, data)
        # manager.visualize_training(args, data)
        manager_new.save_results(args)
        exit()

    if args.pretrain:
        print('Pre-training begin...')
        #manager_p = PretrainModelManager(args, data)
        #manager_p.train(args, data)
        #exit()
        print('Pre-training finished!')
        manager = ModelManager(args, data)
    else:
        manager = ModelManager(args, data)

    print('Training begin...')
    #time1 = datetime.datetime.now()
    #epoch = manager.train(args, data)
    #time2 = datetime.datetime.now()
    manager.load_models(args)
    print('Training finished!')
    #print(time2 - time1)
    #print("best epoch:", epoch)
    #exit()

    print('Evaluation begin...')
    results, y_pred = manager.evaluation(args, data)
    #manager.visualize_training(args, data)
    print('Evaluation finished!')
    data.gen_data(y_pred.tolist(), args)
    #data.change_data(args)
    manager_new = classifier_all(args, data)
    manager_new.train(args, data)
    manager_new.evaluation(args, data)
    # manager.visualize_training(args, data)
    manager_new.save_results(args)
