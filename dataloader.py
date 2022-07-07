from util import *
import os

IND_1=[["credit_cards", "travel", "home", "meta", "utility", "small_talk", "auto_and_commute", "work"],["credit_cards", "travel", "home", "meta", "utility", "small_talk", "kitchen_and_dining", "banking"],["credit_cards", "meta", "utility", "small_talk", "kitchen_and_dining", "banking", "work", "auto_and_commute"]]
OOD_1=[["kitchen_and_dining", "banking"],["work", "auto_and_commute"],["travel","home"]]


IND_2=[["credit_cards", "meta", "utility", "small_talk", "work", "auto_and_commute"],["credit_cards", "meta", "utility", "small_talk", "kitchen_and_dining", "banking"],["travel", "home", "meta", "utility", "small_talk", "kitchen_and_dining"]]
OOD_2=[["travel", "home", "kitchen_and_dining", "banking"],["travel", "home", "auto_and_commute", "work"],["credit_cards", "banking", "auto_and_commute", "work"]]


IND_3=[["credit_cards", "meta", "utility", "small_talk"]]
OOD_3=[["travel", "home", "kitchen_and_dining", "banking","auto_and_commute","work"]]




def set_seed(seed, seed2=0):
    random.seed(seed)
    np.random.seed(seed2)
    torch.manual_seed(seed)
    
class Data:
    
    def __init__(self, args):
        set_seed(args.seed, args.seed2)
        max_seq_lengths = {'clinc':30,'banking':55, 'snips':35, "HWU64":25}
        args.max_seq_length = max_seq_lengths[args.dataset]

        processor = DatasetProcessor()
        self.data_dir = os.path.join(args.data_dir, args.dataset)
        self.all_label_list = processor.get_labels(self.data_dir)
        print("num_all_labels",len(self.all_label_list))
        self.n_known_cls = round(len(self.all_label_list) * args.known_cls_ratio)
        self.n_unknown_cls = len(self.all_label_list) - self.n_known_cls
        self.known_label_list = list(np.random.choice(np.array(self.all_label_list), self.n_known_cls, replace=False))
        self.unknown_label_list = list(set(self.all_label_list).difference(set(self.known_label_list)))

        oos_examples = []

        if args.mode == "cross_domain":
            if args.OOD_class == "none":
                if args.known_cls_ratio == 0.8:
                    self.known_label_list, self.unknown_label_list = cross_domain_division(
                        IND_domains=IND_1[args.type],
                        OOD_domain=OOD_1[args.type]
                        # OOD_domain=[args.OOD_class]
                    )
                if args.known_cls_ratio == 0.6:
                    self.known_label_list, self.unknown_label_list = cross_domain_division(
                        IND_domains=IND_2[args.type],
                        OOD_domain=OOD_2[args.type]
                        # OOD_domain=[args.OOD_class]
                    )
                if args.known_cls_ratio == 0.4:
                    self.known_label_list, self.unknown_label_list = cross_domain_division(
                        IND_domains=IND_3[args.type],
                        OOD_domain=OOD_3[args.type]
                        # OOD_domain=[args.OOD_class]
                    )


            else:
                self.known_label_list, self.unknown_label_list = cross_domain_division(
                    IND_domains=["credit_cards", "meta", "utility", "home", "small_talk", "work"],
                    #OOD_domain=["kitchen_and_dining", "banking", "auto_and_commute", "travel"]
                    OOD_domain=[args.OOD_class]
                )
            #self.unknown_label_list = list(set(self.all_label_list).difference(set(self.known_label_list)))
        elif args.mode == "noise_ood":
            self.oos_train = get_oos(ratio=0.8)
            print("selected oos num:", len(self.oos_train))

            for (i, line) in enumerate(self.oos_train):
                if i == 0:
                    continue
                if len(line) != 2:
                    continue
                guid = "%s-%s" % ("oos", i)
                text_a = line[0]
                label = line[1]

                oos_examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        elif args.mode == "IND_noise":
            self.train_noise_selected = self.get_examples(processor, args, 'ind_noise_0.1')
            for example in self.train_noise_selected:
                example.label = "oos"
                oos_examples.append(example)
        elif args.mode == "imbalanced":
            #self.OOD_list_ranking = get_imbalanced(self.unknown_label_list)
            self.train_OOD_selected = self.get_examples(processor, args, 'imbalanced') # 读取不平衡OOD数据


        self.n_known_cls = len(self.known_label_list)
        self.n_unknown_cls = len(self.unknown_label_list)
        print("num_IND_labels", self.n_known_cls, len(self.known_label_list))
        print("num_OOD_labels", self.n_unknown_cls, len(self.unknown_label_list))
        self.num_labels = len(self.unknown_label_list)*3
        print(self.unknown_label_list)
        for k in range(len(self.unknown_label_list)):
            print(self.unknown_label_list[k])

        if args.IND_ratio!=1.0:
            self.n_known_cls = round(len(self.known_label_list) * args.IND_ratio)
            self.known_label_list = list(np.random.choice(np.array(self.known_label_list), self.n_known_cls, replace=False))
            print("revised: the numbers of IND labels:", len(self.known_label_list), self.n_known_cls)

        self.all_label_list = []
        self.all_label_list.extend(self.known_label_list)
        self.all_label_list.extend(self.unknown_label_list)


        self.train_labeled_examples, self.train_unlabeled_examples, self.train_all_examples = self.get_examples(processor, args, 'train')
        self.train_unlabeled_examples.extend(oos_examples)
        self.train_all_examples = []
        self.train_all_examples.extend(self.train_labeled_examples)
        self.train_all_examples.extend(self.train_unlabeled_examples)
        random.shuffle(self.train_all_examples)

        print('train_num_labeled_samples',len(self.train_labeled_examples))
        print('train_num_unlabeled_samples',len(self.train_unlabeled_examples))
        print('train_num_all_samples', len(self.train_all_examples))
        self.eval_labeled_examples, self.eval_unlabeled_examples, self.eval_all_examples = self.get_examples(processor, args, 'eval')
        print('eval_num_labeled_samples', len(self.eval_labeled_examples))
        print('eval_num_unlabeled_samples', len(self.eval_unlabeled_examples))
        print('eval_num_all_samples', len(self.eval_all_examples))
        self.test_labeled_examples, self.test_unlabeled_examples, self.test_all_examples = self.get_examples(processor, args, 'test')
        print('test_num_labeled_samples', len(self.test_labeled_examples))
        print('test_num_unlabeled_samples', len(self.test_unlabeled_examples))
        print('test_num_all_samples', len(self.test_all_examples))


        self.train_labeled_dataloader = self.get_loader(self.train_labeled_examples, self.known_label_list, args,
                                                        'train')
        self.train_unlabeled_dataloader = self.get_loader(self.train_unlabeled_examples, self.unknown_label_list, args,
                                                        'train')
        self.train_all_dataloader = self.get_loader(self.train_all_examples, self.all_label_list, args,
                                                          'train')

        self.eval_labeled_dataloader = self.get_loader(self.eval_labeled_examples, self.known_label_list, args,
                                                       'eval')
        self.eval_unlabeled_dataloader = self.get_loader(self.eval_unlabeled_examples, self.unknown_label_list, args,
                                                         'eval')
        self.eval_all_dataloader = self.get_loader(self.eval_all_examples, self.all_label_list, args,
                                                         'eval')

        self.test_labeled_dataloader = self.get_loader(self.test_labeled_examples, self.known_label_list, args,
                                                       'test')
        self.test_unlabeled_dataloader = self.get_loader(self.test_unlabeled_examples, self.unknown_label_list, args,
                                                         'test')
        self.test_all_dataloader = self.get_loader(self.test_all_examples, self.all_label_list, args,
                                                         'test')

        self.semi_input_ids, self.semi_input_mask, self.semi_segment_ids, self.semi_label_ids = self.get_unlabelled(
            self.train_unlabeled_examples, args)

        #self.train_semi_dataloader = self.get_semi_loader(self.semi_input_ids, self.semi_input_mask,
        #                                                  self.semi_segment_ids, self.semi_label_ids, args)

        #self.semi_input_ids, self.semi_input_mask, self.semi_segment_ids, self.semi_label_ids = self.get_semi(self.train_labeled_examples, self.train_unlabeled_examples, args)
        #print(self.semi_input_ids.shape)
        #self.train_semi_dataloader = self.get_semi_loader(self.semi_input_ids, self.semi_input_mask, self.semi_segment_ids, self.semi_label_ids, args)

        #self.eval_dataloader = self.get_loader(self.eval_examples, args, 'eval')
        #self.test_dataloader = self.get_loader(self.test_examples, args, 'test')

    def get_examples(self, processor, args, mode = 'train'):
        ori_examples = processor.get_examples(self.data_dir, mode)

        if mode == "imbalanced":
            return ori_examples

        if mode == "IND_0.4" or mode == "IND_0.2":
            return ori_examples

        if mode == "ind_noise_0.05" or mode == "ind_noise_0.1":
            return ori_examples
        
        if mode == 'train':
            train_labels = np.array([example.label for example in ori_examples])
            train_labeled_ids = []
            for label in self.known_label_list:
                num = round(len(train_labels[train_labels == label]) * args.labeled_ratio)
                pos = list(np.where(train_labels == label)[0])                
                train_labeled_ids.extend(random.sample(pos, num))

            train_labeled_examples, train_unlabeled_examples, train_all_examples = [], [], []
            for idx, example in enumerate(ori_examples):
                if idx in train_labeled_ids:
                    train_labeled_examples.append(example)
                    train_all_examples.append(example)
                elif example.label in self.unknown_label_list:
                    train_unlabeled_examples.append(example)
                    train_all_examples.append(example)

            return train_labeled_examples, train_unlabeled_examples, train_all_examples

        elif mode == 'eval':
            eval_labels = np.array([example.label for example in ori_examples])
            eval_labeled_ids = []
            for label in self.known_label_list:
                num = round(len(eval_labels[eval_labels == label]) * 1)
                pos = list(np.where(eval_labels == label)[0])
                eval_labeled_ids.extend(random.sample(pos, num))

            eval_labeled_examples, eval_unlabeled_examples, eval_all_examples = [], [], []
            for idx, example in enumerate(ori_examples):
                if idx in eval_labeled_ids:
                    eval_labeled_examples.append(example)
                    eval_all_examples.append(example)
                elif example.label in self.unknown_label_list:
                    eval_unlabeled_examples.append(example)
                    eval_all_examples.append(example)

            return eval_labeled_examples, eval_unlabeled_examples, eval_all_examples

        elif mode == 'test':
            test_labels = np.array([example.label for example in ori_examples])
            test_labeled_ids = []
            for label in self.known_label_list:
                num = round(len(test_labels[test_labels == label]) * 1)
                pos = list(np.where(test_labels == label)[0])
                test_labeled_ids.extend(random.sample(pos, num))

            test_labeled_examples, test_unlabeled_examples, test_all_examples = [], [], []
            for idx, example in enumerate(ori_examples):
                if idx in test_labeled_ids:
                    test_labeled_examples.append(example)
                    test_all_examples.append(example)
                elif example.label in self.unknown_label_list:
                    test_unlabeled_examples.append(example)
                    test_all_examples.append(example)

            return test_labeled_examples, test_unlabeled_examples, test_all_examples


        return ori_examples

    def get_semi(self, labeled_examples, unlabeled_examples, args):

        print("train_num_example",len(labeled_examples)+len(unlabeled_examples))
        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)
        labeled_features = convert_examples_to_features(labeled_examples, self.known_label_list, args.max_seq_length,
                                                        tokenizer)
        unlabeled_features = convert_examples_to_features(unlabeled_examples, self.all_label_list, args.max_seq_length,
                                                          tokenizer)

        labeled_input_ids = torch.tensor([f.input_ids for f in labeled_features], dtype=torch.long)
        labeled_input_mask = torch.tensor([f.input_mask for f in labeled_features], dtype=torch.long)
        labeled_segment_ids = torch.tensor([f.segment_ids for f in labeled_features], dtype=torch.long)
        labeled_label_ids = torch.tensor([f.label_id for f in labeled_features], dtype=torch.long)

        unlabeled_input_ids = torch.tensor([f.input_ids for f in unlabeled_features], dtype=torch.long)
        unlabeled_input_mask = torch.tensor([f.input_mask for f in unlabeled_features], dtype=torch.long)
        unlabeled_segment_ids = torch.tensor([f.segment_ids for f in unlabeled_features], dtype=torch.long)
        unlabeled_label_ids = torch.tensor([-1 for f in unlabeled_features], dtype=torch.long)

        semi_input_ids = torch.cat([labeled_input_ids, unlabeled_input_ids])
        semi_input_mask = torch.cat([labeled_input_mask, unlabeled_input_mask])
        semi_segment_ids = torch.cat([labeled_segment_ids, unlabeled_segment_ids])
        semi_label_ids = torch.cat([labeled_label_ids, unlabeled_label_ids])
        return semi_input_ids, semi_input_mask, semi_segment_ids, semi_label_ids

    def get_unlabelled(self, unlabeled_examples, args):
        
        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)    
        #labeled_features = convert_examples_to_features(labeled_examples, self.known_label_list, args.max_seq_length, tokenizer)
        unlabeled_features = convert_examples_to_features(unlabeled_examples, self.unknown_label_list, args.max_seq_length, tokenizer)

        '''
        labeled_input_ids = torch.tensor([f.input_ids for f in labeled_features], dtype=torch.long)
        labeled_input_mask = torch.tensor([f.input_mask for f in labeled_features], dtype=torch.long)
        labeled_segment_ids = torch.tensor([f.segment_ids for f in labeled_features], dtype=torch.long)
        labeled_label_ids = torch.tensor([f.label_id for f in labeled_features], dtype=torch.long)      
        '''
        unlabeled_input_ids = torch.tensor([f.input_ids for f in unlabeled_features], dtype=torch.long)
        unlabeled_input_mask = torch.tensor([f.input_mask for f in unlabeled_features], dtype=torch.long)
        unlabeled_segment_ids = torch.tensor([f.segment_ids for f in unlabeled_features], dtype=torch.long)
        unlabeled_label_ids = torch.tensor([-1 for f in unlabeled_features], dtype=torch.long)

        '''
        semi_input_ids = torch.cat([labeled_input_ids, unlabeled_input_ids])
        semi_input_mask = torch.cat([labeled_input_mask, unlabeled_input_mask])
        semi_segment_ids = torch.cat([labeled_segment_ids, unlabeled_segment_ids])
        semi_label_ids = torch.cat([labeled_label_ids, unlabeled_label_ids])
        '''
        return unlabeled_input_ids, unlabeled_input_mask, unlabeled_segment_ids, unlabeled_label_ids

    def get_loader_all(self, examples,label_list,args,mode="train"):
        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)
        features = convert_examples_to_features(examples, label_list, args.max_seq_length,tokenizer)
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        data = TensorDataset(input_ids, input_mask, segment_ids, label_ids)


        if mode == 'train':
            sampler = RandomSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=args.train_batch_size)
        else:
            sampler = SequentialSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=args.eval_batch_size)

        return dataloader

    def change_data(self, args):

        # 得到不划分的train set
        train_all_examples = []
        train_all_examples.extend(self.train_labeled_examples)
        train_all_examples.extend(self.train_unlabeled_examples)

        self.train_all_dataloader = self.get_loader_all(train_all_examples, self.all_label_list, args, "train")

        eval_all_examples = []
        eval_all_examples.extend(self.eval_labeled_examples)
        eval_all_examples.extend(self.eval_unlabeled_examples)

        self.eval_all_dataloader = self.get_loader_all(eval_all_examples, self.all_label_list, args, "test")

        test_all_examples = []
        test_all_examples.extend(self.test_labeled_examples)
        test_all_examples.extend(self.test_unlabeled_examples)

        self.test_all_dataloader = self.get_loader_all(test_all_examples, self.all_label_list, args, "test")


    def gen_data(self, label_list, args):
        print(max(label_list))
        print('gen_data...lable list')
        print(len(label_list))
        train_unknown_label_lists = []
        unknown_label_list = [str(i) for i in range(self.n_unknown_cls)]
        for index in label_list:
            train_unknown_label_lists.append(str(index))

        print('Generate new dataset...')
        print('train_unlabeled_examples.train_x')
        print(len(self.train_unlabeled_examples))
        print(len(train_unknown_label_lists))
        # 得到新的train set

        for i in range(len(self.train_unlabeled_examples)):
            self.train_unlabeled_examples[i].label = train_unknown_label_lists[i]


        train_all_examples = []
        train_all_examples.extend(self.train_labeled_examples)
        train_all_examples.extend(self.train_unlabeled_examples)

        label_list=[]
        label_list.extend(self.known_label_list)
        label_list.extend(unknown_label_list)

        self.train_all_dataloader = self.get_loader_all(train_all_examples, label_list, args, "train")

        eval_all_examples = []
        eval_all_examples.extend(self.eval_labeled_examples)
        eval_all_examples.extend(self.eval_unlabeled_examples)

        self.eval_all_dataloader = self.get_loader_all(eval_all_examples, self.all_label_list, args, "test")

        test_all_examples = []
        test_all_examples.extend(self.test_labeled_examples)
        test_all_examples.extend(self.test_unlabeled_examples)

        self.test_all_dataloader = self.get_loader_all(test_all_examples, self.all_label_list, args, "test")

        '''
        content_list_train = self.train_labeled_examples.train_x + self.train_unlabeled_examples.train_x
        labels_list_train = self.train_labeled_examples.train_y + train_unknown_label_lists
        self.train_all_examples = OriginSamples(content_list_train, labels_list_train)
        self.train_all_dataloader = self.get_loader_all(self.train_all_examples, self.all_label_list, args, "train")

        # 得到新的eval set
        self.eval_all_examples = self.get_samples(self.get_datasets(self.data_dir, 'eval'), args, "eval")
        self.eval_all_dataloader = self.get_loader_all(self.eval_all_examples, self.all_label_list, args, "eval")

        # 得到新的test set
        self.test_all_examples = self.get_samples(self.get_datasets(self.data_dir, 'test'), args, "test")
        self.test_all_dataloader = self.get_loader_all(self.test_all_examples, self.all_label_list, args, "test")
        '''

    def get_semi_loader(self, semi_input_ids, semi_input_mask, semi_segment_ids, semi_label_ids, args):
        semi_data = TensorDataset(semi_input_ids, semi_input_mask, semi_segment_ids, semi_label_ids)
        semi_sampler = SequentialSampler(semi_data)
        semi_dataloader = DataLoader(semi_data, sampler=semi_sampler, batch_size = args.train_batch_size) 

        return semi_dataloader


    def get_loader(self, examples, label_list, args, mode = 'train'):
        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)    
        
        if mode == 'train' or mode == 'eval':
            features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer)
        elif mode == 'test':
            features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer)

        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        data = TensorDataset(input_ids, input_mask, segment_ids, label_ids)
        
        if mode == 'train':
            sampler = SequentialSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size = args.train_batch_size)    
        elif mode == 'eval' or mode == 'test':
            sampler = SequentialSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size = args.eval_batch_size) 
        
        return dataloader


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                '''
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                '''
                lines.append(line)
            return lines

class DatasetProcessor(DataProcessor):

    def get_examples(self, data_dir, mode):
        if mode == 'train':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
        elif mode == 'eval':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev.tsv")), "train")
        elif mode == 'test':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")
        elif mode == "imbalanced":
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "imbalanced_ood_6.tsv")), "imbalanced")
        elif mode == "IND_0.8":
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "IND_0.8.tsv")), "IND")
        elif mode == "IND_0.6":
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "IND_0.6.tsv")), "IND")
        elif mode == "IND_0.4":
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "IND_0.4.tsv")), "IND")
        elif mode == "IND_0.2":
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "IND_0.2.tsv")), "IND")
        elif mode == "ind_noise_0.05":
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "ind_noise_0.05.tsv")), "noise")
        elif mode == "ind_noise_0.1":
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "ind_noise_0.1.tsv")), "noise")
        elif mode == "ind_noise_0.15":
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "ind_noise_0.15.tsv")), "noise")

    def get_labels(self, data_dir):
        """See base class."""
        import pandas as pd
        test = pd.read_csv(os.path.join(data_dir, "train.tsv"), sep="\t")
        labels = np.unique(np.array(test['label']))
            
        return labels

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            if len(line) != 2:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    label_map = {}
    for i, label in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if example.label == "oos":
            label_id = 149
        else:
            label_id = label_map[example.label]

        #label_id = label_map[example.label]

        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join(
        #         [str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info(
        #         "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop(0)  # For dialogue context
        else:
            tokens_b.pop()
