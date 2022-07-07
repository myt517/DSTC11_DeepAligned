from util import *
from keras.utils.np_utils import to_categorical

def onehot_labelling(int_labels, num_classes):
    categorical_labels = to_categorical(int_labels, num_classes=num_classes)
    return categorical_labels

def pair_cosine_similarity(x, x_adv, eps=1e-8):
    n = x.norm(p=2, dim=1, keepdim=True)
    n_adv = x_adv.norm(p=2, dim=1, keepdim=True)
    #print(x.shape)
    #print(x_adv.shape)
    #print(n.shape)
    #print(n_adv.shape)
    #print((n * n.t()).shape)
    return (x @ x.t()) / (n * n.t()).clamp(min=eps), (x_adv @ x_adv.t()) / (n_adv * n_adv.t()).clamp(min=eps), (x @ x_adv.t()) / (n * n_adv.t()).clamp(min=eps)

def nt_xent(x, x_adv, mask, cuda=True, t=0.5):
    x, x_adv, x_c = pair_cosine_similarity(x, x_adv)
    x = torch.exp(x / t)
    x_adv = torch.exp(x_adv / t)
    x_c = torch.exp(x_c / t)
    mask_count = mask.sum(1)
    mask_reverse = (~(mask.bool())).long()
    if cuda:
        dis = (x * (mask - torch.eye(x.size(0)).long().cuda()) + x_c * mask) / (x.sum(1) + x_c.sum(1) - torch.exp(torch.tensor(1 / t))) + mask_reverse
        dis_adv = (x_adv * (mask - torch.eye(x.size(0)).long().cuda()) + x_c.T * mask) / (x_adv.sum(1) + x_c.sum(0) - torch.exp(torch.tensor(1 / t))) + mask_reverse
    else:
        dis = (x * (mask - torch.eye(x.size(0)).long()) + x_c * mask) / (x.sum(1) + x_c.sum(1) - torch.exp(torch.tensor(1 / t))) + mask_reverse
        dis_adv = (x_adv * (mask - torch.eye(x.size(0)).long()) + x_c.T * mask) / (x_adv.sum(1) + x_c.sum(0) - torch.exp(torch.tensor(1 / t))) + mask_reverse
    loss = (torch.log(dis).sum(1) + torch.log(dis_adv).sum(1)) / mask_count
    #loss = dis.sum(1) / (x.sum(1) + x_c.sum(1) - torch.exp(torch.tensor(1 / t))) + dis_adv.sum(1) / (x_adv.sum(1) + x_c.sum(0) - torch.exp(torch.tensor(1 / t)))
    return -loss.mean()
    #return -torch.log(loss).mean()

        
class BertForModel(BertPreTrainedModel):
    def __init__(self,config,num_labels):
        super(BertForModel, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        #self.rnn = nn.GRU(input_size=config.hidden_size, hidden_size=config.hidden_size, num_layers=1,
        #                  dropout=config.hidden_dropout_prob, batch_first=True, bidirectional=True)
        #self.dense = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids = None, token_type_ids = None, attention_mask=None , labels = None, mode = None, centroids = None, labeled = False, feature_ext = False):

        encoded_layer_12, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers = False)
        #_, pooled_output = self.rnn(encoded_layer_12)
        #pooled_output = torch.cat((pooled_output[0].squeeze(0), pooled_output[1].squeeze(0)), dim=1)
        #pooled_output = self.dense(pooled_output)

        pooled_output = self.dense(encoded_layer_12.mean(dim = 1))
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)        
        logits = self.classifier(pooled_output)


        if feature_ext:
            return pooled_output
        elif mode == 'train':
            loss = nn.CrossEntropyLoss()(logits, labels)

            # 监督对比学习损失
            label_ids = labels.cpu()
            labels = onehot_labelling(label_ids, self.num_labels)
            labels = torch.from_numpy(labels)
            labels = labels.cuda()
            label_mask = torch.mm(labels, labels.T).bool().long()
            encoded_layer_12_02, pooled_output_02 = self.bert(input_ids, token_type_ids, attention_mask,
                                                        output_all_encoded_layers=False)
            pooled_output_02 = self.dense(encoded_layer_12_02.mean(dim = 1))
            pooled_output_02 = self.activation(pooled_output_02)
            pooled_output_02 = self.dropout(pooled_output_02)
            sup_cont_loss = nt_xent(pooled_output, pooled_output_02, label_mask, cuda=True)

            loss = loss + sup_cont_loss

            return loss
        else:
            return pooled_output, logits
