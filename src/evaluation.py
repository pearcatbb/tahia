import torch
import util_funcs as uf
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, average_precision_score, \
    confusion_matrix, classification_report, precision_score

def torch_f1_score(pred, target, n_class):
    '''
    Returns macro-f1 and micro-f1 score
    Args:
        pred:
        target:
        n_class:

    Returns:
        ma_f1,mi_f1: numpy values of macro-f1 and micro-f1 scores.
    '''

    def true_positive(pred, target, n_class):
        return torch.tensor([((pred == i) & (target == i)).sum()
                             for i in range(n_class)])

    def false_positive(pred, target, n_class):
        return torch.tensor([((pred == i) & (target != i)).sum()
                             for i in range(n_class)])

    def false_negative(pred, target, n_class):
        return torch.tensor([((pred != i) & (target == i)).sum()
                             for i in range(n_class)])

    def precision(tp, fp):
        res = tp / (tp + fp)
        res[torch.isnan(res)] = 0
        return res

    def recall(tp, fn):
        res = tp / (tp + fn)
        res[torch.isnan(res)] = 0
        return res

    def f1_score(prec, rec):
        f1_score = 2 * (prec * rec) / (prec + rec)
        f1_score[torch.isnan(f1_score)] = 0
        return f1_score

    def cal_maf1(tp, fp, fn):
        prec = precision(tp, fp)
        rec = recall(tp, fn)
        ma_f1 = f1_score(prec, rec)
        return torch.mean(ma_f1).cpu().numpy()

    def cal_mif1(tp, fp, fn):
        gl_tp, gl_fp, gl_fn = torch.sum(tp), torch.sum(fp), torch.sum(fn)
        gl_prec = precision(gl_tp, gl_fp)
        gl_rec = recall(gl_tp, gl_fn)
        mi_f1 = f1_score(gl_prec, gl_rec)
        return mi_f1.cpu().numpy()

    tp = true_positive(pred, target, n_class).to(torch.float)
    fn = false_negative(pred, target, n_class).to(torch.float)
    fp = false_positive(pred, target, n_class).to(torch.float)
    
    ma_f1, mi_f1 = cal_maf1(tp, fp, fn), cal_mif1(tp, fp, fn)
    return ma_f1, mi_f1


def eval_logits(logits, target_x, target_y):
    pred_y = torch.argmax(logits[target_x], dim=1)
    
    # auc = roc_auc_score(target_y.cpu(), pred_y.cpu().numpy())
    auc = roc_auc_score(target_y.cpu(), torch.softmax(logits[target_x], dim=1).detach().cpu().numpy()[:,1])
    print(" AUC : " + str(auc))
    # print(classification_report(target_y.cpu(), pred_y.cpu().numpy(), digits=4))
    trec = recall_score(target_y.cpu(), pred_y.cpu().numpy(), average='macro')
    tpre = precision_score(target_y.cpu(), pred_y.cpu().numpy(), average='macro')
    acc = accuracy_score(target_y.cpu(), pred_y.cpu().numpy())
    ma_f1, mi_f1 = torch_f1_score(pred_y, target_y, n_class=logits.shape[1])
    return auc, trec, tpre, acc, ma_f1, mi_f1


def eval_and_save(cf, logits, test_x, test_y, val_x, val_y, stopper=None, res={}):
    auc, trec, tpre, acc, test_f1, test_mif1 = eval_logits(logits, test_x, test_y)
    auc, trec, tpre, acc, val_f1, val_mif1 = eval_logits(logits, val_x, val_y)
    save_results(cf, test_f1, val_f1, test_mif1, val_mif1, stopper, auc, trec, tpre, acc, res)


def save_results(cf, test_f1, val_f1, test_mif1=0, val_mif1=0, stopper=None, auc=0, trec=0, tpre=0, acc=0, res={}):
    if stopper != None:
        res.update({'test_f1': f'{test_f1:.4f}', 'test_mif1': f'{test_mif1:.4f}',
                    'val_f1': f'{val_f1:.4f}', 'val_mif1': f'{val_mif1:.4f}',
                    'best_epoch': stopper.best_epoch, 'Auc': f'{auc:.4f}', 'Recall': f'{trec:.4f}', 'Precision': f'{tpre:.4f}', 'Accuracy': f'{acc:.4f}'})
    else:
        res.update({'test_f1': f'{test_f1:.4f}', 'test_mif1': f'{test_mif1:.4f}',
                    'val_f1': f'{val_f1:.4f}', 'val_mif1': f'{val_mif1:.4f}','Auc': f'{auc:.4f}', 'Recall': f'{trec:.4f}', 'Precision': f'{tpre:.4f}', 'Accuracy': f'{acc:.4f}'})
    print(f"Seed{cf.seed}")
    res_dict = {'res': res, 'parameters': cf.get_model_conf()}
    print(f'\n\n\nTrain finished, results:{res_dict}')
    uf.write_nested_dict(res_dict, cf.res_file)
