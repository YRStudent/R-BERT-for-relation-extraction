import jsonlines
from win32con import PRINTER_FONTTYPE

labelled_file = 'YuOpera Dataset/test_data.jsonl'
pred_labelled_file = 'pred_labelled_data_spo.jsonl'

def evaluate_metrics(pred_file,true_file):
    total_f1 = 0
    total_recall = 0
    total_precision = 0
    count = 0
    true_data = []
    total_pred = []
    total_true = []

    with open(pred_file, encoding='utf-8') as pf,open(true_file, encoding='utf-8') as tf:
        pred_reader = jsonlines.Reader(pf)
        true_reader = jsonlines.Reader(tf)

        total_f1 = 0
        total_recall = 0
        total_precision = 0
        total_count = 0


        for line_pred, line_true in zip(pred_reader,true_reader):

            spo_pred = line_pred['spo']
            total_pred.append(spo_pred)

            true_e1_pos = line_true['h']['pos']
            true_e2_pos = line_true['t']['pos']
            true_e1 = line_true['text'][true_e1_pos[0]:true_e1_pos[1]]
            true_e2 = line_true['text'][true_e2_pos[0]:true_e2_pos[1]]

            true_rel = line_true['relation']
            spo_true = [true_e1, true_e2, true_rel]
            #print(spo_true)
            total_true.append(spo_true)

        T = set(t for t in spo_true)
        print(T)
        P = set(tuple(spo) for spo in spo_pred)
        Intersection_len = len(T & P)
        f1 = (2 * Intersection_len) / (len(spo_pred) + len(spo_true))
        recall = Intersection_len / len(spo_true)
        precision = Intersection_len / len(spo_pred)




        '''for content in true_reader:
            true_data.append(content)

        for line in pred_reader:
            text_pred = line['text']
            #print(text_pred)
            spo_pred = line['spo']

            for content in true_data:
                text_true = content['text']
                #print(text_true)
                spo_true = content['spo']
                
                
                if text_true == text_pred:
                    T = set(tuple(spo) for spo in spo_true)
                    P = set(tuple(spo) for spo in spo_pred)
                    Intersection_len = len(T & P)
                    f1 = (2 * Intersection_len) / (len(spo_pred) + len(spo_true))
                    recall = Intersection_len/len(spo_true)
                    precision = Intersection_len/len(spo_pred)
                    #print(f1,recall,precision)

                    count += 1
                    total_f1 += f1
                    total_recall += recall
                    total_precision += precision'''

            #true_reader = jsonlines.Reader(tf)


    '''F1 = total_f1 / count
    Recall = total_recall / count
    Precision = total_precision / count
    print(count)
    return F1, Recall, Precision'''

F1, recall, precision = evaluate_metrics(pred_labelled_file, labelled_file)
print(F1, recall, precision)

