import sys
import os
import json
import numpy as np

if __name__ == '__main__':
    print(sys.argv)
    output_dir = sys.argv[1]
    # f = open(os.path.join(output_dir, 'out.txt'), 'a')
    # sys.stdout = f
    csv_list = []
    avg_jga = []
    n_seed_runs = 0
    log_jgas = []

    for order in [1,2,3,4,5]:
        # res_path = os.path.join(output_dir.replace('seed1', 'seed{}'.format(seed)), 'FINAL/result.txt')
        res_path = os.path.join(output_dir.replace('order1', 'order{}'.format(order)), 'FINAL/result.txt')
        if os.path.exists(res_path):
            n_seed_runs += 1
            with open(res_path) as f:
                result_one_seed = []
                log_domains = []
                for lineid, l in enumerate(f.readlines()[2:]):
                    results = l.split('|')
                    domain = results[1].strip()
                    log_domains.append(domain)
                    if lineid == 0:
                        jga = float(results[2].strip())
                    else:
                        results[2] = results[2].replace("'", '"')
                        jga_dict = json.loads(results[2].strip())
                        jga = jga_dict['turn_level_joint_acc']
                    result_one_seed.append(round(jga*100, 2))
                log_jgas.append(log_domains)
                log_jgas.append(result_one_seed)

    # avg_res = ['%.2f(%.2f)' % (np.mean(jga), np.std(jga)) for jga in zip(*log_jgas)]
    csv_list += log_jgas
    # csv_list.append(avg_res)
    from tabulate import tabulate
    print(tabulate(csv_list))
    import csv
    with open('gather_res.csv', 'w') as csvfile:
        csv_w = csv.writer(csvfile)
        for line_w in csv_list:
            csv_w.writerow(line_w)