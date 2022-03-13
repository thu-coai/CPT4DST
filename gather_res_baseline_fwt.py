import sys
import os
import json
import numpy as np
import glob

if __name__ == '__main__':
    print(sys.argv)
    first_output_dir = sys.argv[1]
    first_output_dir = os.path.join(os.getcwd(), first_output_dir)
    # print(output_dir)
    csv_list = []
    for order in [1,2,3,4,5]:
        output_dir = first_output_dir.replace('order1', 'order{}'.format(order))

        task_dirs = list(glob.glob(os.path.join(output_dir+'/*/')))
        # print(task_dirs)
        # "10.29/t5_vanilla_baseline_seed1_order1/9_['sgd_hotels_4']/"
        task_dirs = [_ for _ in task_dirs if 'FINAL' not in _]
        tasks = [_.split('/')[-2] for _ in task_dirs]  # 9_['sgd_hotels_4']
        tasks = [_.split('_') for _ in tasks]
        tasks = [[_[0], '_'.join(_[1:])] for _ in tasks]
        tasks = [_ for _ in tasks if _[0].isdigit()]


        for i in range(15):
            tasks[i].append(task_dirs[i])

        task_id_to_num_path = {v[1]: [v[0], v[2]] for v in tasks}
        task_list = [0 for _ in range(15)]
        for task_id in task_id_to_num_path:
            task_num = task_id_to_num_path[task_id][0]
            task_list[int(task_num)] = task_id

        # task_list = sorted(task_list)

        jga_list = []
        jga_matrix = []

        for trained_task in task_list:
            trained_task_num, trained_task_path = task_id_to_num_path[trained_task]
            # print(trained_task, trained_task_path)
            res_path = os.path.join(trained_task_path, 'result.txt')
            with open(res_path) as f:
                lines = f.readlines()[3:]
                test_res_list = [0 for _ in range(15)]
                for l in lines:
                    results = l.split('|')
                    test_domain = results[1].strip()
                    test_domain_idx = task_list.index(test_domain)
                    results[2] = results[2].replace("'", '"')
                    jga_dict = json.loads(results[2].strip())
                    jga = jga_dict['turn_level_joint_acc']
                    test_res_list[test_domain_idx] = round(jga*100, 2)
                # print(test_res_list)
                jga_matrix.append([trained_task] + test_res_list)

        task_list = [''] + task_list
        seed_csv_list = [task_list] + jga_matrix
        # csv_list += seed_csv_list
        res_for_fw = []
        for i in range(1, 15):
            res_for_fw.append(seed_csv_list[i][i])
        task_list = task_list[1:]
        csv_list += [task_list, res_for_fw]


    from tabulate import tabulate
    print(tabulate(csv_list))
    import csv
    with open('gather_res.csv', 'w') as csvfile:
        csv_w = csv.writer(csvfile)
        for line_w in csv_list:
            csv_w.writerow(line_w)