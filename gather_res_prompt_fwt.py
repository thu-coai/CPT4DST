import sys
import os
import json
import numpy as np
from collections import OrderedDict


def get_dataset_order(output_dir):
    # f = open(os.path.join(output_dir, 'out.txt'), 'a')
    # sys.stdout = f
    log_domains = ['avg_joint_acc']

    if 'order1_' in output_dir:
        dataset_order = ["['sgd_services_4']", "['sgd_flights_1']", "['sgd_services_3']",
                         "['sgd_flights_3']", "['sgd_trains_1']", "['sgd_homes_2']", "['sgd_rentalcars_2']",
                         "['sgd_restaurants_1']", "['sgd_music_1']", "['sgd_hotels_4']", "['sgd_media_2']",
                         "['sgd_hotels_3']", "['sgd_rentalcars_3']", "['sgd_hotels_1']", "['sgd_homes_1']"]
    elif 'order2_' in output_dir:
        dataset_order = ["['sgd_hotels_4']", "['sgd_flights_3']", "['sgd_rentalcars_2']", "['sgd_rentalcars_3']",
                         "['sgd_media_2']", "['sgd_restaurants_1']", "['sgd_music_1']", "['sgd_trains_1']",
                         "['sgd_services_3']", "['sgd_homes_2']", "['sgd_hotels_3']", "['sgd_flights_1']",
                         "['sgd_services_4']", "['sgd_homes_1']", "['sgd_hotels_1']"]
    elif 'order3_' in output_dir:
        dataset_order = ["['sgd_services_4']", "['sgd_hotels_3']", "['sgd_music_1']", "['sgd_flights_1']",
                         "['sgd_hotels_1']", "['sgd_hotels_4']", "['sgd_media_2']", "['sgd_flights_3']",
                         "['sgd_trains_1']", "['sgd_homes_1']", "['sgd_restaurants_1']", "['sgd_rentalcars_2']",
                         "['sgd_services_3']", "['sgd_homes_2']", "['sgd_rentalcars_3']"]
    elif 'order4_' in output_dir:
        dataset_order = ["['sgd_hotels_1']", "['sgd_media_2']", "['sgd_homes_1']", "['sgd_music_1']",
                         "['sgd_services_4']", "['sgd_restaurants_1']", "['sgd_flights_1']", "['sgd_hotels_4']",
                         "['sgd_services_3']", "['sgd_homes_2']", "['sgd_hotels_3']", "['sgd_trains_1']",
                         "['sgd_flights_3']", "['sgd_rentalcars_2']", "['sgd_rentalcars_3']"]
    elif 'order5_' in output_dir:
        dataset_order = ["['sgd_services_4']", "['sgd_flights_3']", "['sgd_homes_1']", "['sgd_flights_1']",
                         "['sgd_music_1']", "['sgd_services_3']", "['sgd_rentalcars_3']", "['sgd_media_2']",
                         "['sgd_restaurants_1']", "['sgd_hotels_1']", "['sgd_rentalcars_2']", "['sgd_hotels_4']",
                         "['sgd_hotels_3']", "['sgd_homes_2']", "['sgd_trains_1']"]
    elif 'order6_' in output_dir:
        dataset_order = ["['sgd_restaurants_1']", "['sgd_services_3']", "['sgd_flights_1']", "['sgd_trains_1']",
                         "['sgd_hotels_1']", "['sgd_services_4']", "['sgd_hotels_3']", "['sgd_rentalcars_2']",
                         "['sgd_flights_3']", "['sgd_hotels_4']", "['sgd_homes_2']", "['sgd_homes_1']",
                         "['sgd_rentalcars_3']", "['sgd_media_2']", "['sgd_music_1']"]

    elif 'order99' in output_dir:
        # debug
        dataset_order = ["['sgd_hotels_4']", "['sgd_trains_1']"]

    elif 'order30' in output_dir:
        dataset_order = ["['sgd_events_3']", "['sgd_banks_2']", "['sgd_banks_1']", "['sgd_calendar_1']",
                         "['sgd_movies_3']", "['sgd_music_2']", "['sgd_services_2']", "['sgd_payment_1']",
                         "['sgd_media_1']", "['sgd_weather_1']", "['sgd_events_1']", "['sgd_flights_4']",
                         "['sgd_travel_1']", "['sgd_buses_2']", "['sgd_events_2']", "['sgd_alarm_1']",
                         "['sgd_buses_3']", "['sgd_services_1']", "['sgd_buses_1']", "['sgd_restaurants_2']",
                         "['sgd_hotels_2']", "['sgd_ridesharing_2']", "['sgd_rentalcars_1']", "['sgd_movies_1']",
                         "['sgd_ridesharing_1']", "['sgd_media_3']", "['sgd_music_3']", "['sgd_movies_2']",
                         "['sgd_flights_2']", "['sgd_services_4']", "['sgd_flights_1']", "['sgd_services_3']",
                         "['sgd_flights_3']", "['sgd_trains_1']", "['sgd_homes_2']", "['sgd_rentalcars_2']",
                         "['sgd_restaurants_1']", "['sgd_music_1']", "['sgd_hotels_4']", "['sgd_media_2']",
                         "['sgd_hotels_3']", "['sgd_rentalcars_3']", "['sgd_hotels_1']", "['sgd_homes_1']"]
        dataset_order = dataset_order[-5:]
    elif 'order31' in output_dir:
        dataset_order = ["['sgd_events_3']", "['sgd_banks_2']", "['sgd_banks_1']", "['sgd_calendar_1']",
                         "['sgd_movies_3']", "['sgd_music_2']", "['sgd_services_2']", "['sgd_payment_1']",
                         "['sgd_media_1']", "['sgd_weather_1']", "['sgd_events_1']", "['sgd_flights_4']",
                         "['sgd_travel_1']", "['sgd_buses_2']", "['sgd_events_2']", "['sgd_alarm_1']",
                         "['sgd_buses_3']", "['sgd_services_1']", "['sgd_buses_1']", "['sgd_restaurants_2']",
                         "['sgd_hotels_2']", "['sgd_ridesharing_2']", "['sgd_rentalcars_1']", "['sgd_movies_1']",
                         "['sgd_ridesharing_1']", "['sgd_media_3']", "['sgd_music_3']", "['sgd_movies_2']",
                         "['sgd_flights_2']", "['sgd_services_4']", "['sgd_flights_1']", "['sgd_services_3']",
                         "['sgd_flights_3']", "['sgd_trains_1']", "['sgd_homes_2']", "['sgd_rentalcars_2']",
                         "['sgd_restaurants_1']", "['sgd_music_1']", "['sgd_hotels_4']", "['sgd_media_2']",
                         "['sgd_hotels_3']", "['sgd_rentalcars_3']", "['sgd_hotels_1']", "['sgd_homes_1']"]
        dataset_order = dataset_order[-30:]
    elif 'order32' in output_dir:
        dataset_order = ["['sgd_events_3']", "['sgd_banks_2']", "['sgd_banks_1']", "['sgd_calendar_1']",
                         "['sgd_movies_3']", "['sgd_music_2']", "['sgd_services_2']", "['sgd_payment_1']",
                         "['sgd_media_1']", "['sgd_weather_1']", "['sgd_events_1']", "['sgd_flights_4']",
                         "['sgd_travel_1']", "['sgd_buses_2']", "['sgd_events_2']", "['sgd_alarm_1']",
                         "['sgd_buses_3']", "['sgd_services_1']", "['sgd_buses_1']", "['sgd_restaurants_2']",
                         "['sgd_hotels_2']", "['sgd_ridesharing_2']", "['sgd_rentalcars_1']", "['sgd_movies_1']",
                         "['sgd_ridesharing_1']", "['sgd_media_3']", "['sgd_music_3']", "['sgd_movies_2']",
                         "['sgd_flights_2']", "['sgd_services_4']", "['sgd_flights_1']", "['sgd_services_3']",
                         "['sgd_flights_3']", "['sgd_trains_1']", "['sgd_homes_2']", "['sgd_rentalcars_2']",
                         "['sgd_restaurants_1']", "['sgd_music_1']", "['sgd_hotels_4']", "['sgd_media_2']",
                         "['sgd_hotels_3']", "['sgd_rentalcars_3']", "['sgd_hotels_1']", "['sgd_homes_1']"]
        dataset_order = dataset_order

    else:
        dataset_order = ["['sgd_services_4']", "['sgd_flights_1']", "['sgd_services_3']",
                         "['sgd_flights_3']", "['sgd_trains_1']", "['sgd_homes_2']", "['sgd_rentalcars_2']",
                         "['sgd_restaurants_1']", "['sgd_music_1']", "['sgd_hotels_4']", "['sgd_media_2']",
                         "['sgd_hotels_3']", "['sgd_rentalcars_3']", "['sgd_hotels_1']", "['sgd_homes_1']"]
        # raise ValueError
    return dataset_order



if __name__ == '__main__':
    print(sys.argv)
    output_dir = sys.argv[1]
    output_dir = os.path.join(os.getcwd(), output_dir, 'fwt_predictions')

    csv_list = []

    assert 'order1_' in output_dir
    # for order in [1,2,3,4,5]:
    for seed in [1,2,3,4,5]:
        # res_path = os.path.join(output_dir.replace('order1', 'order{}'.format(order)), 'test_res.txt')
        res_path = os.path.join(output_dir.replace('seed1', 'seed{}'.format(seed)), 'test_res.txt')
        print('res_path')
        print(res_path)
        # print(os.path.exists(res_path))
        if os.path.exists(res_path):
            dataset_order = get_dataset_order(res_path)
            print(res_path)
            with open(res_path) as f:
                for l in f.readlines()[-1:]:
                    l = l.strip()
                    l = l.replace('OrderedDict(', '')
                    l = l[:-1]

                    jga_list = eval(l)
                    jga_list = jga_list[-1:] + jga_list[:-1]
                    jga_dict = dict(jga_list)
                    seed_res = []

                    csv_list.append(dataset_order[1:])
                    for domain in dataset_order[1:]:
                        seed_res.append(round(jga_dict[domain]*100, 2))
                    csv_list.append(seed_res)
        # print('domain jga:')
        # print(' '.join(['\t'.join([str(round(jga, 2)) for jga in seed_jgas]) for seed_jgas in domain_avgs]))
        # print('domain jga mean(std):')
        # print('-'.join(['%.2f(%.2f)' % (np.mean(jga), np.std(jga)) for jga in zip(*domain_avgs)]))
        # print('avg jga:', [round(jga) for jga in avgs])
        # print('avg jga mean(std): %.2f(%.2f)' % (np.mean(avgs), np.std(avgs)))

        # avg_res = ['%.2f(%.2f)' % (np.mean(jga), np.std(jga)) for jga in zip(*csv_list[1:])]
        # csv_list.append(avg_res)

        # for _ in csv_list:
        #     _ = _[:-1] + _[-1:]
    print(csv_list)
    import csv
    with open('gather_res.csv', 'w') as csvfile:
        csv_w = csv.writer(csvfile)
        for line_w in csv_list:
            csv_w.writerow(line_w)

