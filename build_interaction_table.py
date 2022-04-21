import re

def main():
    """Entry point."""
    day_temp = [
        'LST_Day_1km_01-01',
        'LST_Day_1km_02-01',
        'LST_Day_1km_03-01',
        'LST_Day_1km_04-01',
        'LST_Day_1km_05-01',
        'LST_Day_1km_06-01',
        'LST_Day_1km_07-01',
        'LST_Day_1km_08-01',
        ]
    night_temp = [
        'LST_Night_1km_01-01',
        'LST_Night_1km_02-01',
        'LST_Night_1km_03-01',
        'LST_Night_1km_04-01',
        'LST_Night_1km_05-01',
        'LST_Night_1km_06-01',
        'LST_Night_1km_07-01',
        'LST_Night_1km_08-01',
        ]
    ndvi = [
        'ndvi_06-01',
        'ndvi_07-01',
        'ndvi_08-01',
        'ndvi_09-01',
        'ndvi_10-01',
        ]
    precip = [
        'precipitation_01-01',
        'precipitation_02-01',
        'precipitation_03-01',
        'precipitation_04-01',
        'precipitation_05-01',
        'precipitation_06-01',
        'precipitation_07-01',
        'precipitation_08-01',
    ]

    print('term1,term2')
    precip_str = ''
    night_str = ''
    day_str = ''
    for ndvi_id in ndvi:
        month_int, day_int = [int(v) for v in re.match('.*_(\d\d)-(\d\d)', ndvi_id).groups()]
        for month_index in range(1, month_int + (1 if day_int > 15 else 0)):
            try:
                precip_id = precip[month_index-1]
                night_id = night_temp[month_index-1]
                day_id = day_temp[month_index-1]
                precip_str += f'{ndvi_id},{precip_id}\n'
                night_str += f'{ndvi_id},{night_id}\n'
                day_str += f'{ndvi_id},{day_id}\n'
            except IndexError:
                continue
    print(precip_str, end='')
    print(night_str, end='')
    print(day_str)


if __name__ == '__main__':
    main()
