

import numpy as np


if __name__ == '__main__':

    starts = np.array([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    ends = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100])

    binary_search = True

    low_value = 310
    high_value = 510

    start_index = 0
    end_index = len(ends)

    if binary_search is True:
        low = 0
        high = len(starts)
        middle = 0
        while low < high:
            middle = int((low + high) / 2)
            middle_start_value = starts[middle]
            middle_end_value = ends[middle]

            low_start = low_value - middle_start_value
            low_distance = low_start + (high_value - low_value)
            high_distance = low_value - middle_end_value

            print("Position: ", low, middle, high)
            print(middle_start_value, middle_end_value, low_value, high_value)
            print("<-", low_start, low_distance, high_distance)
            print("Middle: ", middle)

            if (low_start < 0 and low_distance < 0):
                print("-")
                high = middle - 1
            elif (low_start > 0 and high_distance > 0):
                print("--")
                low = middle + 1
            elif (low_start <= 0 and low_distance >= 0) or (low_start > 0 and high_distance <= 0):
                print("---")
                high = middle - 1
            else:
                print("----")
                break

            print()
        start_index = middle

        if start_index > 0:
            start_index = start_index - 1
        print("===============", start_index)

        low = 0
        high = len(starts)
        middle = 0
        while low < high:
            middle = int((low + high) / 2)
            middle_start_value = starts[middle]
            middle_end_value = ends[middle]

            low_start = low_value - middle_start_value
            low_distance = low_start + (high_value - low_value)
            high_distance = low_value - middle_end_value

            print("->", low_start, low_distance, high_distance)

            print("Middle: ", middle)
            if (low_start < 0 and low_distance < 0):
                print("-")
                high = middle - 1
            elif (low_start > 0 and high_distance > 0):
                print("--")
                low = middle + 1
            elif (low_start <= 0 and low_distance >= 0) or (low_start > 0 and high_distance <= 0):
                print("---")
                low = middle + 1
            else:
                print("----")
                break
            print()
        end_index = middle

        if end_index < len(ends) - 1:
            end_index = end_index + 1

    print("start_index: ", start_index)
    print("end_index: ", end_index)

