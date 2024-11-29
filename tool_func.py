
def route_combine(lst, list1, list2):
    # 遍历所有可能的分割位置
    for i in range(1, len(lst)):  # i 是分割点
        # 分割列表为两部分
        left = lst[:i]
        right = lst[i:]
        # 判断左边部分是否与list1相等，右边部分是否与list2相等
        if left == list1 and right == list2:
            return True
    return False
