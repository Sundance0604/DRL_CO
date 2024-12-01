class SETTING:

    def __init__(self, total_time:int, capacity:int):
        self.total_time = total_time    #共计迭代的次数，可能末几轮不再生成订单
        self.capacity = capacity        #几座位车