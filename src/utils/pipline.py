class Pipeline:
    def __init__(self, value):
        """允许链式调用，执行then中函数，函数传参时直接默认传self.value,仅支持单个结果的输入输出"""
        self.value = value

    def then(self, func):
        self.value = func(self.value)
        return self  # 返回对象本身，允许链式调用

    def result(self):
        """链式调用之后返回最终的结果"""
        return self.value
