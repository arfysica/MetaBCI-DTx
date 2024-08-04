# 单例模式
from threading import RLock

def singleton(cls):
    single_lock = RLock()
    instance = {}
    def singleton_wrapper(*args, **kwargs):
        # 加锁
        with single_lock:
            if cls not in instance:
                instance[cls] = cls(*args, **kwargs)
        return instance[cls]
    return singleton_wrapper
