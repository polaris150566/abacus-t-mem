from rdkit import Chem

def expose(obj, indent=0, max_depth=10, visited=None):
    """
    递归打印对象的所有内容

    Args:
        obj: 要打印的对象
        indent: 缩进级别，用于美化输出
        max_depth: 最大递归深度，超过此深度将不再递归
        visited: 用于跟踪已经访问过的对象，避免无限递归
    """
    if visited is None:
        visited = set()

    prefix = ' ' * indent

    # 检查是否已经访问过该对象
    if id(obj) in visited:
        print(f"{prefix}...(已访问过)")
        return

    # 将当前对象添加到已访问集合中
    visited.add(id(obj))

    # 判断对象类型并相应处理
    if isinstance(obj, (list, tuple)):
        print(f"{prefix}{type(obj).__name__}:")
        for index, item in enumerate(obj):
            print(f"{prefix}  [{index}]:")
            if max_depth > 0:
                expose(item, indent + 4, max_depth - 1, visited)
            else:
                print(f"{prefix}    ... (max depth reached)")
    elif isinstance(obj, dict):
        print(f"{prefix}{type(obj).__name__}:")
        for key, value in obj.items():
            print(f"{prefix}  {key}:")
            if max_depth > 0:
                expose(value, indent + 4, max_depth - 1, visited)
            else:
                print(f"{prefix}    ... (max depth reached)")
    elif hasattr(obj, '__dict__'):
        print(f"{prefix}{type(obj).__name__}:")
        for attr_name in dir(obj):
            if not (attr_name.startswith('__') and attr_name.endswith('__')):
                try:
                    attr_value = getattr(obj, attr_name)
                    print(f"{prefix}  {attr_name}:")
                    if max_depth > 0:
                        expose(attr_value, indent + 4, max_depth - 1, visited)
                    else:
                        print(f"{prefix}    ... (max depth reached)")
                except:
                    print(f"{prefix}  {attr_name}: (无法访问)")
    else:
        print(f"{prefix}{obj}")