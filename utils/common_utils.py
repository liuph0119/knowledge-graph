


def read_lines(file, bucket=None):
    if bucket is None:
        with open(file, "r", encoding="utf-8") as f:
            lines = f.readlines()
    else:
        lines = str(bucket.get_object(file).read(), encoding="utf-8").split("\n")
    return lines


def torch_save(path, bucket=None):
    pass
