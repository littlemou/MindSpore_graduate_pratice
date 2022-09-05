import mindspore.dataset.vision.c_transforms as CV
import mindspore.dataset.transforms.c_transforms as C
import mindspore.common.dtype as mstype
import mindspore.dataset as ds

def create_dataset(data_path, batch_size=32):
    """
    数据处理

    Args:
        dataset_path (str): 数据路径
        batch_size (int): 批量大小
        repeat_num (int): 数据重复次数

    Returns:
        Dataset对象
    """

    # 载入数据集
    data = ds.ImageFolderDataset(data_path)

    # 打乱数据集
    data = data.shuffle(buffer_size=1000)

    # 定义算子
    trans = [
        CV.Decode(), 
        CV.Resize(256),
        CV.CenterCrop(224),
        # 使用训练backbone网络时用的mean和std
        CV.Normalize(mean=(100.03388269705046, 94.57511259248079, 72.14921665851293), 
                     std=(23.35913427414271, 20.336537235643164, 21.376613547858327)),
        CV.HWC2CHW()
    ]
    type_cast_op = C.TypeCast(mstype.int32)


    # 算子运算
    data = data.map(operations=trans, input_columns="image")
    data = data.map(operations=type_cast_op, input_columns="label")


    # 批处理
    data = data.batch(batch_size, drop_remainder=True)

    # 重复
    data = data.repeat(1)

    return data