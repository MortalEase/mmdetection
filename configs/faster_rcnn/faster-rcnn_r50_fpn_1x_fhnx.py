_base_ = [
    './faster-rcnn_r50_fpn_1x_coco.py'
]

# 1. 数据集配置
data_root = '/home/qxy/datasets/FHN-X/'
metainfo = {
    'classes': (
        'normal', 'arco_2', 'arco_3a', 'arco_3b', 'arco_4',
        'normal_ex', 'arco_2_ex', 'arco_3a_ex', 'arco_3b_ex', 'arco_4_ex'
    ),
    'palette': [
        (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
        (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30)
    ]
}

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/train.json',
        data_prefix=dict(img='images/train/')))

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/val.json',
        data_prefix=dict(img='images/val/')))
        # ann_file='annotations/test.json',
        # data_prefix=dict(img='images/test/')))

test_dataloader = val_dataloader

# 2. 模型配置 - 修改类别数
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=10)))

# 3. 评估器配置
val_evaluator = dict(ann_file=data_root + 'annotations/val.json')
# val_evaluator = dict(ann_file=data_root + 'annotations/test.json')
test_evaluator = val_evaluator

# 4. 训练策略 (可选：根据需要调整学习率或训练轮数)
# optim_wrapper = dict(
#     optimizer=dict(lr=0.01)) # 默认 8 GPUs, lr=0.02. 如果是单卡，建议调小

# 5. 运行配置
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3))

work_dir = './work_dirs/faster-rcnn_r50_fpn_1x_fhnx'
