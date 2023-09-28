cifar100_teacher_model_name = [
    'cifar100-vgg13-0', 'cifar100-vgg13-1','cifar100-vgg13-2','cifar100-vgg13-3',

]


# ------------- teacher net --------------------#
teacher_model_path_dict = {
    'cifar100-vgg13-0': '/home/liujiacheng/CA-MKD/save/CAMKD/teachers/models/vgg13_cifar100_lr_0.05_decay_0.0005_trial_0/vgg13_best.pth',
    'cifar100-vgg13-1': '/home/liujiacheng/CA-MKD/save/CAMKD/teachers/models/vgg13_cifar100_lr_0.05_decay_0.0005_trial_1/vgg13_best.pth',
    'cifar100-vgg13-2': '/home/liujiacheng/CA-MKD/save/CAMKD/teachers/models/vgg13_cifar100_lr_0.05_decay_0.0005_trial_2/vgg13_best.pth',
    'cifar100-vgg13-3': '/home/liujiacheng/CA-MKD/save/CAMKD/teachers/models/vgg13_cifar100_lr_0.05_decay_0.0005_trial_3/vgg13_best.pth',
}

"""原版下面
cifar100_teacher_model_name = [
    'cifar100-resnet32x4-0', 'cifar100-resnet32x4-1', 'cifar100-resnet32x4-2',

]


# ------------- teacher net --------------------#
teacher_model_path_dict = {
    'cifar100-resnet32x4-0': '/home/liujiacheng/CA-MKD/save/CAMKD/teachers/models/wrn_40_2_cifar100_lr_0.05_decay_0.0005_trial_0/wrn_40_2_best.pth',
    'cifar100-resnet32x4-1': '/home/liujiacheng/CA-MKD/save/CAMKD/teachers/models/wrn_40_2_cifar100_lr_0.05_decay_0.0005_trial_1/wrn_40_2_best.pth',
    'cifar100-resnet32x4-2': '/home/liujiacheng/CA-MKD/save/CAMKD/teachers/models/wrn_40_2_cifar100_lr_0.05_decay_0.0005_trial_2/wrn_40_2_best.pth',
}
"""