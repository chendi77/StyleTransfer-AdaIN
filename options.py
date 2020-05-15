import argparse


def Options():
    parser = argparse.ArgumentParser(description='AdaIN Style Transfer Train')

    # 训练相关的配置信息
    parser.add_argument('--batch_size', '-b', type=int, default=8,
                        help='Number of images in each mini-batch')
    parser.add_argument('--max_iter', '-i', type=int, default=160000,
                        help='最大迭代数')
    parser.add_argument('--gpu_id', '-g', type=int, default=0,
                        help='GPU ID, 为负表示CPU')
    parser.add_argument('--learning_rate', '-lr', type=int, default=1e-4,
                        help='学习率')
    parser.add_argument('--lr_decay', '-lr_decay', type=int, default=5e-5,
                        help='学习率衰减')
    parser.add_argument('--content_dir', '-cd', type=str, default='content',
                        help='内容图像所在文件夹（可以包括子文件夹）')
    parser.add_argument('--style_dir', '-sd', type=str, default='style',
                        help='风格图像所在文件夹（可以包括子文件夹）')
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints',
                        help='保存生成结果的文件夹')
    parser.add_argument('--model_path', '-m', type=str, default=None,
                        help='已训练模型的路径')
    parser.add_argument('--save_interval', type=str, default=1000,
                        help='每多少个迭代保存模型和图像')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='风格图像的风格信息的比重（0-1），0表示不使用风格图像，1表示完全使用风格图像的风格信息')
    parser.add_argument('--lamda', type=float, default=10.0,
                        help='风格损失的比重')
    parser.add_argument('--mean', type=tuple, default=(0.5, 0.5, 0.5),
                        help='图像各个通道归一化的平均值')
    parser.add_argument('--std', type=tuple, default=(0.5, 0.5, 0.5),
                        help='图像各个通道归一化的标准差')

    # 测试相关的配置信息
    parser.add_argument('--content_image', '-c', type=str, default=None,
                        help='内容图像的路径')
    parser.add_argument('--style_image', '-s', type=str, default=None,
                        help='风格图像的路径')
    parser.add_argument('--result_image', '-r', type=str, default=None,
                        help='生成结果的保存路径')

    opt = parser.parse_args()
    return opt

