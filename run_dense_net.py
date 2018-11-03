import argparse
    
from models.dense_net import DenseNet
from models.comprese_dense_net import CompreseDenseNet
from data_providers.utils import get_data_provider_by_name
import tensorflow as tf
import numpy as np
import re

train_params_cifar = {
    'batch_size': 64,
    'n_epochs': 300,
    'initial_learning_rate': 0.1,
    'reduce_lr_epoch_1': 150,  # epochs * 0.5
    'reduce_lr_epoch_2': 225,  # epochs * 0.75
    'validation_set': True,
    'validation_split': None,  # None or float
    'shuffle': 'every_epoch',  # None, once_prior_train, every_epoch
    'normalization': 'by_chanels',  # None, divide_256, divide_255, by_chanels
}

train_params_svhn = {
    'batch_size': 64,
    'n_epochs': 40,
    'initial_learning_rate': 0.1,
    'reduce_lr_epoch_1': 20,
    'reduce_lr_epoch_2': 30,
    'validation_set': True,
    'validation_split': None,  # you may set it 6000 as in the paper
    'shuffle': True,  # shuffle dataset every epoch or not
    'normalization': 'divide_255',
}

COMPOSITE_NAME_REGEX = re.compile('(Block_)(\d)(/layer_)(\d{1,2})(/composite_function/kernel:0)')
BOTTLENECK_NAME_REGEX = re.compile('(Block_)(\d)(/layer_)(\d{1,2})(/bottleneck/kernel:0)')
TRANSITION_NAME_REGEX = re.compile('(Transition_after_block_)(\d{1,2})(/composite_function/kernel:0)')
BATCH_NORN_NAME_REGEX = re.compile('(Block_)(\d)(/layer_)(\d{1,2})(/bottleneck/BatchNorm/)(beta|gamma|moving_mean|moving_variance)(:0)')
TRANSITION_BATCH_NORN_NAME_REGEX = re.compile('(Transition_after_block_)(\d)(/composite_function/BatchNorm/)(beta|gamma|moving_mean|moving_variance)(:0)')
TRANSITION_TO_CLASS_BATCH_NORN_NAME_REGEX = re.compile('(Transition_to_classes/BatchNorm/)(beta|gamma|moving_mean|moving_variance)(:0)')

def get_train_params_by_name(name):
    if name in ['C10', 'C10+', 'C100', 'C100+']:
        return train_params_cifar
    if name == 'SVHN':
        return train_params_svhn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train', action='store_true',
        help='Train the model')
    parser.add_argument(
        '--test', action='store_true',
        help='Test model for required dataset if pretrained model exists.'
             'If provided together with `--train` flag testing will be'
             'performed right after training.')
    parser.add_argument(
        '--count_zero', action='store_true',
        help='Check the number of 0 weights is the end of the training.')
    parser.add_argument(
        '--comprese', action='store_true',
        help='Compressing the model by clustering the 3X3 kernels')
    parser.add_argument(
        '--model_type', '-m', type=str, choices=['DenseNet', 'DenseNet-BC'],
        default='DenseNet',
        help='What type of model to use')
    parser.add_argument(
        '--clusster_num', '-c', type=int, default=12,
        help='If compresion, state how much clster to each layer.')
    parser.add_argument(
        '--growth_rate', '-k', type=int, choices=[12, 24, 40],
        default=12,
        help='Grows rate for every layer, '
             'choices were restricted to used in paper')
    parser.add_argument(
        '--depth', '-d', type=int, choices=[40, 100, 190, 250],
        default=40,
        help='Depth of whole network, restricted to paper choices')
    parser.add_argument(
        '--dataset', '-ds', type=str,
        choices=['C10', 'C10+', 'C100', 'C100+', 'SVHN'],
        default='C10',
        help='What dataset should be used')
    parser.add_argument(
        '--total_blocks', '-tb', type=int, default=3, metavar='',
        help='Total blocks of layers stack (default: %(default)s)')
    parser.add_argument(
        '--keep_prob', '-kp', type=float, metavar='',
        help="Keep probability for dropout.")
    parser.add_argument(
        '--weight_decay', '-wd', type=float, default=1e-4, metavar='',
        help='Weight decay for optimizer (default: %(default)s)')
    parser.add_argument(
        '--nesterov_momentum', '-nm', type=float, default=0.9, metavar='',
        help='Nesterov momentum (default: %(default)s)')
    parser.add_argument(
        '--reduction', '-red', type=float, default=0.5, metavar='',
        help='reduction Theta at transition layer for DenseNets-BC models')
    parser.add_argument(
        '--logs', dest='should_save_logs', action='store_true',
        help='Write tensorflow logs')
    parser.add_argument(
        '--no-logs', dest='should_save_logs', action='store_false',
        help='Do not write tensorflow logs')
    parser.set_defaults(should_save_logs=True)

    parser.add_argument(
        '--saves', dest='should_save_model', action='store_true',
        help='Save model during training')
    parser.add_argument(
        '--no-saves', dest='should_save_model', action='store_false',
        help='Do not save model during training')
    parser.set_defaults(should_save_model=True)

    parser.add_argument(
        '--renew-logs', dest='renew_logs', action='store_true',
        help='Erase previous logs for model if exists.')
    parser.add_argument(
        '--not-renew-logs', dest='renew_logs', action='store_false',
        help='Do not erase previous logs for model if exists.')
    parser.set_defaults(renew_logs=True)

    args = parser.parse_args()

    if not args.keep_prob:
        if args.dataset in ['C10', 'C100', 'SVHN']:
            args.keep_prob = 0.8
        else:
            args.keep_prob = 1.0
    if args.model_type == 'DenseNet':
        args.bc_mode = False
        args.reduction = 1.0
    elif args.model_type == 'DenseNet-BC':
        args.bc_mode = True

    model_params = vars(args)

    if not args.train and not args.test and not args.comprese:
        print("You should train or test or comprese your network. Please check params.")
        exit()

    # some default params dataset/architecture related
    train_params = get_train_params_by_name(args.dataset)
    print("Params:")
    for k, v in model_params.items():
        print("\t%s: %s" % (k, v))
    print("Train params:")
    for k, v in train_params.items():
        print("\t%s: %s" % (k, v))

    print("Prepare training data...")
    data_provider = get_data_provider_by_name(args.dataset, train_params)
    print("Initialize the model..")
    model = DenseNet(for_test_only=False, init_variables=None, init_global=None, bottleneck_output_size=None, first_output_features=None, data_provider=data_provider, **model_params)
    if args.train:
        print("Data provider train images: ", data_provider.train.num_examples)
        model.train_all_epochs(train_params)
    if args.comprese:
        if not args.train:
            model.load_model()
        old_varaible = model.get_trainable_variables_value()
        print("Commpresing the network")
        comprese_model = CompreseDenseNet(model, args.clusster_num)
        all_new_comprese_kernels, all_new_bottleneck_kernels, all_new_batch_norm, all_new_transion_kernels, all_new_batch_norm_for_transion, new_W, new_transion_to_class_batch_norm = comprese_model.comprese()
        init_variables = []
        for var in tf.trainable_variables():
            composite_match = COMPOSITE_NAME_REGEX.match(var.name)
            bottleneck_match = BOTTLENECK_NAME_REGEX.match(var.name)
            transition_match = TRANSITION_NAME_REGEX.match(var.name)
            batch_norm_match = BATCH_NORN_NAME_REGEX.match(var.name)
            transition_batch_norm_match = TRANSITION_BATCH_NORN_NAME_REGEX.match(var.name)
            transition_to_class_batch_norm_match = TRANSITION_TO_CLASS_BATCH_NORN_NAME_REGEX.match(var.name)
            if composite_match:
                block_index = int(composite_match.groups()[1])
                layer_index = int(composite_match.groups()[3])
                init_variables.append((np.float32(all_new_comprese_kernels[block_index][layer_index]), var.name))
            elif bottleneck_match:
                block_index = int(bottleneck_match.groups()[1])
                layer_index = int(bottleneck_match.groups()[3])
                init_variables.append((np.float32(all_new_bottleneck_kernels[block_index][layer_index]), var.name))
            elif batch_norm_match:
                block_index = int(batch_norm_match.groups()[1])
                layer_index = int(batch_norm_match.groups()[3])
                param_type = batch_norm_match.groups()[5]
                if param_type == 'beta':
                    init_variables.append((np.float32(all_new_batch_norm[block_index][layer_index][0]), var.name))
                if param_type == 'gamma':
                    init_variables.append((np.float32(all_new_batch_norm[block_index][layer_index][1]), var.name))    
            elif transition_match:
                block_index = int(transition_match.groups()[1])
                init_variables.append((np.float32(all_new_transion_kernels[block_index]), var.name))
            elif transition_batch_norm_match:
                block_index = int(transition_batch_norm_match.groups()[1])
                param_type = transition_batch_norm_match.groups()[3]
                if param_type == 'beta':
                    init_variables.append((np.float32(all_new_batch_norm_for_transion[block_index][0]), var.name))
                if param_type == 'gamma':
                    init_variables.append((np.float32(all_new_batch_norm_for_transion[block_index][1]), var.name))
            elif var.name == 'Transition_to_classes/W:0':
                init_variables.append((np.float32(new_W), var.name))
            elif transition_to_class_batch_norm_match:
                param_type = transition_to_class_batch_norm_match.groups()[1]
                if param_type == 'beta':
                    init_variables.append((np.float32(new_transion_to_class_batch_norm[0]), var.name))
                if param_type == 'gamma':
                    init_variables.append((np.float32(new_transion_to_class_batch_norm[1]), var.name))       
            else:
                var_vector = np.float32(model.sess.run(var))
                init_variables.append((var_vector, var.name))
        init_global = []
        for var in tf.global_variables():
            if 'moving' in var.name:
                batch_norm_match = BATCH_NORN_NAME_REGEX.match(var.name)
                transition_batch_norm_match = TRANSITION_BATCH_NORN_NAME_REGEX.match(var.name)
                transition_to_class_batch_norm_match = TRANSITION_TO_CLASS_BATCH_NORN_NAME_REGEX.match(var.name)
                if batch_norm_match:
                    block_index = int(batch_norm_match.groups()[1])
                    layer_index = int(batch_norm_match.groups()[3])
                    param_type = batch_norm_match.groups()[5]
                    if param_type == 'moving_mean':
                        init_global.append((np.float32(all_new_batch_norm[block_index][layer_index][2]), var.name))
                    if param_type == 'moving_variance':
                        init_global.append((np.float32(all_new_batch_norm[block_index][layer_index][3]), var.name)) 
                elif transition_batch_norm_match:
                    block_index = int(transition_batch_norm_match.groups()[1])
                    param_type = transition_batch_norm_match.groups()[3]
                    if param_type == 'moving_mean':
                        init_global.append((np.float32(all_new_batch_norm_for_transion[block_index][2]), var.name))
                    if param_type == 'moving_variance':
                        init_global.append((np.float32(all_new_batch_norm_for_transion[block_index][3]), var.name))
                elif transition_to_class_batch_norm_match:
                    param_type = transition_to_class_batch_norm_match.groups()[1]
                    if param_type == 'moving_mean':
                        init_global.append((np.float32(new_transion_to_class_batch_norm[2]), var.name))
                    if param_type == 'moving_variance':
                        init_global.append((np.float32(new_transion_to_class_batch_norm[3]), var.name))                     
                else:
                    var_vector = np.float32(model.sess.run(var))
                    init_global.append((var_vector, var.name))
        model.close()
        model_params['growth_rate'] = args.clusster_num
        model_params['n_epochs'] = 50
        model_params['initial_learning_rate'] = 0.001
        model_params['reduce_lr_epoch_1'] = 25
        model_params['reduce_lr_epoch_2'] = 40
        model_params['should_save_model'] = False
        model = DenseNet(for_test_only=True, init_variables=init_variables, init_global=init_global, bottleneck_output_size=4*args.growth_rate ,first_output_features=2*args.growth_rate,data_provider=data_provider, **model_params)
        print("Data provider train images: ", data_provider.train.num_examples)
        model.train_all_epochs(train_params)
    if args.test:
        if not args.train and not args.comprese:
            model.load_model()
        print("Data provider test images: ", data_provider.test.num_examples)
        print("Testing...")
        loss, accuracy = model.test(data_provider.test, batch_size=200)
        print("mean cross_entropy: %f, mean accuracy: %f" % (loss, accuracy))
        if args.count_zero:
            print("Counting Zero")
            zero_num = model.count_zero()
            print("weight_decay: %f, mean accuracy: %f, zero_num: %f" % (args.weight_decay, accuracy, zero_num))

