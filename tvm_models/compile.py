import sys
import os
import argparse


import mxnet as mx
from tvm import relay
from gluoncv import model_zoo, data, utils
from tvm.autotvm.measure.measure_methods import set_cuda_target_arch

import tvm
from tvm.contrib import graph_runtime
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm import autotvm


# You can skip the implementation of this function for this tutorial. 
def tune_tasks(tasks,
               measure_option,
               tuner='xgb',
               n_trial=200,
               early_stopping=None,
               log_filename='tuning.log',
               use_transfer_learning=True,
               try_winograd=True,
               quantization=False):

    if quantization:
        for i in range(len(tasks)):
            output_channel = tsk.workload[1][0]
            input_channel = tsk.workload[1][1]
            if output_channel % 4 == 0 and input_channel % 4 == 0:
                tsk = autotvm.task.create(tasks[i].name, tasks[i].args,
                                          tasks[i].target, tasks[i].target_host, 'int8')
                tasks[i] = tsk

    if try_winograd:
        for i in range(len(tasks)):
            try:  # try winograd template
                tsk = autotvm.task.create(tasks[i].name, tasks[i].args,
                                          tasks[i].target, tasks[i].target_host, 'winograd')
                input_channel = tsk.workload[1][1]
                if input_channel >= 64:
                    tasks[i] = tsk
            except Exception:
                pass
    # create tmp log file
    tmp_log_file = log_filename  + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)
    #with open(tmp_log_file,'w') as f:
    #    f.write('')
    #with open(log_filename, 'w') as f:
    #    f.write('')
    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(tsk, loss_type='rank')
        elif tuner == 'ga':
            tuner_obj = GATuner(tsk, pop_size=100)
        elif tuner == 'random':
            tuner_obj = RandomTuner(tsk)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)
        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))
        # do tuning
        tuner_obj.tune(n_trial=min(n_trial, len(tsk.config_space)),
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(n_trial, prefix=prefix),
                           autotvm.callback.log_to_file(tmp_log_file)])
    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)

def tune_and_evaluate(tuning_option, target_host, cc=None):
    # extract workloads from relay program
    print("Extract tasks...")

    # extract the model
    mod, params, target = tuning_option['model']['compile'](use_compiler=False)

    # place holder
    target_host = False

    if tuning_option['quantization']:
        with relay.quantize.qconfig(store_lowbit_output=False):
            mod['main'] = relay.quantize.quantize(mod['main'], params=params)

    if target_host:
        tasks = autotvm.task.extract_from_program(mod["main"], target=target, target_host=target_host,
                                                  params=params,  ops=(relay.op.get("nn.conv2d"),))#ops=(relay.op.nn.conv2d,))
    else:
        tasks = autotvm.task.extract_from_program(mod["main"], target=target,
                                                  params=params, ops=(relay.op.get("nn.conv2d"),))#ops=(relay.op.nn.conv2d,))
    if tuning_option['quantization']:
        for i in range(len(tasks)):
            tsk = tasks[i]
            input_channel = tsk.workload[1][1]
            output_channel = tsk.workload[1][0]
            if output_channel % 4 == 0 and input_channel % 4 == 0:
                tsk = autotvm.task.create(tasks[i].name, tasks[i].args,
                                          tasks[i].target, tasks[i].target_host, 'int8')
                tasks[i] = tsk

    # run tuning tasks
    print("Tuning...")
    tune_tasks(tasks,
               tuning_option['measure_option'],
               tuner=tuning_option['tuner'],
               n_trial=tuning_option['n_trial'],
               early_stopping=tuning_option['early_stopping'],
               log_filename=tuning_option['log_filename'],
               use_transfer_learning=tuning_option['use_transfer_learning'],
               try_winograd=tuning_option['try_winograd'],
               quantization=tuning_option['quantization'])
    # compile kernels with history best records

    with autotvm.apply_history_best(tuning_option['log_filename']):
        print("Compile...")
        # level 3 optimization gave TVMError: Check failed: ir: :
        # VerifyMemory(x, target->device_type): Direct host side access to
        # device memory is detected in fused_nn_contrib_conv2d_winograd_weight_transform_3. Did you forget to bind?
        # So I set opt_level=2 and that worked for compiling
        tvm_compiler(tuning_option['model']['name'], mod, params, target)
        print("exported")

def tvm_compiler(name, mod, params, target):
    print('[*] Compile To Target {}'.format(target))
    with relay.build_config(opt_level=2):
        graph, lib, params = relay.build(mod, target, params=params)
    
    print(type(graph), type(lib), type(params))
    lib.export_library(
        "{}.so".format(MODEL_CONFIG[name]['output_name']))
    print('lib export success')
    with open("{}.json".format(MODEL_CONFIG[name]['output_name']), "w") as fo:
        fo.write(graph)
    print("graph export success")
    with open("{}.params".format(MODEL_CONFIG[name]['output_name']), "wb") as fo:
        fo.write(relay.save_param_dict(params))
    print("params export success")

def get_basic_mxnet_network(name, input_shape, dtype):
    """Get the symbol definition and random weight of a network"""
    block = model_zoo.get_model(name, pretrained=True)
    mod, params = relay.frontend.from_mxnet(block, shape={'data': input_shape}, dtype=dtype)
    #net = mod["main"]
    return mod, params

def compile_hand_detector(use_compiler=False):
    print("compiling hand detector")
    target = 'cuda -libs=cudnn,cublas'
    if ARCH == 'aarch64':
        target += ' -model={}'.format(ARGS.board)
    block = model_zoo.get_model('yolo3_mobilenet1.0_custom', classes=['hands'])
    block.load_parameters('models/yolo3_mobilenet1.0_hands.params')
    block.hybridize(static_alloc=True)
    mod, params = relay.frontend.from_mxnet(block, shape={'data': (1, 3, 320, 320)}, dtype='float32')
    if use_compiler:
        tvm_compiler('hand_detector', mod, params, target)
    return mod, params, target

def compile_object_detector(use_compiler=False):
    print("compiling object detector")
    target = 'cuda -libs=cudnn,cublas'
    if ARCH == 'aarch64':
        target += ' -model={}'.format(ARGS.board)
    block = model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained=True)
    block.hybridize(static_alloc=True)
    mod, params = relay.frontend.from_mxnet(block, shape={'data': (1, 3, 512, 512)}, dtype='float32')
    if use_compiler:
        tvm_compiler('object_detector', mod, params, target)
    return mod, params, target

def compile_simple_pose(use_compiler=False):
    print("compiling pose detector")
    target = 'cuda -libs=cudnn,cublas'
    if ARCH == 'aarch64':
        target += ' -model={}'.format(ARGS.board)
    block = model_zoo.get_model('simple_pose_resnet18_v1b', pretrained=True)
    block.hybridize(static_alloc=True)
    mod, params = relay.frontend.from_mxnet(block, shape={'data': (1, 3, 256, 192)}, dtype='float32')
    if use_compiler:
        tvm_compiler('simple_pose', mod, params, target)
    return mod, params, target

def compile_face_detector(use_compiler=False):
    print("compiling face detector")
    target = 'cuda -libs=cudnn,cublas'
    if ARCH == 'aarch64':
        target += ' -model={}'.format(ARGS.board)
    sym, arg_params, aux_params = mx.model.load_checkpoint('models/mnet.25/mnet.25', 0)
    shape_dict = {'data': (1, 3, 480, 640)}
    relay_sym, relay_params = relay.frontend.from_mxnet(
        symbol=sym,
        shape=shape_dict,
        dtype="float32",
        arg_params=arg_params,
        aux_params=aux_params)
    if use_compiler:
        tvm_compiler('face_detector', relay_sym, relay_params, target)
    return relay_sym, relay_params, target

def compile_face_embedder(use_compiler=False):
    print("compiling face_embedder")
    target = 'cuda -libs=cudnn,cublas'
    if ARCH == 'aarch64':
        target += ' -model={}'.format(ARGS.board)
    sym, arg_params, aux_params = mx.model.load_checkpoint('models/arcface_facerec', 0)
    shape_dict = {'data': (1, 3, 112, 112)}
    relay_sym, relay_params = relay.frontend.from_mxnet(
        symbol=sym,
        shape=shape_dict,
        dtype="float32",
        arg_params=arg_params,
        aux_params=aux_params)
    if use_compiler:
        tvm_compiler('face_embedder', relay_sym, relay_params, target)
    return relay_sym, relay_params, target

if __name__ == '__main__':
    AVAILABLE_MODELS = {'object_detector', 'simple_pose', 'face_detector', 'face_embedder', 'hand_detector'}
    PARSER = argparse.ArgumentParser(description='')
    PARSER.add_argument('--network', type=str, default='object_detector', help='Network Architecture')
    PARSER.add_argument('--target', type=str, default='cuda', help='Deploy Target')
    PARSER.add_argument('--board', type=str, default='tx2', help='board')
    PARSER.add_argument('--dtype', type=str, default='float32', help='Data Type')
    PARSER.add_argument('--tune', type=int, default=0, help='whether to tune the models for the current arch')
    PARSER.add_argument('--ctx', type=int, default=0, help='TVM')
    PARSER.add_argument('--cc', type=str, default=None, help='if on x86, use "aarch64-linux-gnu-g++" to compile for aarch64 - might not work')
    PARSER.add_argument('--n_trial', type=int, default=2000, help='TVM')
    PARSER.add_argument('--quantization', type=bool, default=False, help='TVM')
    PARSER.add_argument('--custom_savename', type=str, default=None, help='TVM')
    ARGS = PARSER.parse_args()

    if ARGS.network not in AVAILABLE_MODELS:
        raise Exception("{0} not in list of acceptable models: {1}".format(ARGS.network, list(AVAILABLE_MODELS)))
    
    if ARGS.ctx!=0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(ARGS.ctx)

    print(ARGS)
    # maybe input your own board here???
    DEVICE_CUDA_ARCH = {
        "tx2": {
            "arch": "sm_62"
            },
        "xavier": {
            "arch": "sm_72"
            },
        "nano": {
            "arch": "sm_53"
        },
        "1080": { # maxwell
            "arch": "sm_52"
        },
        "turing": {
          "arch" : "sm_75"
        }
    }

    if ARGS.board not in {'tx2', 'xavier', 'nano'} :
        ARCH = 'x86'
    else:
        ARCH = 'aarch64'

    TARGET_ARCH = DEVICE_CUDA_ARCH[ARGS.board]['arch']
    set_cuda_target_arch(TARGET_ARCH)

    MODEL_CONFIG = {
        'object_detector':
            {
                'shape': (1, 3, 512, 512),
                'output_name': 'mnet1.0_yolo_{}_cuda'.format(ARCH),
                'dtype': 'float32',
                'cuda': True,
                'compile': compile_object_detector,
                'name': 'object_detector',
            },
        'simple_pose':
            {
                'shape': (1, 3, 256, 192),
                'output_name': 'pose_{}_cuda'.format(ARCH),
                'dtype': 'float32',
                'cuda': True,
                'compile': compile_simple_pose,
                'name': 'simple_pose',
            },
        'face_detector':
            {
                'shape': (1, 3, 480, 640),
                'output_name': 'mnet.25.{}.gpu'.format(ARCH),
                'dtype': 'float32',
                'cuda': True,
                'compile': compile_face_detector,
                'name': 'face_detector',
            },
        'face_embedder':
            {
                'shape': (1, 3, 112, 112),
                'output_name': 'mnet.facerec.{}.gpu'.format(ARCH),
                'dtype': 'float32',
                'cuda': True,
                'compile': compile_face_embedder,
                'name': 'face_embedder',
            },
        'hand_detector' :
            {
                'shape': (1, 3, 320, 320),
                'output_name': 'mnet.1.{}.hands.cuda'.format(ARCH),
                'dtype': 'float32',
                'cuda': True,
                'compile': compile_hand_detector,
                'name': 'hand_detector',
            }
    }

    # compi;e model
    #print(ARGS.tune)
    if not ARGS.tune:
        MODEL_CONFIG[ARGS.network]['compile'](True)
    else:
        TUNING_OPTION = {
            'log_filename': ARGS.network + '.log',
            'tuner': 'xgb',
            'n_trial': int(ARGS.n_trial),
            'early_stopping': 600,
            'use_transfer_learning': True, # this failed?
            'try_winograd': True,
            'measure_option': autotvm.measure_option(
                builder=autotvm.LocalBuilder(timeout=10),
                runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150),
                #runner=autotvm.RPCRunner(
                #    device_key,  # change the device key to your key
                #    '0.0.0.0', 8192,
                #    number=20, repeat=3, timeout=4, min_repeat_ms=150)
                ),
            'model': MODEL_CONFIG[ARGS.network],
            'quantization': ARGS.quantization
            }
        tune_and_evaluate(TUNING_OPTION, False)
