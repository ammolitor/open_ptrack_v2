import re
import subprocess
import sys
import os
import argparse
import tempfile
from time import time

import numpy as np
import mxnet as mx
from tvm import relay
from gluoncv import model_zoo, data, utils
from tvm.autotvm.measure.measure_methods import set_cuda_target_arch

import tvm
from tvm.contrib import graph_runtime
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner
from tvm import autotvm
# git checkout 38118befc0a7e8a3db87d652b30a9369abb60363 for pre-thrust, non slowness. 
from non_nms_yolo import yolo3_mobilenet1_0_coco, yolo3_mobilenet1_0_custom

import logging
logging.getLogger('autotvm').setLevel(logging.DEBUG)

def available_cpu_count():
    """ Number of available virtual or physical CPUs on this system, i.e.
    user/real as output by time(1) when called with an optimally scaling
    userspace-only program"""

    # cpuset
    # cpuset may restrict the number of *available* processors
    try:
        m = re.search(r'(?m)^Cpus_allowed:\s*(.*)$',
                      open('/proc/self/status').read())
        if m:
            res = bin(int(m.group(1).replace(',', ''), 16)).count('1')
            if res > 0:
                return res
    except IOError:
        pass

    # Python 2.6+
    try:
        import multiprocessing
        return multiprocessing.cpu_count()
    except (ImportError, NotImplementedError):
        pass

    # https://github.com/giampaolo/psutil
    try:
        import psutil
        return psutil.cpu_count()   # psutil.NUM_CPUS on old versions
    except (ImportError, AttributeError):
        pass

    # POSIX
    try:
        res = int(os.sysconf('SC_NPROCESSORS_ONLN'))

        if res > 0:
            return res
    except (AttributeError, ValueError):
        pass

    # Windows
    try:
        res = int(os.environ['NUMBER_OF_PROCESSORS'])

        if res > 0:
            return res
    except (KeyError, ValueError):
        pass

    # jython
    try:
        from java.lang import Runtime
        runtime = Runtime.getRuntime()
        res = runtime.availableProcessors()
        if res > 0:
            return res
    except ImportError:
        pass

    # BSD
    try:
        sysctl = subprocess.Popen(['sysctl', '-n', 'hw.ncpu'],
                                  stdout=subprocess.PIPE)
        scStdout = sysctl.communicate()[0]
        res = int(scStdout)

        if res > 0:
            return res
    except (OSError, ValueError):
        pass

    # Linux
    try:
        res = open('/proc/cpuinfo').read().count('processor\t:')

        if res > 0:
            return res
    except IOError:
        pass

    # Solaris
    try:
        pseudoDevices = os.listdir('/devices/pseudo/')
        res = 0
        for pd in pseudoDevices:
            if re.match(r'^cpuid@[0-9]+$', pd):
                res += 1

        if res > 0:
            return res
    except OSError:
        pass

    # Other UNIXes (heuristic)
    try:
        try:
            dmesg = open('/var/run/dmesg.boot').read()
        except IOError:
            dmesgProcess = subprocess.Popen(['dmesg'], stdout=subprocess.PIPE)
            dmesg = dmesgProcess.communicate()[0]

        res = 0
        while '\ncpu' + str(res) + ':' in dmesg:
            res += 1

        if res > 0:
            return res
    except OSError:
        pass

    raise Exception('Can not determine number of CPUs on this system')

def prune_old_tasks(tasks, log_file):
    if os.path.isfile(log_file):
        new_tasks = []
        history = autotvm.record.ApplyHistoryBest(log_file)
        for task in tasks:
            if history._query_inside(task.target, task.workload) is None:
                new_tasks.append(task)
        return new_tasks
    else:
        return tasks


# Use graph tuner to achieve graph level optimal schedules
# Set use_DP=False if it takes too long to finish.
def tune_graph(graph, dshape, records, opt_sch_file, use_DP=True):
    target_op = [relay.op.get("nn.conv2d"),]
    Tuner = DPTuner if use_DP else PBQPTuner
    executor = Tuner(graph, {'data': dshape}, records, target_op, target)
    executor.benchmark_layout_transform(min_exec_num=2000)
    executor.run()
    executor.write_opt_sch2record_file(opt_sch_file)

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

    #if try_winograd:
    #    for i in range(len(tasks)):
    #        try:  # try winograd template
    #            tsk = autotvm.task.create(tasks[i].name, tasks[i].args,
    #                                      tasks[i].target, tasks[i].target_host, 'winograd')
    #            input_channel = tsk.workload[1][1]
    #            if input_channel >= 64:
    #                tasks[i] = tsk
    #        except Exception:
    #            pass
    # create tmp log file

    tmp_log_file = log_filename  + ".tmp"
    tmp_task_log_file = log_filename + '.task.tmp'
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        #if i == 0:
        #    continue
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
        if use_transfer_learning and os.path.isfile(tmp_log_file):
            try:
                # https://github.com/apache/incubator-tvm/blob/master/python/tvm/autotvm/tuner/xgboost_cost_model.py
                # https://github.com/apache/incubator-tvm/blob/master/python/tvm/autotvm/tuner/xgboost_cost_model.py#L222
                # when inp.task.name != self.task.name
                # nothing is appended to 'data' var
                # this errors out, so we'll just have to avoid it here
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))
            except:
                pass
        

        with tempfile.NamedTemporaryFile() as tmp_task_log_file:
            # do tuning
            tuner_obj.tune(n_trial=min(n_trial, len(tsk.config_space)),
                        early_stopping=early_stopping,
                        measure_option=measure_option,
                        callbacks=[
                            autotvm.callback.progress_bar(n_trial, prefix=prefix),
                            autotvm.callback.log_to_file(tmp_task_log_file.name)])

            with open(tmp_log_file, 'a') as tmp_log_f:
                tmp_log_f.write(tmp_task_log_file.read().decode('utf8'))
                            
    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)


# https://discuss.tvm.ai/t/transfer-learning-doesnt-work-tuner-obj-load-history/5328/3
# https://discuss.tvm.ai/t/solved-can-we-resume-an-autotuning-session/3329/6
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
    tune_graph(mod["main"], tuning_option['model']['shape'], tuning_option['log_filename'], tuning_option['graph_opt_sch_file'])

    #with autotvm.apply_history_best(tuning_option['log_filename']):
    with autotvm.apply_graph_best(tuning_option['graph_opt_sch_file']):
        print("Compile...")
        # level 3 optimization gave TVMError: Check failed: ir: :
        # VerifyMemory(x, target->device_type): Direct host side access to
        # device memory is detected in fused_nn_contrib_conv2d_winograd_weight_transform_3. Did you forget to bind?
        # So I set opt_level=2 and that worked for compiling
        tvm_compiler(tuning_option['model']['name'], mod, params, target)
        print("exported")

def build_save_dict():
    pass

def export(save_dict):
    confg = save_dict['config']
    graph = save_dict['graph']
    lib = save_dict['lib']
    params = save_dict['params']
    opt_path = save_dict['opt_path']
    pose_pipe = save_dict['pose_pipe']
    config_path = os.path.join(opt_path)


def tvm_compiler(name, mod, params, target, n_dets=None):
    print('[*] Compile To Target {}'.format(target))
    #with relay.build_config(opt_level=ARGS.opt_level):
    #    graph, lib, params = relay.build(mod, target, params=params)
    with tvm.transform.PassContext(opt_level=ARGS.opt_level):
        graph, lib, params = relay.build_module.build(
            mod, target=target, params=params)    

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
    print(MODEL_CONFIG[name]['output_name'])

def get_basic_mxnet_network(name, input_shape, dtype):
    """Get the symbol definition and random weight of a network"""
    block = model_zoo.get_model(name, pretrained=True)
    mod, params = relay.frontend.from_mxnet(block, shape={'data': input_shape}, dtype=dtype)
    #net = mod["main"]
    #net = mod["main"]
    #net = relay.Function(net.params, net.body, None, net.type_params, net.attrs)
    #mod = tvm.IRModule.from_expr(net)
    return mod, params

def compile_hand_detector(use_compiler=False):
    print("compiling hand detector")
    target = 'cuda -libs=cudnn,cublas'
    if not CUDA:
        target = 'llvm'
    if ARCH == 'aarch64':
        target += ' -model={}'.format(ARGS.board)
    block = model_zoo.get_model('yolo3_mobilenet1.0_custom', classes=['hands'])
    block.load_parameters('models/yolo3_mobilenet1.0_hands.params')
    block.hybridize(static_alloc=True)
    mod, params = relay.frontend.from_mxnet(block, shape={'data': MODEL_CONFIG["hand_detector"]["shape"]}, dtype='float32')
    #net = mod["main"]
    #net = relay.Function(net.params, net.body, None, net.type_params, net.attrs)
    #mod = tvm.IRModule.from_expr(net)
    if use_compiler:
        tvm_compiler('hand_detector', mod, params, target)
    return mod, params, target

def compile_nonms_hand_detector(use_compiler=False):
    print("compiling hand detector")
    target = 'cuda -libs=cudnn,cublas'
    if not CUDA:
        target = 'llvm'
    if ARCH == 'aarch64':
        target += ' -model={}'.format(ARGS.board)
    block = yolo3_mobilenet1_0_custom(classes=['hands'])
    block.load_parameters('models/yolo3_mobilenet1.0_hands.params')
    block.hybridize(static_alloc=True)

    x = np.random.randn(*MODEL_CONFIG["nonms_hand_detector"]["shape"])
    x = mx.nd.array(x)
    x = block(x)
    N_DETS = x.shape[1]
    print("*********")
    print("*** WARNING ***")
    print("*********")
    print("Make sure you add 'n_dets' output in hand_detector.json")
    print("n_dets: {}".format(N_DETS))
    print("*********")

    mod, params = relay.frontend.from_mxnet(block, shape={'data': MODEL_CONFIG["nonms_hand_detector"]["shape"]}, dtype='float32')
    #net = mod["main"]
    #net = relay.Function(net.params, net.body, None, net.type_params, net.attrs)
    #mod = tvm.IRModule.from_expr(net)
    if use_compiler:
        tvm_compiler('nonms_hand_detector', mod, params, target)
    return mod, params, target

def compile_nonms_object_detector(use_compiler=False):
    print("compiling object detector")
    target = 'cuda -libs=cudnn,cublas -model='+ARGS.board
    #target = 'cuda -libs=cudnn'
    #target = 'cuda'
    if not CUDA:
        target = 'llvm'
    if ARCH == 'aarch64':
        target += ' -model={}'.format(ARGS.board)
    print(target)
    block = yolo3_mobilenet1_0_coco(pretrained=True)
    block.hybridize(static_alloc=True)
    x = np.random.randn(*MODEL_CONFIG["object_detector"]["shape"])
    x = mx.nd.array(x)
    x = block(x)
    N_DETS = x.shape[1]
    print("*********")
    print("*** WARNING ***")
    print("*********")
    print("Make sure you add 'n_dets' output in pose_model.json")
    print("n_dets: {}".format(N_DETS))
    print("*********")
    mod, params = relay.frontend.from_mxnet(block, shape={'data': MODEL_CONFIG["nonms_object_detector"]["shape"]}, dtype='float32')
    #net = mod["main"]
    # fused_nn_softmax: num_args should be 4
    # https://discuss.tvm.ai/t/something-wrong-when-my-model-run/3300
    #net = relay.Function(net.params, net.body, None, net.type_params, net.attrs)
    #mod = tvm.IRModule.from_expr(net)
    if use_compiler:
        tvm_compiler('nonms_object_detector', mod, params, target)
    return mod, params, target

def compile_object_detector(use_compiler=False):
    print("compiling object detector")
    target = 'cuda -libs=cudnn,cublas -model='+ARGS.board
    #target = 'cuda -libs=cudnn'
    #target = 'cuda'
    if not CUDA:
        target = 'llvm'
    if ARCH == 'aarch64':
        target += ' -model={}'.format(ARGS.board)
    print(target)
    block = model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained=True)
    block.hybridize(static_alloc=True)
    mod, params = relay.frontend.from_mxnet(block, shape={'data': MODEL_CONFIG["object_detector"]["shape"]}, dtype='float32')
    #net = mod["main"]
    # fused_nn_softmax: num_args should be 4
    # https://discuss.tvm.ai/t/something-wrong-when-my-model-run/3300
    #net = relay.Function(net.params, net.body, None, net.type_params, net.attrs)
    #mod = tvm.IRModule.from_expr(net)
    if use_compiler:
        tvm_compiler('object_detector', mod, params, target)
    return mod, params, target

def compile_simple_pose(use_compiler=False):
    print("compiling pose detector")
    target = 'cuda -libs=cudnn,cublas'
    if not CUDA:
        target = 'llvm'
    if ARCH == 'aarch64':
        target += ' -model={}'.format(ARGS.board)
    block = model_zoo.get_model('simple_pose_resnet18_v1b', pretrained=True)
    block.hybridize(static_alloc=True)
    mod, params = relay.frontend.from_mxnet(block, shape={'data': MODEL_CONFIG["simple_pose"]["shape"]}, dtype='float32')
    #net = mod["main"]
    #net = relay.Function(net.params, net.body, None, net.type_params, net.attrs)
    #mod = tvm.IRModule.from_expr(net)
    if use_compiler:
        tvm_compiler('simple_pose', mod, params, target)
    return mod, params, target

def compile_face_detector(use_compiler=False):
    print("compiling face detector")
    target = 'cuda -libs=cudnn,cublas'
    if not CUDA:
        target = 'llvm'
    if ARCH == 'aarch64':
        target += ' -model={}'.format(ARGS.board)
    sym, arg_params, aux_params = mx.model.load_checkpoint('models/mnet.25/mnet.25', 0)
    shape_dict = {'data': MODEL_CONFIG["face_detector"]["shape"]}
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
    if not CUDA:
        target = 'llvm'
    if ARCH == 'aarch64':
        target += ' -model={}'.format(ARGS.board)
    sym, arg_params, aux_params = mx.model.load_checkpoint('models/arcface_facerec', 0)
    shape_dict = {'data': MODEL_CONFIG["face_embedder"]["shape"]}
    relay_sym, relay_params = relay.frontend.from_mxnet(
        symbol=sym,
        shape=shape_dict,
        dtype="float32",
        arg_params=arg_params,
        aux_params=aux_params)
    if use_compiler:
        tvm_compiler('face_embedder', relay_sym, relay_params, target)
    return relay_sym, relay_params, target

def evaluate_model(config, loaded_json, loaded_lib, loaded_params, ctx):
    module = graph_runtime.create(loaded_json, loaded_lib, ctx)
    module.load_params(loaded_params)
    print('[*] Graph RunTime is Created')
    tvm_input = np.random.randn(*config['shape']).astype(config['dtype'])
    tvm_input = tvm.nd.array(tvm_input, ctx=ctx)
    module.set_input('data', tvm_input)
    print('[*] Run Test')
    avg = []
    for i in range(100):
        tvm_input = np.random.randn(*config['shape']).astype(config['dtype'])
        tvm_input = tvm.nd.array(tvm_input, ctx=ctx)
        module.set_input('data', tvm_input)
        start = time()
        module.run()
        ctx.sync()
        e = time() - start
        print('Time Cost : ', e)
        # print('anchor sum : ', anchor_boxes.sum())
        print('=========================')
        avg.append(e)

    print('[!] Evaluation Done')
    print('[!] First pass time: {}'.format(avg[0]))
    print('[!] Average time: {}'.format(np.mean(avg[1:])))

def load_raw_model(path_base):
    loaded_json = open("{}.json".format(path_base)).read()
    loaded_lib = tvm.runtime.load_module("{}.so".format(path_base))
    loaded_params = bytearray(open("{}.params".format(path_base), "rb").read())
    return loaded_json, loaded_lib, loaded_params

if __name__ == '__main__':
    AVAILABLE_MODELS = {'object_detector', 'simple_pose', 'face_detector', 'face_embedder', 'hand_detector', 'nonms_hand_detector','nonms_object_detector', 'pose_pipeline_nonms', 'pose_pipeline'}
    PARSER = argparse.ArgumentParser(description='')
    PARSER.add_argument('--network', type=str, default='nonms_object_detector', help='Network Architecture')
    PARSER.add_argument('--target', type=str, default='cuda', help='Deploy Target')
    PARSER.add_argument('--board', type=str, default='titanx', help='board')
    PARSER.add_argument('--dtype', type=str, default='float32', help='Data Type')
    PARSER.add_argument('--tune', type=int, default=0, help='whether to tune the models for the current arch')
    PARSER.add_argument('--ctx', type=int, default=0, help='TVM')
    PARSER.add_argument('--cc', type=str, default=None, help='if on x86, use "aarch64-linux-gnu-g++" to compile for aarch64 - might not work')
    PARSER.add_argument('--n_trial', type=int, default=2000, help='TVM')
    PARSER.add_argument('--quantization', type=bool, default=False, help='TVM')
    PARSER.add_argument('--custom_savename', type=str, default=None, help='TVM')
    PARSER.add_argument('--profile_speed', type=int, default=0, help='TVM')
    PARSER.add_argument('--profile_speed_name', type=str, default=None, help='TVM')
    PARSER.add_argument('--opt_level', type=int, default=2, help='TVM')
    PARSER.add_argument('--early_stopping', type=int, default=600, help='TVM') 
    PARSER.add_argument('--rpc', type=int, default=2, help='TVM')
    PARSER.add_argument('--override-shape', type=int, default=0, help='TVM')
    PARSER.add_argument('--override-width', type=int, default=0, help='TVM')
    PARSER.add_argument('--override-height', type=int, default=0, help='TVM')
    ARGS = PARSER.parse_args()

    # https://stackoverflow.com/questions/1006289/how-to-find-out-the-number-of-cpus-using-python
    # get available n threads
    num_threads  = available_cpu_count()
    #os.environ["TVM_NUM_THREADS"] = str(num_threads)

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
        "titanx": { # maxwell
            "arch": "sm_52"
        },
        "1080ti": { # maxwell
            "arch": "sm_61"
        },
        "turing": {
          "arch" : "sm_75"
        }
    }

    if ARGS.board not in {'tx2', 'xavier', 'nano'} :
        ARCH = 'x86'
    else:
        ARCH = 'aarch64'

    if 'cuda' in ARGS.target:
        TARGET_ARCH = DEVICE_CUDA_ARCH[ARGS.board]['arch']
        set_cuda_target_arch(TARGET_ARCH)
        CUDA = True
        suffix = 'gpu'
    else:
        CUDA = False
        suffix = 'cpu'

    MODEL_CONFIG = {
        'nonms_object_detector':
            {
                'shape': (1, 3, 512, 512),
                'output_name': 'mnet1.0.yolo.nonms.{}.{}'.format(ARCH, suffix),
                'dtype': 'float32',
                'cuda': CUDA,
                'compile': compile_nonms_object_detector,
                'name': 'nonms_object_detector',
            },
        'object_detector':
            {
                'shape': (1, 3, 512, 512),
                'output_name': 'mnet1.0.yolo.{}.{}'.format(ARCH, suffix),
                'dtype': 'float32',
                'cuda': CUDA,
                'compile': compile_object_detector,
                'name': 'object_detector',
            },
        'simple_pose':
            {
                'shape': (1, 3, 256, 192),
                'output_name': 'pose.{}.{}'.format(ARCH, suffix),
                'dtype': 'float32',
                'cuda': CUDA,
                'compile': compile_simple_pose,
                'name': 'simple_pose',
            },
        'face_detector':
            {
                'shape': (1, 3, 480, 640), # yes, this is correct, the height is 640 and width is 480
                'output_name': 'mnet.25.{}.{}'.format(ARCH, suffix),
                'dtype': 'float32',
                'cuda': CUDA,
                'compile': compile_face_detector,
                'name': 'face_detector',
            },
        'face_embedder':
            {
                'shape': (1, 3, 112, 112),
                'output_name': 'mnet.facerec.{}.{}'.format(ARCH, suffix),
                'dtype': 'float32',
                'cuda': CUDA,
                'compile': compile_face_embedder,
                'name': 'face_embedder',
            },
        'hand_detector' :
            {
                'shape': (1, 3, 320, 320),
                'output_name': 'mnet.1.{}.hands.{}'.format(ARCH, suffix),
                'dtype': 'float32',
                'cuda': CUDA,
                'compile': compile_hand_detector,
                'name': 'hand_detector',
            },
        'nonms_hand_detector' :
            {
                'shape': (1, 3, 320, 320),
                'output_name': 'mnet.1.nonms.{}.hands.{}'.format(ARCH, suffix),
                'dtype': 'float32',
                'cuda': CUDA,
                'compile': compile_nonms_hand_detector,
                'name': 'nonms_hand_detector',
            }
    }

    if ARGS.override_shape:
        height = ARGS.override_height
        width = ARGS.override_width
        MODEL_CONFIG[ARGS.network]['shape'] = (1, 3, height, width)
        MODEL_CONFIG[ARGS.network]['output_name'] = "{}.{}.{}".format(height, width, MODEL_CONFIG[ARGS.network]['output_name'])

    MODEL_CONFIG[ARGS.network]['output_name'] = tvm.__version__ + "." + MODEL_CONFIG[ARGS.network]['output_name']


    if ARGS.profile_speed and ARGS.profile_speed_name:
        config = MODEL_CONFIG[ARGS.network]
        loaded_json, loaded_lib, loaded_params = load_raw_model(ARGS.profile_speed_name)

        if CUDA:
            ctx = tvm.gpu(0)
        else:
            ctx = tvm.cpu()
        evaluate_model(config, loaded_json, loaded_lib, loaded_params, ctx)

    # compile model
    #print(ARGS.tune)
    elif not ARGS.tune:
        MODEL_CONFIG[ARGS.network]['compile'](True)
        if CUDA:
            ctx = tvm.gpu(0)
        else:
            ctx = tvm.cpu()
        loaded_json, loaded_lib, loaded_params = load_raw_model(MODEL_CONFIG[ARGS.network]['output_name'])
        evaluate_model(MODEL_CONFIG[ARGS.network], loaded_json, loaded_lib, loaded_params, ctx)

    else:
        TUNING_OPTION = {
            'log_filename': ARGS.network + '.log',
            'graph_opt_sch_file' : ARGS.network + "_graph_opt.log",
            'tuner': 'xgb',
            'n_trial': int(ARGS.n_trial),
            'early_stopping': ARGS.early_stopping,
            'use_transfer_learning': True, # this failed?
            'try_winograd': True,
            'measure_option': autotvm.measure_option(
                builder=autotvm.LocalBuilder(timeout=10),
                runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150)
                ),
            'model': MODEL_CONFIG[ARGS.network],
            'quantization': ARGS.quantization
            }
        if ARGS.rpc:
            # https://docs.tvm.ai/tutorials/autotvm/tune_nnvm_cuda.html#scale-up-measurement-by-using-multiple-devices
            TUNING_OPTION['measure_option'] = autotvm.measure_option(
                builder=autotvm.LocalBuilder(timeout=10),
                runner=autotvm.RPCRunner(
                    ARGS.board,  # change the device key to your key
                    '0.0.0.0', 9190,
                    number=20, repeat=3, timeout=4, min_repeat_ms=150)
                )
        tune_and_evaluate(TUNING_OPTION, False)
