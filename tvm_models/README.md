# Usage

# available models (5/26/2020)

    AVAILABLE_MODELS = {'object_detector', 'simple_pose', 'face_detector', 'face_embedder', 'hand_detector'}

# available devices (5/26/2020)

  http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/

        "tx2": {
            "arch": "sm_62"
            },
        "xavier": {
            "arch": "sm_72"
            },
        "nano": {
            "arch": "sm_53"
        },
        "1080": {
            "arch": "sm_52"
        },
        "turing": {
          "arch" : "sm_75"
        }


# to compile with no tuning (fastest for testing)
    
    python compile.py --network object_detector --board turing --tune 0
    python compile.py --network simple_pose --board turing --tune 0
    python compile.py --network face_detector --board turing --tune 0
    python compile.py --network face_embedder --board turing --tune 0
    python compile.py --network hand_detector --board turing --tune 0

# to compile with tuning (takes a long time.., but is a faster model - needed for prod)
    
    python compile.py --network object_detector --board turing --tune 1 --n_trial 1000
    python compile.py --network simple_pose --board turing --tune 1 --n_trial 1000
    python compile.py --network face_detector --board turing --tune 1 --n_trial 1000
    python compile.py --network face_embedder --board turing --tune 1 --n_trial 1000
    python compile.py --network hand_detector --board turing --tune 1 --n_trial 1000


# how to use it

    python compile.py --network hand_detector --board turing --tune 0

take the output of the compiling: 
    
    mnet.1.x86.hands.cuda.json, mnet.1.x86.hands.cuda.so, mnet.1.x86.hands.cuda.params

and copy the parameters to the recognition/data location in the proper folder

    cp mnet.1.x86.hands.cuda* ../recognition/data/hand_detetector/

change the particular model's config within the recognition directory:

    vi ../recognition/cfg/hand_detector.json

from 

    {
      "deploy_lib_path":        "/data/object_detector/mnet.1.aarch64_cuda.hand.so",
      "deploy_graph_path":      "/data/object_detector/mnet.1.aarch64_cuda.hands.json",
      "deploy_param_path":      "/data/object_detector/mnet.1.aarch64_cuda.hands.params",
      "device_id":              0,
      "width" :                 320,
      "height" :                320,
      "gpu" :                   true
    }

to

    {
      "deploy_lib_path":        "/data/object_detector/mnet.1.x86.cuda.hand.so",
      "deploy_graph_path":      "/data/object_detector/mnet.1.x86.cuda.hands.json",
      "deploy_param_path":      "/data/object_detector/mnet.1.x86.cuda.hands.params",
      "device_id":              0,
      "width" :                 320,
      "height" :                320,
      "gpu" :                   true
    }
