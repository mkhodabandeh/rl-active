from argparse import ArgumentParser
#import aspy.yaml as yaml
from aspy.yaml import ordered_dump

def PromptSolverConfig(args):
    return {}

def PromptRLConfig(args):
    relevant_args = ['ENV', 'RUN_TIME', 'THREADS', 'OPTIMIZERS', 'THREAD_DELAY', 'GAMMA', 'N_STEP_RETURN', \
                    'EPS_START', 'EPS_STOP', 'EPS_STEPS', 'MIN_BATCH', 'LEARNING_RATE', 'LOSS_V', 'LOSS_ENTROPY', \
                    'STATE_SIZE', 'NUM_CLASSES', 'NUM_DATA']
    args.GAMMA_N = args.GAMMA ** args.N_STEP_RETURN
    rl_argdict = {}
    for key in relevant_args: rl_argdict.update({key:getattr(args, key)})

    return rl_argdict

def PromptClassifierConfig(args):
    relevant_args = ['NUM_DATA']
    classifier_argdict = {}
    for key in relevant_args: classifier_argdict.update({key:getattr(args, key)})
    return classifier_argdict


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--NUM_DATA', type=int, default=1000,help='number of training images')
    parser.add_argument('--ENV', type=str, default='ActiveLearningEnv-v0', help='environment name')
    parser.add_argument('--OUTPUT_YAML', type=str, default='config.yml', help='yaml config filename')
    parser.add_argument('--RUN_TIME', type=int, default=48*60*60, help='Training time in seconds')
    parser.add_argument('--THREADS', type=int, default=7, help='Number of threads')
    parser.add_argument('--OPTIMIZERS', type=int, default=1, help='Number of optimizers')
    parser.add_argument('--THREAD_DELAY', type=float, default=0.001, help='Thread latency')
    parser.add_argument('--GAMMA', type=float, default=0.99, help='Gamma')
    parser.add_argument('--N_STEP_RETURN', type=int, default=7, help='Number of steps to return')
    parser.add_argument('--EPS_START', type=float, default=0.4, help='EPS_START')
    parser.add_argument('--EPS_STOP', type=float, default=0.05, help='EPS_STOP')
    parser.add_argument('--EPS_STEPS', type=int, default=75000, help='EPS_STEPS')
    parser.add_argument('--MIN_BATCH', type=int, default=5, help='MIN_BATCH')
    parser.add_argument('--LEARNING_RATE', type=float, default=5e-3, help='LEARNING_RATE')
    parser.add_argument('--LOSS_V', type=float, default=.5, help='loss co-efficient')
    parser.add_argument('--LOSS_ENTROPY', type=float, default=.01, help='loss entropy')
    parser.add_argument('--STATE_SIZE', type=int, default=128, help='state size')
    parser.add_argument('--NUM_CLASSES', type=int, default=10, help='number of classes')


    args = parser.parse_args()
    argdict = {}
    solver_argdict = PromptSolverConfig(args)
    rl_argdict = PromptRLConfig(args)
    classifier_argdict = PromptClassifierConfig(args)
    argdict.update(solver_argdict)
    argdict.update(rl_argdict)
    argdict.update(classifier_argdict)
    yaml_obj = open(args.OUTPUT_YAML, 'w')
    #ordered_dump(argdict, stream=yaml_obj)
    yaml_obj.write('#Solver Args\n')
    if solver_argdict:
        rdered_dump(solver_argdict, stream=yaml_obj, default_flow_style=False)
    yaml_obj.write('#RL Args\n')
    if rl_argdict:
        rl_argdict = {'RL': rl_argdict}
        ordered_dump(rl_argdict, stream=yaml_obj, default_flow_style=False)
    yaml_obj.write('#Classifier Args\n')
    if classifier_argdict:
        classifier_argdict = {'classifiers': classifier_argdict}
        ordered_dump(classifier_argdict, stream=yaml_obj, default_flow_style=False)
    yaml_obj.close()
