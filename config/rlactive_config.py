from argparse import ArgumentParser
#import aspy.yaml as yaml
from aspy.yaml import ordered_dump

def PromptSolverConfig(args):
    return {}

def PromptRLConfig(args):
    return {}

def PromptClassifierConfig(args):
    relevant_args = ['NUM_DATA']
    classifier_argdict = {}
    for key in relevant_args: classifier_argdict.update({key:getattr(args, key)})
    return classifier_argdict


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--NUM_DATA', type=int, default=1000,help='number of training images')
    parser.add_argument('--OUTPUT_YAML', type=str, default='config.yml', help='yaml config filename')
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
        ordered_dump(rl_argdict, stream=yaml_obj, default_flow_style=False)
    yaml_obj.write('#Classifier Args\n')
    if classifier_argdict:
        classifier_argdict = {'classifiers': classifier_argdict}
        ordered_dump(classifier_argdict, stream=yaml_obj, default_flow_style=False)
    yaml_obj.close()
