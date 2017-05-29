import sys,os

exp=sys.argv[1]
summary_dir = '{run}_{num}:/local-scratch/mkhodaba/rl-active/{exp}/summaries/{run}_{num}'
agents=int(sys.argv[2])
optimizers=1

dirs= [summary_dir.format(exp=exp,run='agent',num=i) for i in xrange(agents)]
dirs += [summary_dir.format(exp=exp,run='optimizer',num=i) for i in xrange(optimizers)]

command = 'tensorboard --logdir={dir} --port 6060'
command = command.format(dir=','.join(dirs))
print command
os.system(command)
