import sys,os
import subprocess
current_user = subprocess.check_output(['whoami']).strip()

exp=sys.argv[1]
summary_dir = '{run}_{num}:/local-scratch/{user}/rl-active/{exp}/summaries/{run}_{num}'
agents=int(sys.argv[2])
optimizers=1

dirs= [summary_dir.format(exp=exp,run='agent',num=i,user=current_user) for i in xrange(agents)]
dirs += [summary_dir.format(exp=exp,run='optimizer',num=i,user=current_user) for i in xrange(optimizers)]

command = 'tensorboard --logdir={dir} --port 6060'
command = command.format(dir=','.join(dirs))
print command
os.system(command)
