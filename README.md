#### 详细材料

登录方式：[1] https://docs.hpc.sjtu.edu.cn/login/index.html

作业提交：[2] https://docs.hpc.sjtu.edu.cn/job/slurm.html （超算使用slurm作业调度系统，请避免在登录节点运行作业！）

dgx2: [3] https://docs.hpc.sjtu.edu.cn/job/dgx.html

pytorch: [4] https://docs.hpc.sjtu.edu.cn/app/ai/pytorch.html

#### 环境配置

##### 思源一号

思源1号集群中可供使用的gpu队列是a100

1. 登录

   ```
   ssh username@sylogin.hpc.sjtu.edu.cn
   ```

2. 加载模块

   ```
   module load miniconda3/4.10.3
   module load cuda/11.3.1
   ```

3. 创建环境

   ```
   conda create -n medical python=3.7.11 //创建一个名为medical的python版本为3.7.11的conda环境
   
   source activate medical //登入环境
   ```

4. 安装python包

   首先安装pytorch，其次是其它包

   ```
   pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html --no-cache
   pip install -r requirements.txt
   ```

至此，环境创建完毕，之后每次登录后，只需要加载模块进入环境便可提交任务运行代码：

```
module load miniconda3/4.10.3
module load cuda/11.3.1
source activate medical

sbatch run_cmeee.sbatch //提交任务
```

##### $\pi$集群    

$\pi$集群中可供使用的gpu队列是dgx2

1. 登录

   ```
   ssh username@login.hpc.sjtu.edu.cn
   ```

2. 加载模块

   ```
   module load miniconda3/4.8.2
   module load cuda/10.2.89-gcc-8.3.0
   ```

3. 创建环境

   ```
   conda create -n medical python=3.7.11 //创建一个名为medical的python版本为3.7.11的conda环境
   
   source activate medical //登入环境
   ```

4. 安装python包

   首先安装pytorch，其次是其它包

   ```
   pip install torch==1.9.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html --no-cache
   pip install -r requirements.txt
   ```

至此，环境创建完毕，之后每次登录后，只需要加载模块进入环境便可提交任务运行代码：

```
module load miniconda3/4.8.2
module load cuda/10.2.89-gcc-8.3.0
source activate medical

sbatch run_cmeee.sbatch //提交任务
```

#### 数据传输

$\pi$集群

```shell
scp [-r] [file] username@data.hpc.sjtu.edu.cn:~/ #将本地file传送到集群
scp [-r] username@data.hpc.sjtu.edu.cn:~/file ./ #将集群file传送至本地
```

思源一号

```shell
scp [-r] [file] username@sydata.hpc.sjtu.edu.cn:~/ #将本地file传送到集群
scp [-r] username@sydata.hpc.sjtu.edu.cn:~/file ./ #将集群file传送至本地
```

#### 任务提交

sbatch run_cmeee.sbatch

```shell
#!/bin/bash
#SBATCH --job-name=run_cmeee 				//任务名
#SBATCH --partition=a100					//队列名, 两个gpu队列为a100,dgx2
#SBATCH -N 1								//申请一个节点
#SBATCH -n 1								//任务仅有一个进程
#SBATCH --ntasks-per-node=1					//每个节点上有一个任务
#SBATCH --gres=gpu:1						//申请一张gpu卡
#SBATCH --mem=10G							//申请10G内存
#SBATCH --output=../logs/run_cmeee-%A.log	//log文件地址
#SBATCH --error=../logs/run_cmeee-%A.log

python ...
```

#### 交互式作业

srun -N 1 -n 1 -p a100 --gres=gpu:1 --pty /bin/bash

可以申请得到一张gpu卡，之后就可以像在本地一样操作，这种方式适合debug。

